"""Pure lifecycle around delegated official training and external audit evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import uuid

from repro_integration.critic_snapshot import build_snapshot
from repro_integration.trainer_hook import compute_action_credits, inject_auxiliary
from repro_integration.trace_schema import content_hash, validate_trajectory
from repro_integration.window_store import WindowStore

from .common import atomic_json, value


class ExternalRuntime:
    def __init__(self, config):
        self.config = config
        self.run_id = value(config, "repro.run_id")
        self.config_hash = value(config, "repro.config_hash")
        self.model_contract_hash = value(config, "repro.model_contract_hash")
        self.enabled = bool(value(config, "repro.enabled", True))
        self.output = Path(value(config, "repro.output_dir"))
        self.output.mkdir(parents=True, exist_ok=True)
        self.max_budget = int(value(config, "env.max_steps"))
        self.group_size = int(value(config, "env.rollout.n"))
        self.beta = float(value(config, "repro.beta", 1))
        self.lambda_ = float(value(config, "repro.lambda", .5))
        self.window = WindowStore(self.run_id, self.config_hash,
            max_budget=self.max_budget, window_iterations=int(value(config, "repro.window_iterations", 4)))
        self.snapshot = None
        self.current = []
        self.phase = "idle"
        self.optimizer_steps = 0
        self.iteration_id = None
        if not self.model_contract_hash:
            raise ValueError("External RePro requires an immutable model/protocol/config contract hash")

    def start_iteration(self, iteration_id):
        if self.phase not in {"idle", "committed"}:
            raise RuntimeError("A previous RePro actor update is still pending")
        self.iteration_id = int(iteration_id)
        self.snapshot = build_snapshot(self.window, self.iteration_id,
            method="repro_transfer" if self.enabled else "gigpo",
            min_trajectories=int(value(self.config, "repro.min_trajectories", 5)),
            min_donor_tasks=int(value(self.config, "repro.min_donor_tasks", 2)),
            kappa=float(value(self.config, "repro.kappa", 8)))
        self.context = dict(run_id=self.run_id, iteration_id=self.iteration_id,
            policy_version=f"{self.model_contract_hash}:update-{self.iteration_id - 1}")
        self.current = []
        self.phase = "collecting"
        atomic_json(self.output / "snapshots" / f"iteration_{self.iteration_id}.json", self.snapshot.metadata)
        return dict(self.context)

    def after_collection(self, traces):
        if self.phase != "collecting":
            raise RuntimeError("Native collection completed outside a frozen iteration")
        self.current = [validate_trajectory(trace, max_budget=self.max_budget) for trace in traces]
        self.credits = compute_action_credits(self.snapshot, self.current,
            expected_group_size=self.group_size, lambda_=self.lambda_, enabled=self.enabled, beta=self.beta)
        if not self.current:
            raise ValueError("No unique original trajectories were collected")
        payload = "".join(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n" for trace in self.current).encode("utf-8")
        path = self.output / "train_traces.jsonl"
        with path.open("ab") as handle:
            start = handle.tell()
            handle.write(payload)
            end = handle.tell()
        self.raw_log_reference = dict(path=str(path.resolve()), start_byte=start, end_byte=end,
            iteration_id=self.iteration_id, run_id=self.run_id, encoding="utf-8")
        self.phase = "collected"

    def inject(self, batch):
        if self.phase != "collected":
            raise RuntimeError("Auxiliary credit must be injected once immediately before the native actor")
        import torch
        native = batch.batch["advantages"]
        if not torch.isfinite(native).all():
            raise FloatingPointError("Native actor advantages are nonfinite")
        original = native.clone()
        injection = inject_auxiliary(batch, self.credits, enabled=self.enabled, beta=self.beta)
        batch.batch["repro_aux_delta"] = batch.batch["advantages"] - original
        batch.meta_info["repro_external_audit_context"] = dict(output_dir=str(self.output.resolve()),
            run_id=self.run_id, iteration_id=self.iteration_id, snapshot_id=self.snapshot.snapshot_id)
        self.injection = injection
        self.phase = "injected"
        return injection

    def commit(self, output):
        if self.phase != "injected":
            raise RuntimeError("Actor completion has no matching pending iteration")
        audit = output.meta_info.get("repro_external_audit", {})
        if (audit.get("run_id") != self.run_id or audit.get("iteration_id") != self.iteration_id
                or audit.get("snapshot_id") != self.snapshot.snapshot_id):
            raise RuntimeError("Native actor audit belongs to a different frozen run/iteration/snapshot")
        steps = audit.get("actual_steps_min", 0)
        if (not audit.get("all_ranks_finite", False) or steps <= 0
                or steps != audit.get("actual_steps_max")
                or steps != audit.get("expected_steps_min")
                or steps != audit.get("expected_steps_max")):
            raise RuntimeError("Native actor update lacks matching finite successful optimizer steps on every rank")
        self.window.commit_iteration(self.iteration_id, self.current, update_succeeded=True,
            policy_version=self.context["policy_version"], raw_log_reference=self.raw_log_reference)
        self.optimizer_steps += int(steps)
        self.phase = "committed"
        output.meta_info["metrics"].update({"repro/" + key: number
            for key, number in self.credits.diagnostics.items() if isinstance(number, (int, float))})
        output.meta_info["metrics"].update({"repro/inject_" + key: number
            for key, number in self.injection.items() if isinstance(number, (int, float, bool))})
        atomic_json(self.output / "diagnostics" / f"iteration_{self.iteration_id}.json",
            dict(context=self.context, snapshot_id=self.snapshot.snapshot_id, audit=audit,
                 credit=self.credits.diagnostics, injection=self.injection,
                 metrics=output.meta_info.get("metrics", {}),
                 completed_iterations=self.window.completed_iterations,
                 optimizer_steps_per_rank=self.optimizer_steps))

    def eval_context(self, iteration_id):
        return dict(run_id=self.run_id, iteration_id=int(iteration_id),
                    policy_version=f"{self.model_contract_hash}:update-{int(iteration_id)}")

    def record_evaluation(self, traces, context):
        validated = [validate_trajectory(trace, max_budget=self.max_budget) for trace in traces]
        if any(trace["split"] not in {"valid_seen", "valid_unseen"} or
               any(trace[key] != item for key, item in context.items()) for trace in validated):
            raise ValueError("Evaluation trace has a different run/policy/split")
        path = self.output / "eval_traces" / f"iteration_{context['iteration_id']}.{uuid.uuid4().hex}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n" for trace in validated), encoding="utf-8")

    def save(self, folder):
        if self.phase != "committed":
            raise RuntimeError("Cannot save a pending external RePro update")
        import numpy as np
        import torch
        folder = Path(folder)
        self.window.runtime_state.update(optimizer_steps=self.optimizer_steps,
            model_contract_hash=self.model_contract_hash, iteration_id=self.iteration_id,
            snapshot_id=self.snapshot.snapshot_id, enabled=self.enabled)
        self.window.save(folder / "repro_window.json")
        rng_path = folder / "repro_driver_rng.pt"
        temporary = folder / "repro_driver_rng.pt.tmp"
        torch.save(dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()), temporary)
        temporary.replace(rng_path)
        atomic_json(folder / "REPRO_EXTERNAL_COMPLETE.json", dict(run_id=self.run_id,
            config_hash=self.config_hash, model_contract_hash=self.model_contract_hash,
            iteration_id=self.iteration_id, enabled=self.enabled,
            window_sha256=content_hash(self.window.state_dict()),
            driver_rng_sha256=hashlib.sha256(rng_path.read_bytes()).hexdigest()))

    def load(self, folder):
        import numpy as np
        import torch
        folder = Path(folder)
        marker = json.loads((folder / "REPRO_EXTERNAL_COMPLETE.json").read_text())
        for name in ("run_id", "config_hash", "model_contract_hash", "enabled"):
            if marker.get(name) != getattr(self, name):
                raise ValueError("External checkpoint belongs to another run/model/protocol/config")
        window = WindowStore.load(folder / "repro_window.json", expected_run_id=self.run_id,
                                  expected_config_hash=self.config_hash)
        if content_hash(window.state_dict()) != marker["window_sha256"]:
            raise ValueError("External checkpoint window digest mismatch")
        if (window.max_budget != self.max_budget or window.window_iterations != self.window.window_iterations
                or not window.completed_iterations or window.completed_iterations[-1] != marker["iteration_id"]):
            raise ValueError("External checkpoint iteration/budget/window differs")
        if folder.name != f"global_step_{marker['iteration_id']}":
            raise ValueError("Native actor checkpoint and external window iteration differ")
        rng_path = folder / "repro_driver_rng.pt"
        if hashlib.sha256(rng_path.read_bytes()).hexdigest() != marker["driver_rng_sha256"]:
            raise ValueError("External driver RNG checkpoint digest mismatch")
        rng = torch.load(rng_path, weights_only=False, map_location="cpu")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        self.window = window
        self.optimizer_steps = window.runtime_state["optimizer_steps"]
        self.iteration_id = marker["iteration_id"]
        self.phase = "idle"
