"""Read-only audits around the unchanged official actor update method."""
from __future__ import annotations

import functools
import math
from numbers import Real
from pathlib import Path

from .common import atomic_json


def optimizer_step_bounds(optimizer):
    steps = []
    for state in optimizer.state.values():
        if "step" in state:
            item = state["step"]
            item = item.item() if hasattr(item, "item") else item
            if not isinstance(item, Real) or not math.isfinite(item) or int(item) != item:
                raise ValueError("Optimizer state has a nonfinite/noninteger step")
            steps.append(int(item))
    return (min(steps), max(steps)) if steps else (0, 0)


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, Real):
        return math.isfinite(value)
    return True


def aggregate_audit(local, *, device=None):
    import torch
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        backend = torch.distributed.get_backend()
        device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
    else:
        device = torch.device("cpu")
    data = torch.tensor([local["actual_steps"], local["expected_steps"], int(local["finite"])],
                        dtype=torch.int64, device=device)
    lower, upper = data.clone(), data.clone()
    if distributed:
        torch.distributed.all_reduce(lower, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(upper, op=torch.distributed.ReduceOp.MAX)
    return dict(actual_steps_min=int(lower[0]), actual_steps_max=int(upper[0]),
                expected_steps_min=int(lower[1]), expected_steps_max=int(upper[1]),
                all_ranks_finite=bool(lower[2]),
                world_size=torch.distributed.get_world_size() if distributed else 1)


def make_audited_actor_worker(native_worker):
    class AuditedActorWorker(native_worker):
        @functools.wraps(native_worker.update_actor)
        def update_actor(self, data):
            import torch
            audit_context = data.meta_info.get("repro_external_audit_context")
            if audit_context is None:
                return super().update_actor(data)
            before = optimizer_step_bounds(self.actor_optimizer)
            mini_batch_size = int(self.config.actor.ppo_mini_batch_size)
            expected = math.ceil(len(data) / mini_batch_size) * int(self.config.actor.ppo_epochs)
            local = dict(rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                before_steps=list(before), expected_steps=expected,
                run_id=audit_context["run_id"], iteration_id=audit_context["iteration_id"],
                snapshot_id=audit_context["snapshot_id"],
                nonzero_aux_tokens_at_worker_input=int(torch.count_nonzero(data.batch["repro_aux_delta"]).item()))
            # Inputs, model, loss, gradients, optimizer and native output all use
            # the original implementation. No sampling or PPO source is replaced.
            output = super().update_actor(data)
            after = optimizer_step_bounds(self.actor_optimizer)
            local.update(after_steps=list(after), actual_steps=after[1] - before[1],
                finite=finite_tree(output.meta_info.get("metrics", {}))
                    and before[0] == before[1] and after[0] == after[1])
            aggregate = aggregate_audit(local)
            aggregate.update(run_id=local["run_id"], iteration_id=local["iteration_id"],
                             snapshot_id=local["snapshot_id"])
            local["aggregate"] = aggregate
            path = Path(audit_context["output_dir"]) / "actor_audit" / (
                f"iteration_{local['iteration_id']}.rank_{local['rank']}.json")
            atomic_json(path, local)
            output.meta_info["repro_external_audit"] = aggregate
            output.meta_info["metrics"].update({"repro_audit/" + key: number
                for key, number in aggregate.items() if isinstance(number, (int, float, bool))})
            return output

    return AuditedActorWorker

