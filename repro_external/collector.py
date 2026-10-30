"""Delegate native collection; add identities only at the existing gather boundary."""
from __future__ import annotations

import copy
from pathlib import Path

from repro_integration.alfworld_adapter import task_uid
from repro_integration.trace_schema import json_copy, validate_trajectory


class ObservationProxy:
    def __init__(self, native):
        self.native = native
        self.initial_infos = None
        self.step_infos = []
        self.generated_text = []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def reset(self, *args, **kwargs):
        result = self.native.reset(*args, **kwargs)
        self.initial_infos = copy.deepcopy(result[1])
        self.step_infos = []
        self.generated_text = []
        return result

    def step(self, text_actions, *args, **kwargs):
        result = self.native.step(text_actions, *args, **kwargs)
        self.step_infos.append(copy.deepcopy(result[3]))
        self.generated_text.append(list(text_actions))
        return result


def actual_split(gamefile):
    splits = [part for part in Path(gamefile).parts if part in {"train", "valid_seen", "valid_unseen"}]
    if not splits:
        raise ValueError("Cannot verify ALFWorld split from the actual native gamefile")
    return splits[-1]


def annotate_gather_rows(total_batch_list, capture, context, *, max_steps, is_train):
    if capture.initial_infos is None or len(capture.initial_infos) != len(total_batch_list):
        raise ValueError("Native reset/gather environment counts disagree")
    traces = []
    for env_index, rows in enumerate(total_batch_list):
        initial = capture.initial_infos[env_index]
        gamefile = initial["extra.gamefile"]
        split = actual_split(gamefile)
        if (is_train and split != "train") or (not is_train and split not in {"valid_seen", "valid_unseen"}):
            raise ValueError("Actual native gamefile disagrees with training/evaluation mode")
        active_rows = [(time, row) for time, row in enumerate(rows) if bool(row["active_masks"])]
        if not active_rows:
            raise ValueError("Native trajectory has no active decision")
        first = active_rows[0][1]
        trace = dict(context, task_uid=task_uid(gamefile), task_family=initial["repro_task_family"],
            group_uid=str(first["uid"]), trajectory_uid=str(first["traj_uid"]),
            split=split, gamefile=gamefile, seed=int(initial["repro_env_seed"]), steps=[])
        for step_id, (time, row) in enumerate(active_rows):
            if str(row["uid"]) != trace["group_uid"] or str(row["traj_uid"]) != trace["trajectory_uid"]:
                raise ValueError("Native gather mixed trajectory/group identity")
            info = capture.step_infos[time][env_index]
            record = json_copy(info["repro_step"])
            if record["step_id"] != step_id:
                raise ValueError("Native active decisions and critic annotation counts disagree")
            action_uid = f"{trace['run_id']}:{trace['trajectory_uid']}:{step_id}"
            row["repro_action_uid"] = action_uid
            response = row["responses"]
            response_length = response.shape[-1]
            record.update(action_uid=action_uid,
                response_ids=response.detach().cpu().tolist(),
                response_mask=row["attention_mask"][-response_length:].detach().cpu().tolist(),
                response_mask_source="official_collector_attention_mask",
                generated_text=capture.generated_text[time][env_index])
            # The native worker exposes TextWorld's score; the native vector
            # environment independently maps won to its training reward.
            if "original_environment_reward" in record:
                record["original_textworld_score"] = record["original_environment_reward"]
            record["official_environment_reward"] = float(row["rewards"])
            if "is_action_valid" in row:
                record["official_is_action_valid"] = bool(row["is_action_valid"])
            if "step_progress" in row:
                record["official_step_progress"] = float(row["step_progress"])
            trace["steps"].append(record)
        final = trace["steps"][-1]
        trace["success"] = bool(final["reward_rel"])
        trace["end_reason"] = ("success" if trace["success"] else "failure" if final["terminated"]
            else "horizon" if final["budget_after"] == 0 else "collector_truncation")
        traces.append(validate_trajectory(trace, max_budget=max_steps))
    return traces


def make_external_collector(native_collector):
    class ExternalCollector(native_collector):
        external_runtime = None

        def multi_turn_loop(self, *args, **kwargs):
            if self.external_runtime is None:
                return super().multi_turn_loop(*args, **kwargs)
            # The pinned trainer calls this public method using keyword arguments.
            if args:
                raise TypeError("External collection requires the native keyword call contract")
            runtime = self.external_runtime
            is_train = bool(kwargs.get("is_train", True))
            if kwargs.get("rollout_mode", "vanilla") != "vanilla":
                raise ValueError("External trace adapter supports native vanilla collection")
            context = runtime.start_iteration(self.external_trainer.global_steps) if is_train else runtime.eval_context(self.external_trainer.global_steps)
            capture = ObservationProxy(kwargs["envs"])
            self._external_collection = (capture, context, is_train)
            try:
                result = super().multi_turn_loop(**dict(kwargs, envs=capture))
                if is_train:
                    runtime.after_collection(self.last_external_traces)
                else:
                    runtime.record_evaluation(self.last_external_eval_traces, context)
                return result
            finally:
                self._external_collection = None

        def gather_rollout_data(self, total_batch_list, *args, **kwargs):
            pending = getattr(self, "_external_collection", None)
            if pending is not None:
                capture, context, is_train = pending
                traces = annotate_gather_rows(total_batch_list, capture, context,
                    max_steps=self.external_runtime.max_budget, is_train=is_train)
            result = super().gather_rollout_data(total_batch_list, *args, **kwargs)
            if pending is not None:
                if is_train:
                    self.last_external_traces = traces
                else:
                    self.last_external_eval_traces = traces
            return result

    return ExternalCollector
