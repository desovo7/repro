"""Committed training history, isolated from pending PPO updates and retries."""

from __future__ import annotations

import json
from pathlib import Path

from .trace_schema import content_hash, json_copy, validate_trajectory


COMPACT_RECORD_FORMAT = "critic_history_without_duplicate_raw_facts_v2"
DROPPED_STEP_FIELDS = ("critic_before", "critic_after")


def compact_for_window(trace, *, max_budget=50):
    """Retain every scientific/token field, dropping only duplicate raw facts."""
    result = validate_trajectory(trace, max_budget=max_budget)
    for step in result["steps"]:
        for field in DROPPED_STEP_FIELDS:
            step.pop(field, None)
    return result


def raw_index_digest(hashes):
    return content_hash([dict(trajectory_uid=uid, raw_sha256=hashes[uid]) for uid in sorted(hashes)])


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


class WindowStore:
    FORMAT_VERSION = 2

    def __init__(self, run_id: str, config_hash: str, *, window_iterations: int = 4,
                 max_budget: int = 50):
        if not run_id or not config_hash:
            raise ValueError("Run and full model/protocol/config hash are required")
        if type(window_iterations) is not int or window_iterations < 1:
            raise ValueError("window_iterations must be positive")
        self.run_id = run_id
        self.config_hash = config_hash
        self.window_iterations = window_iterations
        self.max_budget = max_budget
        self._iterations = {}
        self._committed_digests = {}
        self._compact_digests = {}
        self._raw_trajectory_hashes = {}
        self._compact_trajectory_hashes = {}
        self._raw_log_references = {}
        self._trajectory_owner = {}
        self._action_owner = {}
        self.runtime_state = {}  # Caller-owned JSON RNG/budget/checkpoint metadata.

    @property
    def completed_iterations(self):
        return tuple(sorted(self._iterations))

    def commit_iteration(self, iteration_id, trajectories, *, update_succeeded,
                         policy_version=None, raw_log_reference=None):
        if update_succeeded is not True:
            raise ValueError("Pending or failed actor updates cannot enter the history window")
        if type(iteration_id) is not int or iteration_id < 0:
            raise ValueError("Invalid iteration")
        log_reference = json_copy(raw_log_reference)
        unique, raw_hashes, compact_hashes = {}, {}, {}
        for raw in trajectories:
            raw_hash = content_hash(raw)
            trace = compact_for_window(raw, max_budget=self.max_budget)
            if trace["run_id"] != self.run_id or trace["split"] != "train":
                raise ValueError("Only this run's training split may enter history")
            if trace["iteration_id"] != iteration_id:
                raise ValueError("Cannot commit mixed iterations")
            if policy_version is not None and trace["policy_version"] != policy_version:
                raise ValueError("Policy version mismatch")
            uid = trace["trajectory_uid"]
            if uid in unique and (raw_hashes[uid] != raw_hash or unique[uid] != trace):
                raise ValueError("Conflicting duplicate trajectory")
            unique[uid] = trace
            raw_hashes[uid] = raw_hash
            compact_hashes[uid] = content_hash(trace)
        if not unique:
            raise ValueError("Cannot commit an empty iteration")
        traces = [unique[uid] for uid in sorted(unique)]
        if len({trace["policy_version"] for trace in traces}) != 1:
            raise ValueError("One iteration must use one sampling policy snapshot")
        digest = raw_index_digest(raw_hashes)
        if iteration_id in self._committed_digests:
            if self._committed_digests[iteration_id] != digest:
                raise ValueError("Retry changed an already committed iteration")
            return False
        if self._committed_digests and iteration_id <= max(self._committed_digests):
            raise ValueError("Cannot append historical data out of order")
        actions = {}
        for trace in traces:
            uid = trace["trajectory_uid"]
            if uid in self._trajectory_owner:
                raise ValueError("Trajectory identity reused across iterations")
            for step in trace["steps"]:
                action_uid = step["action_uid"]
                if action_uid in actions or action_uid in self._action_owner:
                    raise ValueError("Action identity collision")
                actions[action_uid] = [iteration_id, uid, step["step_id"]]
        # No mutation precedes validation of every record.
        self._iterations[iteration_id] = traces
        self._committed_digests[iteration_id] = digest
        self._compact_digests[iteration_id] = content_hash(traces)
        self._raw_trajectory_hashes.update(raw_hashes)
        self._compact_trajectory_hashes.update(compact_hashes)
        self._raw_log_references[iteration_id] = log_reference
        self._trajectory_owner.update({t["trajectory_uid"]: iteration_id for t in traces})
        self._action_owner.update(actions)
        cutoff = iteration_id - self.window_iterations + 1
        self._iterations = {i: rows for i, rows in self._iterations.items() if i >= cutoff}
        return True

    def completed_for(self, iteration_id):
        if any(i >= iteration_id for i in self._iterations):
            raise ValueError("Cannot reconstruct a historical snapshot after future data was committed")
        cutoff = iteration_id - self.window_iterations
        return json_copy([trace for i in sorted(self._iterations) if cutoff <= i < iteration_id
                          for trace in self._iterations[i]])

    def provenance_for(self, iteration_id):
        cutoff = iteration_id - self.window_iterations
        selected = sorted(i for i in self._iterations if cutoff <= i < iteration_id)
        uids = sorted(t["trajectory_uid"] for i in selected for t in self._iterations[i])
        return json_copy(dict(format_version=self.FORMAT_VERSION, record_format=COMPACT_RECORD_FORMAT,
            dropped_step_fields=list(DROPPED_STEP_FIELDS), raw_hash_kind="canonical_json_sha256",
            raw_iteration_digest_kind="canonical_sorted_trajectory_raw_hash_index_sha256",
            trajectory_hashes={uid: dict(raw_sha256=self._raw_trajectory_hashes[uid],
                                         compact_sha256=self._compact_trajectory_hashes[uid]) for uid in uids},
            raw_iteration_digests={str(i): self._committed_digests[i] for i in selected},
            compact_iteration_digests={str(i): self._compact_digests[i] for i in selected},
            raw_log_references={str(i): self._raw_log_references.get(i) for i in selected}))

    def state_dict(self):
        return json_copy(dict(format_version=self.FORMAT_VERSION, run_id=self.run_id,
                              record_format=COMPACT_RECORD_FORMAT, dropped_step_fields=list(DROPPED_STEP_FIELDS),
                              raw_hash_kind="canonical_json_sha256",
                              raw_iteration_digest_kind="canonical_sorted_trajectory_raw_hash_index_sha256",
                              config_hash=self.config_hash, window_iterations=self.window_iterations,
                              max_budget=self.max_budget,
                              iterations={str(i): rows for i, rows in self._iterations.items()},
                              committed_digests={str(i): h for i, h in self._committed_digests.items()},
                              compact_digests={str(i): h for i, h in self._compact_digests.items()},
                              raw_trajectory_hashes=self._raw_trajectory_hashes,
                              compact_trajectory_hashes=self._compact_trajectory_hashes,
                              raw_log_references={str(i): value for i, value in self._raw_log_references.items()},
                              trajectory_owner=self._trajectory_owner, action_owner=self._action_owner,
                              runtime_state=self.runtime_state))

    @classmethod
    def from_state_dict(cls, state, *, expected_run_id, expected_config_hash):
        state = json_copy(state)
        if (state.get("format_version") != cls.FORMAT_VERSION or state.get("record_format") != COMPACT_RECORD_FORMAT
                or state.get("dropped_step_fields") != list(DROPPED_STEP_FIELDS)
                or state.get("raw_hash_kind") != "canonical_json_sha256"
                or state.get("raw_iteration_digest_kind") != "canonical_sorted_trajectory_raw_hash_index_sha256"):
            raise ValueError("Unsupported history checkpoint format")
        if state.get("run_id") != expected_run_id or state.get("config_hash") != expected_config_hash:
            raise ValueError("History belongs to a different run/model/protocol/config")
        obj = cls(expected_run_id, expected_config_hash,
                  window_iterations=state["window_iterations"], max_budget=state["max_budget"])
        iterations = {int(i): rows for i, rows in state["iterations"].items()}
        digests = {int(i): h for i, h in state["committed_digests"].items()}
        compact_digests = {int(i): h for i, h in state["compact_digests"].items()}
        raw_hashes, compact_hashes = state["raw_trajectory_hashes"], state["compact_trajectory_hashes"]
        owners, action_owners = state["trajectory_owner"], state["action_owner"]
        logs = {int(i): value for i, value in state["raw_log_references"].items()}
        if any(i < 0 for i in digests) or any(not isinstance(uid, str) or not uid for uid in owners):
            raise ValueError("Invalid committed iteration or trajectory identity")
        if set(digests) != set(compact_digests) or set(logs) != set(digests):
            raise ValueError("Raw/compact iteration ledgers disagree")
        if set(owners) != set(raw_hashes) or set(owners) != set(compact_hashes):
            raise ValueError("Trajectory hash bindings disagree with the identity ledger")
        if any(type(i) is not int or i not in digests for i in owners.values()):
            raise ValueError("Trajectory owner points outside committed iterations")
        if not all(valid_digest(h) for h in list(digests.values()) + list(compact_digests.values())
                   + list(raw_hashes.values()) + list(compact_hashes.values())):
            raise ValueError("Invalid raw or compact content digest")
        if digests:
            cutoff = max(digests) - obj.window_iterations + 1
            if set(iterations) != {i for i in digests if i >= cutoff}:
                raise ValueError("Retained history does not match the committed rolling window")
        elif iterations or owners or action_owners:
            raise ValueError("Empty commit ledger contains history identities")
        for i in digests:
            bound = {uid: raw_hashes[uid] for uid, owner in owners.items() if owner == i}
            if not bound or raw_index_digest(bound) != digests[i]:
                raise ValueError("Raw trajectory hash binding failed")
        for uid, owner in action_owners.items():
            if (not isinstance(owner, list) or len(owner) != 3 or owner[0] not in digests
                    or owners.get(owner[1]) != owner[0] or type(owner[2]) is not int or owner[2] < 0):
                raise ValueError("Action ledger points outside a committed trajectory")
        for i, rows in sorted(iterations.items()):
            validated = [validate_trajectory(row, max_budget=obj.max_budget) for row in rows]
            if content_hash(validated) != compact_digests[i]:
                raise ValueError("Compact history checkpoint digest mismatch")
            if {row["trajectory_uid"] for row in validated} != {uid for uid, owner in owners.items() if owner == i}:
                raise ValueError("Retained trajectory identities differ from their raw bindings")
            if len(validated) != len({row["trajectory_uid"] for row in validated}):
                raise ValueError("Duplicate retained trajectory")
            actual_actions = [step["action_uid"] for row in validated for step in row["steps"]]
            if (len(actual_actions) != len(set(actual_actions))
                    or set(actual_actions) != {uid for uid, owner in action_owners.items() if owner[0] == i}):
                raise ValueError("Retained action identities disagree with the global ledger")
            for row in validated:
                uid = row["trajectory_uid"]
                if (row["run_id"] != expected_run_id or row["split"] != "train" or row["iteration_id"] != i
                        or content_hash(row) != compact_hashes[uid]):
                    raise ValueError("Compact trajectory hash/identity mismatch")
                for step in row["steps"]:
                    if any(field in step for field in DROPPED_STEP_FIELDS):
                        raise ValueError("Compact history contains discarded duplicate raw facts")
                    if action_owners.get(step["action_uid"]) != [i, uid, step["step_id"]]:
                        raise ValueError("Retained action ledger mismatch")
            if len({row["policy_version"] for row in validated}) != 1:
                raise ValueError("Retained iteration mixes sampling policies")
            obj._iterations[i] = validated
        obj._committed_digests = digests
        obj._compact_digests = compact_digests
        obj._raw_trajectory_hashes, obj._compact_trajectory_hashes = raw_hashes, compact_hashes
        obj._raw_log_references = logs
        obj._trajectory_owner, obj._action_owner = owners, action_owners
        obj.runtime_state = state.get("runtime_state", {})
        return obj

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state_dict(), sort_keys=True, indent=2) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path, *, expected_run_id, expected_config_hash):
        return cls.from_state_dict(json.loads(Path(path).read_text()),
                                   expected_run_id=expected_run_id,
                                   expected_config_hash=expected_config_hash)
