"""Add critic facts to native ALFWorld info without changing environment behavior."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os

from repro_integration.alfworld_adapter import AlfworldAnnotator


def make_facts_wrapper(native_infos):
    class CriticFactsInfos(native_infos):
        def load(self, *args, **kwargs):
            result = super().load(*args, **kwargs)
            self._repro_gamefile = os.path.realpath(args[0])
            with open(os.path.join(os.path.dirname(self._repro_gamefile), "traj_data.json"), encoding="utf-8") as handle:
                metadata = json.load(handle)
            self._repro_task = {key: metadata[key] for key in ("task_type", "pddl_params", "task_id") if key in metadata}
            return result

        def _repro_info(self, state):
            state["extra.repro"] = dict(schema="alfworld_facts_v1", gamefile=self._repro_gamefile,
                task=self._repro_task,
                entity_names={str(info.id): str(info.name).strip() for info in self._entity_infos.values()},
                facts=[dict(name=str(fact.name), arguments=[dict(name=str(arg.name), type=str(arg.type))
                    for arg in fact.arguments]) for fact in state["facts"]])
            return state

        def reset(self, *args, **kwargs):
            return self._repro_info(super().reset(*args, **kwargs))

        def step(self, *args, **kwargs):
            state, reward, done = super().step(*args, **kwargs)
            return self._repro_info(state), reward, done

    return CriticFactsInfos


@contextmanager
def critic_info_scope(base_env):
    """The original init_env constructs its wrappers and registry as usual."""
    namespace = base_env.init_env.__func__.__globals__
    native_infos = namespace["AlfredInfos"]
    textworld = namespace["textworld"]
    native_request = textworld.EnvInfos

    class CriticEnvInfos(native_request):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.facts = True
            self.lost = True
            self.extras = list(self.extras)
            if "repro" not in self.extras:
                self.extras.append("repro")

    namespace["AlfredInfos"] = make_facts_wrapper(native_infos)
    textworld.EnvInfos = CriticEnvInfos
    try:
        yield
    finally:
        namespace["AlfredInfos"] = native_infos
        textworld.EnvInfos = native_request


def make_external_worker(native_worker, *, max_steps):
    class CriticOnlyAlfworldWorker(native_worker):
        def __init__(self, config, seed, base_env):
            self._repro_seed = int(seed)
            self._repro_active = False
            with critic_info_scope(base_env):
                super().__init__(config, seed, base_env)

        def reset(self, *args, **kwargs):
            obs, infos = super().reset(*args, **kwargs)
            flat = {key: item[0] for key, item in infos.items()}
            self._repro_annotator = AlfworldAnnotator(flat["extra.gamefile"], max_steps=max_steps)
            state = self._repro_annotator.reset(obs[0], flat)
            self._repro_active = True
            infos["repro_state"] = [state]
            infos["repro_task_family"] = [self._repro_annotator.family]
            infos["repro_env_seed"] = [self._repro_seed]
            return obs, infos

        def step(self, action):
            # Preserve native stepping even for rows already marked done. The
            # original collector owns active masks; only active rows are logged.
            obs, scores, dones, infos = super().step(action)
            infos["repro_task_family"] = [self._repro_annotator.family]
            infos["repro_env_seed"] = [self._repro_seed]
            if self._repro_active:
                flat = {key: item[0] for key, item in infos.items()}
                record = self._repro_annotator.step(action, obs[0], flat,
                    done=bool(dones[0]), original_reward=float(scores[0]))
                infos["repro_step"] = [record]
                self._repro_active = not bool(dones[0]) and record["budget_after"] > 0
            else:
                infos["repro_inactive"] = [True]
            return obs, scores, dones, infos

    return CriticOnlyAlfworldWorker

