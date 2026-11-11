"""Observe the native step exactly once; its projection mutates the input list."""
import numpy as np


def make_recorded_manager(native):
    class RecordedManager(native):
        def reset(self, kwargs):
            result=super().reset(kwargs)
            self.trace_steps=[[] for _ in result[1]]
            self.last_observations=result[0]['anchor']
            self.last_infos=result[1]
            self.initial_signatures=[i['panel_initial_signature'] for i in result[1]]
            return result

        def step(self, text_actions):
            generated=list(text_actions)
            previous_obs=self.last_observations
            previous_infos=self.last_infos
            result=super().step(text_actions)
            obs,rewards,dones,infos=result
            actions=self.envs.last_actions
            for i in range(len(generated)):
                self.trace_steps[i].append(dict(action=actions[i],generated_text=generated[i],
                    observation_before=str(previous_obs[i]),observation_after=str(obs['anchor'][i]),
                    available_before=previous_infos[i]['available_actions'],
                    environment_done=bool(np.asarray(dones).reshape(-1)[i]),
                    won=bool(infos[i]['won']),task_score=float(infos[i]['task_score']),
                    native_reward=float(np.asarray(rewards).reshape(-1)[i]),
                    format_valid=bool(infos[i]['is_action_valid'])))
            self.last_observations=obs['anchor'];self.last_infos=infos
            return result
    return RecordedManager
