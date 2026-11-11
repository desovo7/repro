"""Only task assignment and recording differ; all WebShop transitions stay native."""
import copy
import os
import random
from unittest.mock import patch

import numpy as np
import ray
from agent_system.environments.env_package.webshop import envs as native
from common import JVM, canonical, value_hash


class PanelWorker(native.WebshopWorker):
    def __init__(self, seed, env_kwargs):
        import gym
        import torch
        if os.environ.get('JAVA_TOOL_OPTIONS') != JVM:
            raise ValueError('Expected capped JVM')
        original = gym.make
        def compatible(name, **kwargs):
            return original(name, **dict(kwargs, disable_env_checker=True))
        with patch.object(gym, 'make', compatible):
            super().__init__(seed, env_kwargs)
        self.seed = seed
        self.initial_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.price_hash = value_hash(self.env.unwrapped.server.product_prices)
        self.catalogue = [dict(goal_index=i, goal_hash=value_hash(g), instruction=g['instruction_text'])
                          for i, g in enumerate(self.get_goals()[:500])]

    def metadata(self):
        return dict(worker_seed=self.seed, price_hash=self.price_hash, goals=self.catalogue)

    def reset_task(self, task):
        import torch
        if task['worker_seed'] != self.seed or task['price_hash'] != self.price_hash:
            raise ValueError('Worker/price initial condition changed')
        goal = self.catalogue[task['goal_index']]
        if goal['goal_hash'] != task['goal_hash'] or goal['instruction'] != task['instruction']:
            raise ValueError('Goal identity changed')
        state = copy.deepcopy(self.initial_rng)
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        self.env.unwrapped.server.user_sessions.clear()
        obs, info = super().reset(int(task['goal_index']))
        signature = value_hash([obs, info['available_actions']])
        if task.get('initial_signature') and signature != task['initial_signature']:
            raise ValueError('Initial observation/actions changed')
        info = dict(info, panel_initial_signature=signature, panel_goal_hash=goal['goal_hash'])
        return obs, info


class PanelBackend(native.WebshopMultiProcessEnv):
    def __init__(self, config, source):
        self.group_n = 1
        self.env_num = self.num_processes = 128
        self.is_train = False
        self._closed = False
        self._workers = []
        data = source / 'agent_system/environments/env_package/webshop/webshop/data'
        kwargs = dict(observation_mode='text', num_products=None, human_goals=False,
                      file_path=str(data/'items_shuffle_1000.json'), attr_path=str(data/'items_ins_v2_1000.json'))
        actor = ray.remote(num_cpus=0.1, num_gpus=0)(PanelWorker)
        self._workers = [actor.remote(1000+i, kwargs) for i in range(128)]

    def metadata(self):
        return ray.get([w.metadata.remote() for w in self._workers])

    def assign(self, tasks):
        if len(tasks) != 128 or [t['worker_seed'] for t in tasks] != list(range(1000,1128)):
            raise ValueError('Panel requires frozen128worker order')
        self.tasks = tasks

    def reset(self):
        result = ray.get([w.reset_task.remote(t) for w,t in zip(self._workers,self.tasks,strict=True)])
        return [x[0] for x in result], [x[1] for x in result]

    def step(self, actions):
        self.last_actions = list(actions)
        return super().step(actions)

    def __del__(self):
        # Explicit close is owned by evaluator; interpreter teardown can outlive Ray.
        pass
