"""Evaluation-only task assignment; all observations and transitions stay native."""
import ray
from agent_system.environments.env_package.alfworld import envs as alf_native


class FixedTaskWorker(alf_native.AlfworldWorker):
    def __init__(self, base_env):
        self.base_env = base_env
        self.allowed = set(base_env.game_files)
        self.env = None

    def assign_game(self, gamefile, seed):
        assert gamefile in self.allowed
        self.close_environment()
        self.base_env.game_files = [gamefile]
        self.base_env.num_games = 1
        self.env = self.base_env.init_env(batch_size=1)
        self.env.seed(seed)
        return gamefile

    def close_environment(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        return True


class FixedTaskEnvs(alf_native.AlfworldEnvs):
    def __init__(self, base_env, pool_size=32):
        self.multi_modal = False
        self.group_n = 1
        actor = ray.remote(num_cpus=0.1, num_gpus=0)(FixedTaskWorker)
        self.pool = [actor.remote(base_env) for _ in range(pool_size)]
        self.workers = self.pool
        self.num_processes = pool_size
        self.prev_admissible_commands = [None] * pool_size

    def assign(self, tasks):
        assert 0 < len(tasks) <= len(self.pool)
        self.num_processes = len(tasks)
        self.workers = self.pool[:len(tasks)]
        self.prev_admissible_commands = [None] * len(tasks)
        assigned = ray.get([w.assign_game.remote(t['gamefile'], 1000 + t['manifest_index'])
                            for w, t in zip(self.workers, tasks)])
        assert assigned == [t['gamefile'] for t in tasks]

    def close(self):
        ray.get([w.close_environment.remote() for w in self.pool])
        self.workers = self.pool
        super().close()
