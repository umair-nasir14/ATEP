# Code is modified from:
# https://github.com/uber-research/poet/blob/master/poet_distributed/
# under Apache license 

from ..core import Niche
from .model import Model, simulate
from .env import bipedhard_custom, Env_config, make_env
from collections import OrderedDict

from neat.activations import tanh


DEFAULT_ENV = Env_config(
        name='default_env',
        ground_roughness=0,
        pit_gap=[],
        stump_width=[],
        stump_height=[],
        stump_float=[],
        stair_height=[],
        stair_width=[],
        stair_steps=[])

class Box2DNiche(Niche):
    def __init__(self, env_configs, env_params, seed, init='random', stochastic=False):

        
        self.model = Model(bipedhard_custom)
        if not isinstance(env_configs, list):
            env_configs = [env_configs]
        self.env_configs = OrderedDict()
        for env in env_configs:
            self.env_configs[env.name] = env
        self.env_params = env_params
        self.seed = seed
        self.stochastic = stochastic
        self.model.make_env(seed=seed, env_config=DEFAULT_ENV)
        self.init = init

    def __getstate__(self):
        return {"env_configs": self.env_configs,
                "env_params": self.env_params,
                "seed": self.seed,
                "stochastic": self.stochastic,
                "init": self.init,
                }

    def __setstate__(self, state):
        
        self.model = Model(bipedhard_custom)
        self.env_configs = state["env_configs"]
        self.env_params = state["env_params"]
        self.seed = state["seed"]
        self.stochastic = state["stochastic"]
        self.model.make_env(seed=self.seed, env_config=DEFAULT_ENV)
        self.init = state["init"]


    def add_env(self, env):
        env_name = env.name
        assert env_name not in self.env_configs.keys()
        self.env_configs[env_name] = env

    def delete_env(self, env_name):
        assert env_name in self.env_configs.keys()
        self.env_configs.pop(env_name)

    def initial_theta(self):
        if self.init == 'random':
            return self.model.get_random_model_params()
        elif self.init == 'zeros':
            import numpy as np
            return np.zeros(self.model.param_count)
        else:
            raise NotImplementedError(
                'Undefined initialization scheme `{}`'.format(self.init))

    def rollout(self, genome, random_state, eval=False):
        
        from atep.atep import eval_a_genome

        total_fitness = 0
        total_length = 0
        if self.stochastic:
            seed = random_state.randint(1000000)
        else:
            seed = self.seed
        for env_config in self.env_configs.values():
            fitnesses = eval_a_genome(
                self.model, seed=seed, env_config=env_config, env_params=self.env_params)
            total_fitness += fitnesses
            
        return total_fitness / len(self.env_configs)
