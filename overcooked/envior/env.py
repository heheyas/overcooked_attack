import random
import gym
import numpy as np
from typing import Union
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env import SubprocVecEnv
from human_aware_rl.rllib.utils import get_base_ae, get_required_arguments
from overcooked_ai_py.mdp.actions import Action

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

class OvercookedMultiAgent(gym.Env):
    
    supported_agents = ['ppo', 'bc']
    bc_schedule = self_play_bc_schedule = [(0, 0), (float('inf'), 0)]
    DEFAULT_CONFIG = {
        "mdp_params" : {
            "layout_name" : "cramped_room",
            "rew_shaping_params" : {}
        },
        "env_params" : {
            "horizon" : 400
        },
        "multi_agent_params" : {
            "reward_shaping_factor" : 0.0,
            "reward_shaping_horizon" : 0,
            "bc_schedule" : self_play_bc_schedule,
            "use_phi" : True
        }
    }
    
    def __init__(self, base_env, use_phi=True, reward_shaping_factor=0.0, reward_shaping_horizon=0, bc_schedule=None, fixed_role=False):
        
        if bc_schedule:
            self.bc_schedule = bc_schedule
        self._validate_schedule(self.bc_schedule)
        
        self.base_env = base_env
        self.featurize_fn_map = {
            "ppo": lambda state: self.base_env.lossless_state_encoding_mdp(state),
            "bc": lambda state: self.base_env.featurize_state_mdp(state)
        }
        self._validate_featurize_fns(self.featurize_fn_map)
        self._initial_reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_horizon = reward_shaping_horizon
        self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.use_phi = use_phi
        self.anneal_bc_factor(0)
        self.fixed_role = fixed_role
        self.reset()
        
    def _validate_schedule(self, schedule):
        timesteps = [p[0] for p in schedule]
        values = [p[1] for p in schedule]
        
        assert len(schedule) >= 2, "Need at least 2 points to linearly interpolate schedule"
        assert schedule[0][0] == 0, "Schedule must start at timestep 0"
        assert all([t >=0 for t in timesteps]), "All timesteps in schedule must be non-negative"
        assert all([v >=0 and v <= 1 for v in values]), "All values in schedule must be between 0 and 1"
        assert sorted(timesteps) == timesteps, "Timesteps must be in increasing order in schedule"

        # To ensure we flatline after passing last timestep
        if (schedule[-1][0] < float('inf')):
            schedule.append((float('inf'), schedule[-1][1]))
        
    def _validate_featurize_fns(self, mapping):
        assert 'ppo' in mapping, "At least one ppo agent must be specified"
        for k, v in mapping.items():
            assert k in self.supported_agents, "Unsuported agent type in featurize mapping {0}".format(k)
            assert callable(v), "Featurize_fn values must be functions"
            assert len(get_required_arguments(v)) == 1, "Featurize_fn value must accept exactly one argument"
        
    def _setup_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()
        
        featurize_fn_ppo = lambda state: self.base_env.lossless_state_encoding_mdp(state)
        obs_shape = featurize_fn_ppo(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        self.ppo_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)
        self.observation_space = self.ppo_observation_space
        
        featurize_fn_bc = lambda state: self.base_env.featurize_state_mdp(state)
        obs_shape = featurize_fn_bc(dummy_state)[0].shape
        high = np.ones(obs_shape) * 100
        low = np.ones(obs_shape) * -100
        self.bc_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)
        
    def _get_featurize_fn(self, agent_id):
        if agent_id.startswith('ppo'):
            return lambda state: self.base_env.lossless_state_encoding_mdp(state)
        if agent_id.startswith('bc'):
            return lambda state: self.base_env.featurize_state_mdp(state)
        raise ValueError("Unsupported agent type {0}".format(agent_id))
    
    def _get_obs(self, state):
        # ob_p0 = self._get_featurize_fn(self.curr_agents[0])(state)[0]
        # ob_p1 = self._get_featurize_fn(self.curr_agents[1])(state)[1]
        # return ob_p0.astype(np.float32), ob_p1.astype(np.float32)
        main_agent_obs = self._get_featurize_fn("ppo")(state)[self.main_agent_idx]
        other_agent_ppo_obs = self._get_featurize_fn("ppo")(state)[self.other_agent_idx]
        other_agent_bc_obs = self._get_featurize_fn("bc")(state)[self.other_agent_idx]

        return main_agent_obs, other_agent_ppo_obs, other_agent_bc_obs
    
    
    def _populate_agents(self):
        if not self.fixed_role:
            # Always include at least one ppo agent (i.e. bc_sp not supported for simplicity)
            agents = ['ppo']

            # Coin flip to determine whether other agent should be ppo or bc
            other_agent = 'bc' if np.random.uniform() < self.bc_factor else 'ppo'
            agents.append(other_agent)

            # Randomize starting indices
            np.random.shuffle(agents)
            
            if "bc" not in agents:
                self.main_agent_idx = random.choice([0, 1])
            else:
                self.main_agent_idx = 0 if agents[0] == "ppo" else 1
                
            self.other_agent_idx = 1 - self.main_agent_idx
            self.other_agent_type = agents[self.other_agent_idx]
            self.use_bc = self.other_agent_type == "bc"

            # Ensure agent names are unique
            agents[0] = agents[0] + '_0'
            agents[1] = agents[1] + '_1'
            
            return agents
        else:
            # Always include at least one ppo agent (i.e. bc_sp not supported for simplicity)
            agents = ['ppo']

            # Coin flip to determine whether other agent should be ppo or bc
            other_agent = "ppo"
            agents.append(other_agent)

            # Randomize starting indices
            np.random.shuffle(agents)
            
            self.main_agent_idx = 0
                
            self.other_agent_idx = 1 - self.main_agent_idx
            self.other_agent_type = agents[self.other_agent_idx]
            self.use_bc = self.other_agent_type == "bc"

            # Ensure agent names are unique
            agents[0] = agents[0] + '_0'
            agents[1] = agents[1] + '_1'
            
            return agents
    
    def step(self, action_pair):
        # NOTE action_pair = (main_agent_aciton, other_agent_action)
        # action = [action_dict[self.curr_agents[0]], action_dict[self.curr_agents[1]]]
        action = [action_pair[self.main_agent_idx], action_pair[self.other_agent_idx]]
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        
        if self.use_phi:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=True)
            potential = info['phi_s_prime'] - info['phi_s']
            dense_reward = (potential, potential)
        else:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=False)
            dense_reward = info["shaped_r_by_agent"]
            
        main_agent_obs, other_agent_ppo_obs, other_agent_bc_obs = self._get_obs(next_state)

        shaped_reward_main = sparse_reward + self.reward_shaping_factor * dense_reward[self.main_agent_idx]
        shaped_reward_other = sparse_reward + self.reward_shaping_factor * dense_reward[self.other_agent_idx]
        
        obs = dict(
            main_agent_obs=main_agent_obs,
            other_agent_ppo_obs=other_agent_ppo_obs,
            other_agent_bc_obs=other_agent_bc_obs,
        )
        rewards = [shaped_reward_main, shaped_reward_other]
        dones = done
        infos = info
        return obs, rewards, dones, infos
    
    def reset(self, regen_mdp=True):
        self.base_env.reset(regen_mdp)
        self.curr_agents = self._populate_agents()
        main_agent_obs, other_agent_ppo_obs, other_agent_bc_obs = self._get_obs(self.base_env.state)
        return dict(
            main_agent_obs=main_agent_obs,
            other_agent_ppo_obs=other_agent_ppo_obs,
            other_agent_bc_obs=other_agent_bc_obs,
        )
    
    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            return start_v
        else:
            off_t = curr_t - start_t
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v
        
    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor
        
    def set_bc_factor(self, factor):
        self.bc_factor = factor
    
    def anneal_reward_shaping_factor(self, timesteps):
        # TODO _anneal and set_reward_shaping_factor
        new_factor = self._anneal(self._initial_reward_shaping_factor, timesteps, self.reward_shaping_horizon)
        self.set_reward_shaping_factor(new_factor)
        
    def anneal_bc_factor(self, timesteps):
        p_0 = self.bc_schedule[0]
        p_1 = self.bc_schedule[1]
        i = 2
        while timesteps > p_1[0] and i < len(self.bc_schedule):
            p_0 = p_1
            p_1 = self.bc_schedule[i]
            i += 1
        start_t, start_v = p_0
        end_t, end_v = p_1
        new_factor = self._anneal(start_v, timesteps, end_t, end_v, start_t)
        self.set_bc_factor(new_factor)
        
    @classmethod
    def from_config(cls, env_config):
        assert env_config and "env_params" in env_config and "multi_agent_params" in env_config
        assert "mdp_params" in env_config or "mdp_params_schedule_fn" in env_config, \
            "either a fixed set of mdp params or a schedule function needs to be given"
        # "layout_name" and "rew_shaping_params"
        if "mdp_params" in env_config:
            mdp_params = env_config["mdp_params"]
            outer_shape = None
            mdp_params_schedule_fn = None
        elif "mdp_params_schedule_fn" in env_config:
            mdp_params = None
            outer_shape = env_config["outer_shape"]
            mdp_params_schedule_fn = env_config["mdp_params_schedule_fn"]

        # "start_state_fn" and "horizon"
        env_params = env_config["env_params"]
        # "reward_shaping_factor"
        multi_agent_params = env_config["multi_agent_params"]
        base_ae = get_base_ae(mdp_params, env_params, outer_shape, mdp_params_schedule_fn)
        base_env = base_ae.env

        return cls(base_env=base_env, **multi_agent_params)

class OvercookedMultiAgentFixedRole(OvercookedMultiAgent):
    # TODO impl the env with fixed role
    # main_agent: 0, other_agent: 1;
    def __init__(self, base_env, player_0_type, player_1_type, use_phi=True, reward_shaping_factor=0, reward_shaping_horizon=0, bc_schedule=None):
        # TODO check if addtional setting for player role is nessesary
        super().__init__(base_env, use_phi, reward_shaping_factor, reward_shaping_horizon, bc_schedule)
        self.player_0_type = player_0_type
        self.player_1_type = player_1_type
        self.main_agent_idx = ...
        
    def _populate_agents(self):
        ...

    def reset(self, regen_mdp=True):
        self.base_env.reset(regen_mdp)
        self.curr_agents = self._populate_agents()
        main_agent_obs, other_agent_obs = self._get_obs
        
        
        

from typing import List, Callable, Optional
class OvercookedMultiAgentVecEnv(SubprocVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        super().__init__(env_fns, start_method)
        # dummy_env = env_fns[0]()
        # self.ppo_observation_space = dummy_env.ppo_observation_space
        # self.bc_observation_space = dummy_env.bc_observation_space
    
    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        # print(np.stack(rews))
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos
        
    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        
        return _flatten_obs(obs)
    
    def anneal_reward_shaping_factor(self, timesteps):
        # for remote in self.remotes:
        #     remote.send(("anneal_reward_shaping_factor", timesteps))
        # for remote in self.remotes:
        #     remote.recv()
        self.env_method("anneal_reward_shaping_factor", timesteps)
            
    def anneal_bc_factor(self, timesteps):
        # for remote in self.remotes:
        #     remote.send(("env_method", "anneal_bc_factor", timesteps))
        # for remote in self.remotes:
        #     remote.recv()
        self.env_method("anneal_bc_factor", timesteps)

    def set_reward_shaping_factor(self, reward_shaping_factor):
        self.env_method("set_reward_shaping_factor", reward_shaping_factor)
    
    @property
    def use_bc(self):
        return np.array(self.get_attr("use_bc"), dtype=np.int32)
    
    @property
    def use_ppo(self):
        return 1 - self.use_bc
    @property
    def ppo_observation_space(self):
        return self.get_attr("ppo_observation_space")[0]
    
    @property
    def bc_observation_space(self):
        return self.get_attr("bc_observation_space")[0]

    @property
    def portion_of_bc(self):
        return sum(self.use_bc) / len(self.use_bc)

        
class OvercookedMultiAgentReturnState(gym.Env):
    bc_schedule = self_play_bc_schedule = [(0, 0), (float('inf'), 0)]
    DEFAULT_CONFIG = {
        "mdp_params" : {
            "layout_name" : "cramped_room",
            "rew_shaping_params" : {}
        },
        "env_params" : {
            "horizon" : 400
        },
        "multi_agent_params" : {
            "reward_shaping_factor" : 0.0,
            "reward_shaping_horizon" : 0,
            "bc_schedule" : self_play_bc_schedule,
            "use_phi" : True
        }
    }
    
    def __init__(self, base_env, use_phi=True, reward_shaping_factor=.0, reward_shaping_horizon=0, bc_schedule=None):
        if bc_schedule:
            self.bc_schedule = bc_schedule
        
        self.base_env = base_env
        # self.featurize_fn_map = 

        
class OvercookedCompat(OvercookedMultiAgent):
    def __init__(self, base_env, use_phi=True, reward_shaping_factor=0, reward_shaping_horizon=0, bc_schedule=None):
        super().__init__(base_env, use_phi, reward_shaping_factor, reward_shaping_horizon, bc_schedule)
        self.observation_space = self.bc_observation_space
    
    def _get_obs(self, state):
        main_agent_obs = self._get_featurize_fn("bc")(state)[self.main_agent_idx]
        other_agent_obs = self._get_featurize_fn("bc")(state)[self.other_agent_idx]
        return main_agent_obs, other_agent_obs
    
    def step(self, action_pair):
        action = [action_pair[self.main_agent_idx], action_pair[self.other_agent_idx]]
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        
        if self.use_phi:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=True)
            potential = info['phi_s_prime'] - info['phi_s']
            dense_reward = (potential, potential)
        else:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=False)
            dense_reward = info["shaped_r_by_agent"]
            
        main_agent_obs, other_agent_obs = self._get_obs(next_state)

        shaped_reward_main = sparse_reward + self.reward_shaping_factor * dense_reward[self.main_agent_idx]
        shaped_reward_other = sparse_reward + self.reward_shaping_factor * dense_reward[self.other_agent_idx]
        
        obs = dict(
            main_agent_obs=main_agent_obs,
            other_agent_ppo_obs=other_agent_obs,
        )
        rewards = [shaped_reward_main, shaped_reward_other]
        dones = done
        infos = info
        return obs, rewards, dones, infos
    
    def reset(self, regen_mdp=True):
        self.base_env.reset(regen_mdp)
        self.curr_agents = self._populate_agents()
        main_agent_obs, other_agent_obs = self._get_obs(self.base_env.state)
        return dict(
            main_agent_obs=main_agent_obs,
            other_agent_ppo_obs=other_agent_obs,
        )