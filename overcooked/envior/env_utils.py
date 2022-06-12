import os
import gym
import random
import numpy as np
import torch
from pathlib import Path
from default import bc_policy_saved_path
from .env import OvercookedMultiAgent, OvercookedMultiAgentVecEnv, OvercookedCompat
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv
from a2c_ppo_acktr.model import Policy
# from game_agent import ApagAgentNewVersion
from overcooked_ai_py.mdp.actions import Action

def set_global_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def linear_anealing(timestep, end, threshold=0., start=0.):
    if timestep <= start:
        return 1.
    elif timestep >= end:
        return threshold
    else:
        return 1. - (timestep - start) / (end - start)
            
def get_vectorized_gym_env(env_config, sim_threads, use_compat=False):
    base_env = OvercookedMultiAgent if not use_compat else OvercookedCompat
    if env_config:
        def gym_env_fn():
            return base_env.from_config(env_config)
        
    else:
        def gym_env_fn():
            return base_env.from_config(base_env.DEFAULT_CONFIG)
    vectorized_gym_env = OvercookedMultiAgentVecEnv([gym_env_fn] * sim_threads)
    return vectorized_gym_env


from overcooked_ai_py.agents.agent import Agent, AgentPair
class ApagAgentNewVersion(Agent):
    def __init__(self, actor_critic: Policy, agent_index: int, featurize_fn):
        self.actor_critic = actor_critic
        self.agent_index = agent_index
        self.featurize = featurize_fn
        
    def reset(self):
        if self.actor_critic.is_recurrent:
            # TODO add recurrent policy initial state
            pass
        else:
            self.rnn_state = []
            
    def action_probabilities(self, state):
        obs = self.featurize(state, debug=False)
        my_obs = obs[self.agent_index]
        if not isinstance(my_obs, torch.Tensor):
            my_obs = obs_as_tensor(obs, self.actor_critic.device)
            
        _, feats, rnn_hxs = self.actor_critic.base(obs, rnn_hxs, _)
        
        dist = self.actor_critic.dist(feats)
        
        return dist.probs.cpu().numpy()
            
    def action(self, state):
        obs = self.featurize(state)
        my_obs = obs[self.agent_index]
        
        _, action, action_log_prob, rnn_hxs = self.actor_critic.act(my_obs, self.rhh_hxs, _)
        
        agent_action_info = {
            "action_probs": action_log_prob.exp(),
        }
        agent_action = Action.INDEX_TO_ACTION[action[0]]
        
        self.rnn_hxs = rnn_hxs
        
        return agent_action, agent_action_info
          
def load_apag_policy(save_path, policy_id="ppo_sp"):
    [policy, gym_env] = torch.load(f"{save_path}/{policy_id}.pt")
                    
def load_apag_agent(save_path, policy_id="ppo", agent_index=0):
    policy = load_apag_policy(save_path, policy_id="ppo_sp")
    # TODO fix featurize_fn
    return ApagAgentNewVersion(policy, agent_index=agent_index, featurize_fn=None)
    
import inspect
def get_required_arguments(fn):
    required = []
    params = inspect.signature(fn).parameters.values()
    for param in params:
        if param.default == inspect.Parameter.empty and param.kind == param.POSITIONAL_OR_KEYWORD:
            required.append(param)
    return required

def configure_bc_agent(bc_config):
    from stable_baselines3.common.policies import ActorCriticPolicy as acp
    fpath = Path(bc_policy_saved_path) / bc_config["layout_name"] / bc_config["mode"] / "policy.zip"
    bc_policy = acp.load(fpath)
    
    return bc_policy