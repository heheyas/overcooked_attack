from copy import deepcopy
import os
import sys
from pathlib import Path
from cv2 import DISOPTICAL_FLOW_PRESET_FAST
sys.path.append("..")
from tqdm import tqdm
import time
from collections import deque
import torch
import numpy as np
import stable_baselines3
from pathlib import Path
from stable_baselines3.common.utils import obs_as_tensor, configure_logger
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.algo import PPO

from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from overcooked_ai_py.mdp.actions import Action

from envior.env_utils import get_vectorized_gym_env, configure_bc_agent
from .model import CustomCNN
from .utils import load_ac_from_file

class AdvTrainer():
    def __init__(self, params):
        self.gym_env = get_vectorized_gym_env(params["env_config"], params["sim_threads"], params["use_compat"])
        self.params = params
        
        if not params["use_compat"]:
            policy_kwargs = dict(
                observation_space=self.gym_env.ppo_observation_space,
                recurrent=params["use_rnn"],
                hidden_size=params["hidden_size"],
            )
            
            self.actor_critic = Policy(obs_shape=self.gym_env.ppo_observation_space.shape, action_space=self.gym_env.action_space, base=CustomCNN, base_kwargs=policy_kwargs)
            
        else:
            policy_kwargs = dict(
                observation_space=self.gym_env.bc_observation_space,
                recurrent=params["use_rnn"],
                hidden_size=params["hidden_size"],
            )
            self.actor_critic = Policy(obs_shape=self.gym_env.bc_observation_space.shape, action_space=self.gym_env.action_space)
        
        self.actor_critic.to(params["device"])
        self.logger = configure_logger(params["verbose"], params["tensorboard_log"], params["tb_log_name"])
        
        self.agent = PPO(
            self.actor_critic,
            params["clip_range"],
            params["n_epochs"],
            params["num_mini_batch"],
            params["vf_coef"],
            params["ent_coef"],
            params["lr"],
            params["eps"],
            params["max_grad_norm"],
        )
        
        self.observation_space = self.gym_env.ppo_observation_space if not params["use_compat"] else self.gym_env.bc_observation_space
        
        self.main_agent_rollouts = RolloutStorage(
            params["mini_batch_size"],
            params["sim_threads"],
            self.observation_space.shape,
            self.gym_env.action_space,
            self.actor_critic.recurrent_hidden_state_size,
        )

        self.other_agent_ppo_rollouts = RolloutStorage(
            params["mini_batch_size"],
            params["sim_threads"],
            self.observation_space.shape,
            self.gym_env.action_space,
            self.actor_critic.recurrent_hidden_state_size,
        )
        
        self.other_actor_critic = load_ac_from_file(params)
        if params["bootstrapped"]:
            self.actor_critic = deepcopy(self.other_actor_critic)
        
    def update(self):
        obs = self.gym_env.reset()
        self.main_agent_rollouts.obs[0].copy_(torch.tensor(obs["main_agent_obs"]))
        self.main_agent_rollouts.to(self.params["device"])
        self.other_agent_ppo_rollouts.obs[0].copy_(torch.tensor(obs["other_agent_ppo_obs"]))
        self.other_agent_ppo_rollouts.to(self.params["device"])
        
        episode_rewards = deque(maxlen=self.params["sim_threads"])
        episode_sparse_rewards = deque(maxlen=self.params["sim_threads"])

        start = time.time()
        
        num_updates = int(self.params["total_timesteps"]) // self.params["mini_batch_size"] // self.params["sim_threads"]
        
        for j in range(num_updates):
            timesteps = j * self.params["mini_batch_size"] * self.params["sim_threads"]
        
            self.gym_env.anneal_bc_factor(timesteps)
            self.gym_env.anneal_reward_shaping_factor(timesteps)
                
            self.collect_rollouts(episode_rewards, episode_sparse_rewards)
                
                
            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    self.main_agent_rollouts.obs[-1], self.main_agent_rollouts.recurrent_hidden_states[-1],
                    self.main_agent_rollouts.masks[-1]
                ).detach()
                
            self.main_agent_rollouts.compute_returns(
                next_value,
                self.params["use_gae"],
                self.params["gamma"],
                self.params["gae_lambda"]
            )
                
            value_loss, action_loss, dist_entropy = self.agent.update(self.main_agent_rollouts)
            # value_loss, action_loss, dist_entropy = agent.update(self.rollouts2)
            
            self.main_agent_rollouts.after_update()
            self.other_agent_ppo_rollouts.after_update()
                
            if (j % self.params["save_interval"] == 0
                or j == num_updates - 1) and self.params["save_dir"] != "":
                # save_path = os.path.join(self.params["save_dir"], "ppo_sp")
                save_path = self.params["save_dir"]
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save(self.actor_critic, os.path.join(save_path, "policy.pt"))
                torch.save(self.params, os.path.join(save_path, "config.pt"))

            if j % self.params["log_interval"] == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * self.params["sim_threads"] * self.params["mini_batch_size"]
                end = time.time()
                
                fps = int(total_num_steps / (end - start))
                rew_mean = np.mean(episode_rewards)
                rew_median = np.median(episode_rewards)
                rew_sparse_mean = np.mean(episode_sparse_rewards)
                rew_sparse_median = np.median(episode_sparse_rewards)
                
                self.logger.record("time/iterations", j, exclude="tensorboard")
                self.logger.record("time/num_timesteps", total_num_steps, exclude="tensorboard")
                self.logger.record("time/fps", fps)
                
                self.logger.record("rollout_shaped/mean_reward", rew_mean)
                self.logger.record("rollout_shaped/median_reward", rew_median)
                self.logger.record("rollout_shaped/min_reward", np.min(episode_rewards))
                self.logger.record("rollout_shaped/max_reward", np.max(episode_rewards))
                
                self.logger.record("rollout_sparse/mean_sparse_reward", rew_sparse_mean)
                self.logger.record("rollout_sparse/median_sparse_reward", rew_sparse_median)
                
                self.logger.record("rollout_sparse/min_sparse_reward", np.min(episode_sparse_rewards))
                self.logger.record("rollout_sparse/max_sparse_reward", np.max(episode_sparse_rewards))
                
                self.logger.record("train/entropy", dist_entropy)
                self.logger.record("train/value_loss", value_loss)
                self.logger.record("train/action_loss", action_loss)
                
                self.logger.dump(step=total_num_steps)
                
    def save(self, save_dir=None, save_agent_name=None):
        if save_dir is None:
            save_dir = Path(self.params["save_dir"]) / f"ppo_{self.params['other_agent']}"
        try:
            os.makedirs(save_dir)
        except OSError:
            pass
        
        save_agent_name = f"{self.params['layout_name']}.pt" if save_agent_name is None else f"{save_agent_name}.pt"
        torch.save(self, save_dir / save_agent_name)    

    @torch.no_grad()
    def collect_rollouts(self, episode_rewards, episode_sparse_rewards):
        for step in range(self.params["mini_batch_size"]):
            m_value, m_action, m_action_log_prob, m_recurrent_hidden_states = self.actor_critic.act(self.main_agent_rollouts.obs[step], self.main_agent_rollouts.recurrent_hidden_states[step], self.main_agent_rollouts.masks[step])
            
            op_value, op_action, op_action_log_prob, op_recurrent_hidden_states = self.other_actor_critic.act(self.other_agent_ppo_rollouts.obs[step], self.other_agent_ppo_rollouts.recurrent_hidden_states[step], self.other_agent_ppo_rollouts.masks[step])
            
            joint_actions = []
            for idx in range(len(m_action)):
                joint_actions.append((m_action[idx].item(), op_action[idx].item()))
                
            obs, reward, done, infos = self.gym_env.step(joint_actions)
            
            # the adversarial attacker lower the reward
            reward = -reward
            
            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["ep_shaped_r"])
                    episode_sparse_rewards.append(info["episode"]["ep_sparse_r"])
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
            
            # print(reward)
            self.main_agent_rollouts.insert(torch.tensor(obs["main_agent_obs"]), m_recurrent_hidden_states, m_action, m_action_log_prob, m_value, torch.tensor(reward[:, 0]).unsqueeze(1), masks, bad_masks)
            
            self.other_agent_ppo_rollouts.insert(torch.tensor(obs["other_agent_ppo_obs"]), op_recurrent_hidden_states, op_action, op_action_log_prob, op_value, torch.tensor(reward[:, 1]).unsqueeze(1), masks, bad_masks)
    
    @torch.no_grad()        
    def generate_trajs(self):
        pass