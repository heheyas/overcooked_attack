import torch
import torch.nn as nn
from a2c_ppo_acktr.model import NNBase
from a2c_ppo_acktr.algo import PPO
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class CustomCNN(NNBase):
    def __init__(self, placeholder, observation_space, recurrent=False, hidden_size=32):
        super(CustomCNN, self).__init__(recurrent, hidden_size, hidden_size)
        
        n_input_channels = observation_space.shape[0]
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
        
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(n_input_channels, 25, 5, padding="same")), nn.LeakyReLU(),
            init_(nn.Conv2d(25, 25, 3, padding="same")), nn.LeakyReLU(),
            init_(nn.Conv2d(25, 25, 3, padding="valid")), nn.LeakyReLU(), 
            nn.Flatten(),
        )
        
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
            
        self.linear = nn.Sequential(
            init_(nn.Linear(n_flatten, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, hidden_size)),
            nn.LeakyReLU()
        )
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        self.train()
        
    def forward(self, inputs, rnn_hxs, masks):
        x = self.linear(self.cnn(inputs))
        
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
            
        return self.critic_linear(x), x, rnn_hxs
            
            
class CustomPolicy(nn.Module):
    def forward(self, inputs, rnn_hxs, masks):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        
        return dist

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

class TwoAgentRolloutStorage():
    def __init__(self, kwargs1, kwargs2):
        self.rollouts1 = RolloutStorage(**kwargs1)
        self.rollouts2 = RolloutStorage(**kwargs2)
        
    def to(self, device):
        self.rollouts1.to(device)
        self.rollouts2.to(device)
        
    def insert(self, obs1, recurrent_hidden_state1, actions1, action_log_probs1, obs2, recurrent_hidden_states2, actions2, action_log_probs2):
        self.rollouts1.insert(obs1, recurrent_hidden_state1, actions1, action_log_probs1)
        self.rollouts2.insert(obs2, recurrent_hidden_states2, actions2, action_log_probs2)

    def after_update(self):
        self.rollouts1.after_update()
        self.rollouts2.after_update()
        
    def compute_returns(self, next_value1, next_value2, use_gae, gamma, gae_lambda, use_proper_time_limits=True):
        self.rollouts1.compute_returns(next_value1, use_gae, gamma, gae_lambda, use_proper_time_limits)
        self.rollouts1.compute_returns(next_value2, use_gae, gamma, gae_lambda, use_proper_time_limits)

    def feed_forward_generator(self, advantages1, advantages2, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rollouts1.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, ()
            
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)
        
        for indices in sampler:
            obs1_batch = self.rollouts1.obs[:-1].view(-1, *self.rollouts1.obs.size()[2:])[indices]
            recurrent_hidden_states1_batch = self.rollouts1.recurrent_hidden_states[:-1].view(-1, self.rollouts1.recurrent_hidden_states.size(-1))[indices]
            actions1_batch = self.rollouts1.actions.view(-1, self.rollouts1.actions.size(-1))[indices]
            value_preds1_batch = self.rollout1.value_preds[:-1].view(-1, 1)[indices]
            return1_batch = self.rollouts1.returns[:-1].view(-1, 1)
            masks1_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs1_batch = self.rollouts1.action_log_probs.view(-1, 1)[indices]

            obs2_batch = self.rollouts2.obs[:-1].view(-1, *self.rollouts2.obs.size()[2:])[indices]
            recurrent_hidden_states2_batch = self.rollouts2.recurrent_hidden_states[:-1].view(-1, self.rollouts2.recurrent_hidden_states.size(-1))[indices]
            actions2_batch = self.rollouts2.actions.view(-1, self.rollouts2.actions.size(-1))[indices]
            value_preds2_batch = self.rollout1.value_preds[:-1].view(-1, 1)[indices]
            return2_batch = self.rollouts2.returns[:-1].view(-1, 1)
            masks2_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs2_batch = self.rollouts2.action_log_probs.view(-1, 1)[indices]

            if advantages1 is None:
                adv_targ1 = None
            else:
                adv_targ1 = advantages1.view(-1, 1)[indices]
                
            if advantages2 is None:
                adv_targ2 = None
            else:
                adv_targ2 = advantages2.view(-1, 1)[indices]
                
            yield obs1_batch, recurrent_hidden_states1_batch, actions1_batch, value_preds1_batch, return1_batch, masks1_batch, old_action_log_probs1_batch, adv_targ1, obs2_batch, recurrent_hidden_states2_batch, actions2_batch, value_preds2_batch, return2_batch, masks2_batch, old_action_log_probs2_batch, adv_targ2

    def recurrent_generator(self, advantages1, advantages2, num_mini_batch):
        num_processes = self.rollouts1.rewards.size(1)
        
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs1_batch = []
            recurrent_hidden_states1_batch = []
            actions1_batch = []
            value_preds1_batch = []
            return1_batch = []
            masks1_batch = []
            old_action_log_probs1_batch = []
            adv_targ1 = []
            
            obs2_batch = []
            recurrent_hidden_states2_batch = []
            actions2_batch = []
            value_preds2_batch = []
            return2_batch = []
            masks2_batch = []
            old_action_log_probs2_batch = []
            adv_targ2 = []
            
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs1_batch.append(self.rollou)
            

class OaamPPO(PPO):
    def __init__(self, actor_critic, clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef, lr=None, eps=None, max_grad_norm=None, oaam_coef=None, use_clipped_value_loss=True):
        super().__init__(actor_critic, clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef, lr, eps, max_grad_norm, use_clipped_value_loss)
        
        self.oaam_coef = oaam_coef
        
    def update(self, rollouts1: RolloutStorage, rollouts2: RolloutStorage):
        advantages1 = rollouts1.returns[:-1] - rollouts1.value_preds[:-1]
        
        advantages2 = rollouts2.returns[:-1] - rollouts2.value_preds[:-1]
        
        advantages1 = (advantages1 - advantages1.mean()) / (advantages1.std() + 1e-5)
        advantages2 = (advantages2 - advantages2.mean()) / (advantages2.std() + 1e-5)
        
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        
        # NOTE
        
        