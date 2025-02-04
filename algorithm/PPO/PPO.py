import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .replay_memory import ReplayMemory
import torch.optim as optim

hidden_size = 256
half_hidden_size = int(hidden_size/2)

class Actor(nn.Module):
    def __init__(
        self,
        state_size : int,
        action_size : int,
    ):

        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_size)
        self.std = nn.Linear(hidden_size, action_size)

    # Gaussian distributin N(mu,sigma^2)
    def forward(self, state):
        layer1 = F.relu(self.layer1(state))
        layer2 = F.relu(self.layer2(layer1))

        mu = F.tanh(self.mu(layer2))
        std = F.softplus(self.std(layer2))

        return mu, std

class Critic(nn.Module):
    def __init__(
        self,
        state_size : int,
        action_size : int,
    ):

        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.Qvalue = nn.Linear(hidden_size, 1)

    # Q(s,a; phi)
    def forward(self, state,  softmax_dim = 0
                #action
                ):
        layer1 = F.relu(self.layer1(state))
        layer2 = F.relu(self.layer2(layer1))

        Qvalue = self.Qvalue(layer2)

        return Qvalue

class PPO():
    def __init__(
        self,
        env: gym.Env,
        device,
        args,
    ):

        self.env = env
        self.device = device
        self.args = args


        self.state_size = self.env.observation_space.shape[0]
        print("state dim : ", self.state_size)
        self.action_size = len(env.action_space.high)
        print("action dim : ", self.action_size)
        self.min_action = env.action_space.low[0]
        print("min action : ", self.min_action)
        self.max_action = env.action_space.high[0]
        print("max_action : ", self.max_action)

        self.std_bound = [1e-2, 0.8]
        self.tau = args.tau

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.gamma = 0.98  # discount rate
        self.gae_lambda = 0.95

        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = Critic(self.state_size, self.action_size).to(self.device)

        self.batch_size = args.batch_size
        # capacity : batch_size
        self.buffer = ReplayMemory(self.state_size, self.action_size, self.device, self.batch_size)

        self.a_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.clip_param = 0.1
        #self.actor_update_steps = 10
        self.train_step = 0

        self.epochs = 10
        self.vf_coef = 1.0
        self.ent_coef = 0.01


    def store_sample(self, state, action, reward, next_state, log_old_policy, done):
        self.buffer.push(state, action, reward, next_state, log_old_policy ,done)


    def log_pdf(self, mu, std, action):

        std = np.clip(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (action-mu)**2/var - 0.5*np.log(var*2*np.pi)
        #print(log_policy_pdf)

        return log_policy_pdf

    # mu(s; theta)
    def get_action(self, state, evaluation=False, test=False):
        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
        # output activation func : tanh
        mu, std = self.actor(state)
        std = torch.clip(std, self.std_bound[0], self.std_bound[1])
        dist = Normal(mu,std)
        # 약간의 exploration?
        action = dist.sample()
        action = torch.clip(action, self.min_action, self.max_action)
        action = action.detach().cpu().numpy()
        if test == True:
            return action

        mu = mu.detach().cpu().numpy()
        std = std.detach().cpu().numpy()


        return mu, std, action

    def gae_target(self, rewards, v_values, next_v_values, dones):
        n_step_td_targets = torch.zeros_like(rewards)
        gaes = torch.zeros_like(rewards)
        gae = 0

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.gamma * next_v_values[k] * (1-dones[k]) - v_values[k]
            gae = self.gamma * self.gae_lambda * gae + delta
            gaes[k] = gae
            n_step_td_targets[k] = gae + v_values[k]

        return gaes, n_step_td_targets

    def actor_learn(self, states, actions, log_old_policys, gaes):

        mu, std = self.actor(states)
        dist = Normal(mu, std)
        log_policys = dist.log_prob(actions)
        ratio = torch.exp(log_policys - log_old_policys)
        # ppo Clipped Loss
        surr1 = ratio * gaes
        surr2 = torch.clip(ratio, 1 - self.clip_param, 1 + self.clip_param) * gaes
        cliped_loss = -torch.min(surr1, surr2)

        entropy_bonus = -dist.entropy().mean()
        actor_loss = torch.mean(cliped_loss) + self.ent_coef * entropy_bonus

        return  actor_loss
        # Update Actor


    def critic_learn(self, states, td_targets):
        # Recalculate V_values for Critic
        V_values = self.critic(states)
        critic_loss = F.mse_loss(td_targets, V_values)

        return critic_loss


    def train(self):
        # Sample data from buffer shuffle = true
        loader = self.buffer.sample()
        # epoch
        for i in range(10):
            for batch in loader:
                states, actions, rewards, next_states, log_old_policys, dones = batch
                # Compute V_values and GAE targets
                with torch.no_grad():
                # [batch_size, 1]
                    V_values = self.critic(states)
                    next_V_values = self.critic(next_states)
                    gaes, td_targets = self.gae_target(rewards, V_values, next_V_values, dones)

                # [batch_size, action_size]
                actor_loss = self.actor_learn(states, actions, log_old_policys, gaes)
                critic_loss = self.critic_learn(states, td_targets)


                self.a_opt.zero_grad()
                self.c_opt.zero_grad()
                # Actor와 Critic을 분리하면 retain_graph 필요 없음
                actor_loss.backward()
                critic_loss.backward()

                self.a_opt.step()
                self.c_opt.step()


    def load_weights(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor_parameters.pth',weights_only=True))
        self.critic.load_state_dict(torch.load(path + 'critic_parameters.pth',weights_only=True))

















