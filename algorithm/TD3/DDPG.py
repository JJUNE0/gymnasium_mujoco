import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .replay_memory import ReplayMemory
import torch.optim as optim

hidden_size = 64
half_hidden_size = int(hidden_size/2)

class Actor(nn.Module):
    def __init__(
        self,
        state_size : int,
        action_size : int,
        max_action
    ):

        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, action_size)

        self.max_action = max_action

    # mu(s; theta)
    def forward(self, state):
        layer1 = F.relu(self.layer1(state))
        layer2 = F.relu(self.layer2(layer1))
        policy = torch.tanh(self.policy(layer2))

        return policy * self.max_action

class Critic(nn.Module):
    def __init__(
        self,
        state_size : int,
        action_size : int,
    ):

        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.layer_s = nn.Linear(state_size, half_hidden_size)
        self.layer_a = nn.Linear(action_size, half_hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.Qvalue = nn.Linear(hidden_size, 1)

    # Q(s,a; phi)
    def forward(self, state, action):
        layer_s = F.relu(self.layer_s(state))
        layer_a = F.relu(self.layer_a(action))

        layer_cat = torch.cat([layer_s, layer_a], dim=1)
        layer2 = F.relu(self.layer2(layer_cat))

        Qvalue = self.Qvalue(layer2)

        return Qvalue

class DDPG():
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
        self.tau = args.tau

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.gamma = 0.99  # discount rate
        self.actor = Actor(self.state_size, self.action_size, self.max_action).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, self.max_action).to(self.device)
        self.critic = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size).to(self.device)

        # default capcity : 1e6
        self.buffer = ReplayMemory(self.state_size, self.action_size, self.device)
        self.batch_size = args.batch_size

        self.a_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.actor_update_steps = 10
        self.train_step = 0
        self.update_target()

    # mu(s; theta)
    def get_action(self, state, evaluation=False):

        # exploration
        if not evaluation:
            actions = self.env.action_space.sample()
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
            actions = self.actor(state)
            mean = 0
            std = 0.1
            noise = mean + self.max_action * std * torch.randn_like(actions)
            actions += noise
            actions = torch.clamp(actions, self.min_action, self.max_action).detach().cpu().numpy()



        return actions

    def store_sample(self,state, action, reward, next_state, done):
        self.buffer.push(state,action,reward,next_state,done)

    def update_target(self, tau=None):
        if tau is None:
            tau = self.tau

        # target network 가 network의 파라미터 변화를 tau 만큼만 반영
        # 즉 target network를 천천히 변화시키겠다는 것. polyak update라고도 한다.
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_((param.data * tau) + (target_param.data * (1 - tau)))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((param.data * tau) + (target_param.data * (1 - tau)))

    def train(self):
        # Tensor로 변환되서 나옴.
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Q(s,a; phi)
        Qvalues = self.critic(states, actions)   # (batch_size,1)
        # mu_tar(s'; theta')
        next_Policy_targs = self.actor_target(next_states).detach()  # (batch_size, action_size)
        # Q_tar(s',mu_tar(s'); phi')
        next_Qvalue_targs = self.critic_target(next_states, next_Policy_targs).detach()  # (batch_size, 1)

        # y
        expected_Qs = rewards + self.gamma * next_Qvalue_targs * (1 - dones)            # (batch_size, 1)

        # L(phi) = [y - Q(s,a)]^2
        critic_loss = F.mse_loss(Qvalues, expected_Qs, reduction='mean')

        # J(theta) = E[Q(s,mu(s; theta))]
        self.c_opt.zero_grad()
        critic_loss.backward()  # Backpropagate critic loss
        self.c_opt.step()


        # Zero gradients before backward pass
        self.train_step += 1
        # Delayed Policy update
        if self.train_step % self.actor_update_steps == 0:
            # mu(s; theta)
            Policys = self.actor(states)  # (batch_size, action_size)
            actor_loss = -self.critic(states, Policys).mean()  # Negative Q-value for actor loss

            self.a_opt.zero_grad()
            actor_loss.backward()  #
            self.a_opt.step()
