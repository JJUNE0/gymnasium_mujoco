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
        max_action
    ):

        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.policy = nn.Linear(256, action_size)

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

        # self.layer_s = nn.Linear(state_size, 200)
        # self.layer_a = nn.Linear(action_size, 200)
        self.Q1_layer1 = nn.Linear(state_size+action_size,256)
        self.Q1_layer2 = nn.Linear(256, 256)
        self.Q1_value = nn.Linear(256, 1)

        self.Q2_layer1 = nn.Linear(state_size+action_size,256)
        self.Q2_layer2 = nn.Linear(256, 256)
        self.Q2_value = nn.Linear(256, 1)

    # Q(s,a; phi)
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.Q1_layer1(sa))
        q1 = F.relu(self.Q1_layer2(q1))
        q1 = self.Q1_value(q1)

        q2 = F.relu(self.Q2_layer1(sa))
        q2 = F.relu(self.Q2_layer2(q2))
        q2 = self.Q2_value(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.Q1_layer1(sa))
        q1 = F.relu(self.Q1_layer2(q1))
        q1 = self.Q1_value(q1)

        return  q1


class TD3():
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

        self.actor_update_steps = 2

        self.train_step = 0
        self.policy_noise = 0.2
        self.noise_clip = 0.5

        self.hard_update(self.actor,self.actor_target)
        self.hard_update(self.critic, self.critic_target)

    # mu(s; theta)
    def get_action(self, state, evaluation=False, test=True):

        # exploration
        if not evaluation:
            actions = self.env.action_space.sample()
        else:

            state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
            # output activation func : tanh
            actions = self.actor(state)
            mean = 0
            std = 0.1
            # max action?
            noise = mean + self.max_action * std * torch.randn_like(actions)
            actions += noise
            # (action_size)
            actions = torch.clamp(actions, -1, 1).detach().cpu().numpy()


        # Tensor -> numpy (no gradient)
        return actions
    def store_sample(self,state, action, reward, next_state, done):
        self.buffer.push(state,action,reward,next_state,done)

    def hard_update(self, network, target_network):
        with torch.no_grad():
            for param, target_param in zip(network.parameters(), target_network.parameters()):
                target_param.data.copy_(param.data)

    def soft_update(self, network, target_network,tau=None):
        if tau is None:
            tau = self.tau
        # target network 가 network의 파라미터 변화를 tau 만큼만 반영
        # 즉 target network를 천천히 변화시키겠다는 것. polyak update라고도 한다.
        with torch.no_grad():
            for target_param, param in zip(target_network.parameters(), network.parameters()):
                target_param.data.copy_((param.data * tau) + (target_param.data * (1 - tau)))


    def train(self):
        # Tensor로 변환되서 나옴.
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():

            next_Policy_tars = self.actor(next_states)

            # Target Policy smoothing
            noise = torch.clamp(torch.normal(mean=0.0, std=0.2, size=next_Policy_tars.size()), -0.5, 0.5).to(self.device)
            next_Policy_tars += noise
            next_Policy_tars = self.max_action * torch.clamp(next_Policy_tars, self.min_action, self.max_action)

            # Q_tar(s',mu_tar(s'); phi'_1)
            # Q_tar(s',mu_tar(s'); phi'_2)
            next_Q1value_targs, next_Q2value_targs = self.critic_target(next_states, next_Policy_tars)  # (batch_size, 1)
            # Clipped Double Q
            next_Qvalue_targs = torch.min(next_Q1value_targs, next_Q2value_targs)  # (batch_size, 1)
            # y
            target_Qs = rewards + self.gamma * next_Qvalue_targs * (1 - dones)  # (batch_size, 1)

        # Q(s,a; phi_1)
        # Q(s,a; phi_2)
        Q1values, Q2values = self.critic(states, actions)  # (batch_size,1)

        # L(phi_1,phi_2) = [y - Q1(s,a)]^2 + [y - Q2(s,a)]^2
        critic_loss = F.mse_loss(Q1values, target_Qs, reduction='mean') + F.mse_loss(Q2values, target_Qs, reduction='mean')

        self.c_opt.zero_grad()
        critic_loss.backward()
        self.c_opt.step()

        self.train_step +=1

        # Delayed Policy update
        if self.train_step % self.actor_update_steps == 0:
            # mu(s; theta)
            Policys = self.actor(states)    # (batch_size, action_size)
            # # J(theta) = E[Q(s,mu(s; theta))]
            #print(-self.critic(states, Policys))
            actor_loss = -self.critic.Q1(states, Policys).mean()  # Negative Q-value for actor loss

            self.a_opt.zero_grad()
            actor_loss.backward()
            self.a_opt.step()

            self.soft_update(self.actor,self.actor_target)
            self.soft_update(self.critic,self.critic_target)


    def load_weights(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor_parameters.pth',weights_only=True))
        self.actor_target.load_state_dict(torch.load(path + 'actor_target_parameters.pth',weights_only=True))
        self.critic.load_state_dict(torch.load(path + 'critic_parameters.pth',weights_only=True))
        self.critic_target.load_state_dict(torch.load(path + 'critic_target_parameters.pth',weights_only=True))
