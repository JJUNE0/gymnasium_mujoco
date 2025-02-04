import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random

from .replay_memory import  ReplayMemory

class QR_Network(nn.Module):
    def __init__(self,
                 state_dim : int,
                 action_dim : int,
                 quantile_len,
                 hidden_dim : int = 256,
                 activation = nn.ReLU()
                 ):
     super(QR_Network, self).__init__()

     self.quantile_len = quantile_len

     self.layer1 = nn.Linear(state_dim, hidden_dim)
     self.layer2 = nn.Linear(hidden_dim, hidden_dim)

     # output : learnable support
     self.Z_value = nn.Linear(hidden_dim, action_dim * quantile_len)

     self.activation = activation

    def forward(self, state):

         layer1 = self.activation(self.layer1(state))
         layer2 = self.activation(self.layer2(layer1))

         Z = self.Z_value(layer2)

         # Reshape to [batch_size, action_dim, quantile_len]
         Z = Z.view(-1, self.Z_value.out_features // self.quantile_len, self.quantile_len)

         return Z

class QR_DQN():
    def __init__(
            self,
            env : gym.Env,
            device,
            args):

        self.env = env
        self.device = device
        self.args = args

        self.state_dim = self.env.observation_space.shape[0]
        print("state dim : ", self.state_dim)
        self.action_dim = self.env.action_space.n
        print("action dim : ", self.action_dim)

        self.buffer = ReplayMemory(self.state_dim, self.action_dim, self.device)
        self.batch_size = args.batch_size

        self.quantile_len = 8
        # quantile midpoint
        self.tau_hat = torch.Tensor((2*np.arange(self.quantile_len)+1)/(2.0*self.quantile_len)).view(1,-1).to(self.device)

        self.qr_dqn = QR_Network(self.state_dim, self.action_dim, self.quantile_len).to(self.device)
        self.target_qr_dqn = QR_Network(self.state_dim, self.action_dim, self.quantile_len).to(self.device)
        self.target_qr_dqn.load_state_dict(self.qr_dqn.state_dict())

        self.gamma = args.gamma
        self.lr = 1e-4#args.lr

        self.optimizer = optim.Adam(self.qr_dqn.parameters(), lr=self.lr)

    def soft_update(self, network, target_network, tau = 0.005):
        with torch.no_grad():
            for param, target_param in zip(network.parameters(), target_network.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, network = None, target_network = None):
        if network == None:
            network = self.qr_dqn
        if target_network == None:
            target_network = self.target_qr_dqn

        with torch.no_grad():
            for param, target_param in zip(network.parameters(), target_network.parameters()):
                target_param.data.copy_(param.data)

    def huber(self, u, k=1.0):
        return torch.where(u.abs() <= k, 0.5 * u.pow(2), k * (u.abs() - 0.5 * k))

    def store_sample(self,state, action, reward, next_state, done):
        self.buffer.push(state,action,reward,next_state,done)

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                z_value = self.qr_dqn(state)
                action = z_value.mean(2).max(1)[1].item()
                return action

    def train(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            # [batch size, action dim ,quantile length]
            next_q_values_dist = self.target_qr_dqn(next_states)
            # [batch_size]
            max_actions = next_q_values_dist.mean(2).max(1)[1]
            # [batch size, quantile length]
            next_max_theta = next_q_values_dist[torch.arange(self.batch_size), max_actions]

        # [batch size, quantile length] : 각 batch 에서 선택된 action 의 quantile distribution.
        theta = self.qr_dqn(states)[torch.arange(self.batch_size), actions.view(-1).long()]

        # [batch size, quantile length]
        td_target = rewards + self.gamma * (1 - dones) * next_max_theta.detach()

        # why transpose?
        # [quantile length, batch size, quantile length]
        td_error = td_target.t().unsqueeze(-1) - theta.unsqueeze(0)
        # self.tau_hat.shape : [1, 200]
        # [quantile length, batch size, quantile length]
        quantile_huber_loss = self.huber(td_error) * (self.tau_hat - (td_error.detach() < 0).float()).abs()
        loss = quantile_huber_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
