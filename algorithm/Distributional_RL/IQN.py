import gymnasium as gym
import math

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random

from .replay_memory import  ReplayMemory

class IQR_Network(nn.Module):
    def __init__(self,
                 state_dim : int,
                 action_dim : int,
                 device,
                 hidden_dim : int = 256,
                 quantile_len: int = 8,
                 embeding_dim : int = 64,
                 activation = nn.ReLU()
                 ):
        super(IQR_Network, self).__init__()

        self.quantile_len = quantile_len
        self.embeding_dim = embeding_dim
        self.device = device

        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        # output : learnable support
        self.Z_value = nn.Linear(hidden_dim, action_dim * quantile_len)

        self.cosine_net = nn.Sequential(
            nn.Linear(self.embeding_dim, state_dim),
            nn.ReLU()
        )

        # [1, 1, embeding_dim]
        #self.pis = torch.FloatTensor([np.pi * i for i in range(self.embeding_dim)]).view(1, 1, self.embeding_dim).to(device)
        self.pis = torch.pi * torch.arange(1, self.embeding_dim + 1).view(1,1,-1).to(device)

        self.activation = activation

    def calc_cosine(self, taus):
        # 나중에 다시 확인.
        # [batch_size, n_taus, embeding dim]
        cos = torch.cos(taus * self.pis)
        return cos

    def forward(self, state, taus):

        batch_size = state.shape[0]
        num_tau = taus.shape[1]
        # [batch size, n_taus, embeding dim]
        cos = self.calc_cosine(taus)
        # [batch size, n_taus, state dim]
        """ 확인 필요 """
        # [batch_size, n_taus, state dim]
        cos_x = self.cosine_net(cos)
        # [batch_size, n_taus, state dim]
        x = torch.mul(state.unsqueeze(1), cos_x)
        # [batch size, n_taus, hidden dim]
        x = self.activation(self.layer1(x))
        # [batch size, n_taus, hidden dim]
        x = self.activation(self.layer2(x))
        # [batch size, n_taus, quantile len * action dim]
        Z = self.Z_value(x)
        # Reshape to [batch_size, n_taus, action_dim, quantile_len]
        Z = Z.view(batch_size, num_tau, -1, self.quantile_len)
        # Z = Z.view(batch_size, num_tau, -1, self.quantile_len
        return Z

# iqn = IQR_Network(4,2)
# a = iqn(torch.randn(1, 4))
# print(a.shape)
class IQN():
    def __init__(self,
                 env : gym.Env,
                 device,
                 args):

        self.env = env
        self.device= device
        self.args = args

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.buffer = ReplayMemory(self.state_dim, self.action_dim, self.device)
        self.batch_size = args.batch_size

        self.quantile_len = 8
        self.iqn = IQR_Network(self.state_dim, self.action_dim, device, self.quantile_len).to(self.device)
        self.target_iqn = IQR_Network(self.state_dim, self.action_dim, device, self.quantile_len).to(self.device)
        self.target_iqn.load_state_dict(self.iqn.state_dict())

        self.gamma = args.gamma
        self.lr = args.lr

        self.optimizer = optim.Adam(self.iqn.parameters(), lr=self.lr)

        self.K = 32
        self.N = 8
        self.N_prime = 8

    def risk_cpw(self, tau, eta=0.71):
        return (pow(tau, eta)/(pow(tau,eta) + pow(1-tau, eta))**(1/eta)).clamp(0.0,1.0)

    def risk_pow(self, tau, eta=0):
        # risk-seeking
        if eta >=0:
            return (pow(tau,(1/(1+abs(tau))))).clamp(0.0, 1.0)
        # risk-averse
        else:
            return (1 - pow((1 - tau), (1/(1+abs(eta)))))

    def hard_update(self, network = None, target_network = None):
        if network == None:
            network = self.iqn
        if target_network == None:
            target_network = self.target_iqn

        with torch.no_grad():
            for param, target_param in zip(network.parameters(), target_network.parameters()):
                target_param.data.copy_(param.data)

    def huber(self, u, k=1.0):
        return torch.where(u.abs() <= k, 0.5 * u.pow(2), k * (u.abs() - 0.5 * k))

    def store_sample(self,state, action, reward, next_state, done):
        self.buffer.push(state,action,reward,next_state,done)

    # get action에서만 risk-sensitive
    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                batch_size = state.shape[0]
                # [batch_size, n_tau, 1]
                tilde_taus = torch.randn(batch_size, self.K).to(self.device).unsqueeze(-1)
                tilde_taus = self.risk_cpw(tilde_taus)
                # [1, n_tau, action_dim, quantile len]
                z_value = self.iqn(state, tilde_taus)
                action = z_value.mean(1).mean(2).max(1)[1].item()

                return action

    def calc_delta(self, rewards, dones, Z, next_max_Z):
        # [batch_size, n_taus, quantile lenght]
        delta = torch.empty(self.batch_size, Z.shape[1], next_max_Z.shape[1], device=self.device)

        rewards_expanded = rewards.view(self.batch_size, 1, 1)  # [batch_size, 1, 1]
        dones_expanded = dones.view(self.batch_size, 1, 1)  # [batch_size, 1, 1]

        next_max_Z_expanded = next_max_Z.unsqueeze(1)  # [batch_size, 1, next_max_Z.shape[1], quantile_length]

        # 3. Z를 차원 확장
        Z_expanded = Z.unsqueeze(2)  # [batch_size, Z.shape[1], 1, quantile_length]

        # 4. delta 계산
        delta = rewards_expanded + self.gamma * (1 - dones_expanded) * next_max_Z_expanded - Z_expanded

        return delta

    def train(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            # [batch size, sample num , 1]
            taus = torch.rand(self.batch_size, self.N,  device=self.device).unsqueeze(-1)
            target_taus = torch.rand(self.batch_size, self.N_prime, device=self.device).unsqueeze(-1)
            # risk distortion.
            tilde_taus = torch.rand(self.batch_size, self.K, device= self.device).unsqueeze(-1)
            tilde_taus = self.risk_cpw(tilde_taus)
            # [batch size, n_taus , action dim ,quantile length]
            next_Z= self.target_iqn(next_states, target_taus)
            # [batch_size]
            max_next_actions = self.target_iqn(next_states, tilde_taus).mean(1).mean(2).max(1)[1]
            # [batch size, n_taus, quantile length]
            next_max_Z = next_Z[torch.arange(self.batch_size), : ,max_next_actions, : ]

        # [batch size, n_taus_i, quantile length] : 각 batch 에서 선택된 action 의 quantile distribution.
        Z_tau = self.iqn(states, taus)[torch.arange(self.batch_size), :,actions.view(-1).long(), :]
        # [batch size, n_taus_j, quantile length]
        td_target = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * next_max_Z.detach()

        # broad casting
        # [batch size, n_taus_i, n_taus_j, quantile length]
        """ 다시 확인. """
        td_error = td_target.unsqueeze(1) - Z_tau.unsqueeze(2)
        # [batch size, n_taus_i, n_taus_j, quantile length]
        quantile_huber_loss = self.huber(td_error) * (taus.unsqueeze(-1).to(self.device) - (td_error.detach() < 0).float()).abs()

        loss = quantile_huber_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()