import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import random

from .replay_memory import  ReplayMemory

class Categorical_Q_Network(nn.Module):
    def __init__(self,
                 state_dim  : int,
                 action_dim : int,
                 atom : torch.Tensor,
                 hidden_dim : int = 128,
                 ):
        super(Categorical_Q_Network,self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atom = atom
        self.N = len(atom)

        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.Z_value = nn.Linear(hidden_dim, action_dim * self.N)

        self.activation = nn.ReLU()

    def forward(self, state):
        x = self.activation(self.layer1(state))
        x = self.activation(self.layer2(x))

        # action_dim * support  discretization
        # Z[0] : action 0 에 대한 이산 분포(N 개로 쪼개진).
        Z = self.Z_value(x)
        # support의 차원에 softmax 적용하여 이산 확률 분포로 만듬.
        # p_i(x,a), i=0,...,N-1
        dist = torch.softmax(Z.view(len(state), self.action_dim, self.N), dim=2)

        return dist

class C51Agent():
    def __init__(
        self,
        env: gym.Env,
        device,
        args,
    ):

        self.env = env
        self.device = device
        self.args = args

        self.state_dim = self.env.observation_space.shape[0]
        print("state dim : ", self.state_dim)
        self.action_dim = self.env.action_space.n
        print("action dim : ", self.action_dim)

        self.buffer = ReplayMemory(self.state_dim, self.action_dim, self.device)
        self.batch_size = args.batch_size

        self.v_min = args.v_min
        self.v_max = args.v_max
        self.atoms_length = args.atoms_length

        self.atoms = torch.linspace(self.v_min, self.v_max, steps=self.atoms_length, device=self.device)    # [atoms_length]
        self.delta_z = (self.v_max - self.v_min) / (len(self.atoms) - 1)
        # m_i = 0, i ∈ [0, atoms_length-1]
        self.m = torch.zeros((self.batch_size, self.atoms_length), device=self.device)
        # offset = tensor([0, atoms_length, 2*atoms_length, ..., (batch_size-1)*atoms_length])
        self.offset = torch.linspace(0, (self.batch_size - 1) * self.atoms_length, self.batch_size,
                                     device=self.device).unsqueeze(-1).long()

        self.c51 = Categorical_Q_Network(self.state_dim, self.action_dim, self.atoms).to(self.device)
        self.target_c51 = Categorical_Q_Network(self.state_dim, self.action_dim, self.atoms).to(self.device)
        self.target_c51.load_state_dict(self.c51.state_dict())

        self.gamma = args.gamma  # discount rate
        self.lr = args.lr

        self.optimizer = optim.Adam(self.c51.parameters(), lr=self.lr)

        self.dist = torch.empty_like(self.atoms)

    def hard_update(self, network = None, target_network = None):
        if network == None:
            network = self.c51
        if target_network == None:
            target_network = self.target_c51

        with torch.no_grad():
            for param, target_param in zip(network.parameters(), target_network.parameters()):
                target_param.data.copy_(param.data)

    def store_sample(self,state, action, reward, next_state, done):
        self.buffer.push(state,action,reward,next_state,done)

    def get_action(self, state, epsilon):

        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                z_value = self.c51(state)                               # ([1, action_dim, N])
                # atoms 차원에서 expectation 함으로써 E[Z] = Q 구함.
                q_value = (z_value * self.atoms).sum(dim=2)             # ([1, action_dim])
                action =  torch.argmax(q_value, dim=1).item()           # action]

            return action

    def train(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        loss = self.ddqn_loss(states, actions, rewards, next_states, dones)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def dqn_loss(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():

            next_dists = self.target_c51(next_states)       # (batch_size, action_dim, n_atoms)
            max_next_dists = next_dists.max(1)[0]           # (batch_size, n_atoms)
            #max_next_actions = next_dists.max(1)[1]

            # init m to 0
            self.m *= 0
            # [Shift reward & gamma-contraction].clamp(V_min,V_max)
            T_z = (rewards + (1-dones) * self.gamma * self.atoms).clamp(self.v_min, self.v_max)     # (batch_size, n_atoms)
            # 범위 맞춰줌.
            b = (T_z - self.v_min) / self.delta_z  # b ∈ [0,n_atoms-1];                               (batch_size, n_atoms)

            # Projection 을 위해 내림과 올림.
            l = b.floor().long()                            # (batch_size, n_atoms)
            u = b.ceil().long()                             # (batch_size, n_atoms)

            # Projection.
            # index_add_(dim, index, value):
            delta_m_l = (u + (l == u) - b) * max_next_dists     # (batch_size, n_atoms)
            delta_m_u = (b - l) * max_next_dists                # (batch_size, n_atoms)

            '''Distribute probability with tensor operation. Much more faster than the For loop in the original paper.'''
            self.m.view(-1).index_add_(0, (l + self.offset).view(-1), delta_m_l.view(-1))
            self.m.view(-1).index_add_(0, (u + self.offset).view(-1), delta_m_u.view(-1))

        actions = actions.unsqueeze(-1).expand(-1,1,self.atoms_length).long()                       # (batch_size, 1, n_atoms)
        curr_dists = torch.gather(input=self.c51(states), dim=1, index=actions).squeeze(1)                 # (batch_size, n_atoms)
        # Compute Corss Entropy Loss:
        # q_loss = (-(self.m * batched_distribution.log()).sum(-1)).mean() # Original Cross Entropy loss, not stable
        loss = (-(self.m * curr_dists.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()  # more stable

        self.dist = curr_dists[0]

        return loss

    def ddqn_loss(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            """ Using Double dqn"""
            next_argmax_actions = self.c51(next_states).max(1)[1].unsqueeze(1)              # (batch_size, 1, atoms_n)
            #print(next_argmax_actions.shape)
            next_target_dists = torch.gather(self.target_c51(next_states), dim=1, index=next_argmax_actions).squeeze(1)  # [batch_size, atoms_n]

            # init m to 0
            self.m *= 0
            # [Shift reward & gamma-contraction].clamp(V_min,V_max)
            T_z = (rewards + (1 - dones) * self.gamma * self.atoms).clamp(self.v_min, self.v_max)  # (batch_size, n_atoms)
            # 범위 맞춰줌.
            b = (T_z - self.v_min) / self.delta_z  # b ∈ [0,n_atoms-1];                               (batch_size, n_atoms)

            # Projection 을 위해 내림과 올림.
            l = b.floor().long()  # (batch_size, n_atoms)
            u = b.ceil().long()  # (batch_size, n_atoms)

            # Projection.
            # index_add_(dim, index, value):
            delta_m_l = (u + (l == u) - b) * next_target_dists  # (batch_size, n_atoms)
            delta_m_u = (b - l) * next_target_dists  # (batch_size, n_atoms)

            '''Distribute probability with tensor operation. Much more faster than the For loop in the original paper.'''
            self.m.view(-1).index_add_(0, (l + self.offset).view(-1), delta_m_l.view(-1))
            self.m.view(-1).index_add_(0, (u + self.offset).view(-1), delta_m_u.view(-1))

        actions = actions.unsqueeze(-1).expand(-1, 1, self.atoms_length).long()  # (batch_size, 1, n_atoms)
        curr_dists = torch.gather(input=self.c51(states), dim=1, index=actions).squeeze(1)  # (batch_size, n_atoms)
        # Compute Corss Entropy Loss:
        # loss = (-(self.m * curr_dists.log()).sum(-1)).mean() # Original Cross Entropy loss, not stable
        loss = (-(self.m * curr_dists.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()  # more stable

        self.dist = curr_dists[0]

        return loss

    def record(self):
        return self.dist