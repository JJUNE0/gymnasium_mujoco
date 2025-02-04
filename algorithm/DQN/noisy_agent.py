import gymnasium as gym

import torch
import torch.optim as optim

import random

from .noisy_network import NoisyNetwork, NoisyDuelingNetwork
from .replay_memory import ReplayMemory

torch.autograd.set_detect_anomaly(True)

class NoisyDQN():
    def __init__(
            self,
            env : gym.Env,
            device,
            args,
    ):

        self.env=env
        self.device = device
        self.args = args

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.gamma = args.gamma
        self.lr = args.lr

        self.is_Dueling = args.is_dueling
        self.set_network(self.is_Dueling)

        self.buffer = ReplayMemory(self.state_dim, self.action_dim, self.device)
        self.batch_size = args.batch_size


    def set_network(self, is_Dueling):
        print("is_Dueling : ", is_Dueling)
        if is_Dueling:
            self.dqn = NoisyDuelingNetwork(self.state_dim, self.action_dim, self.device).to(self.device)
            self.target_dqn = NoisyDuelingNetwork(self.state_dim, self.action_dim, self.device).to(self.device)
        else:
            self.dqn = NoisyNetwork(self.state_dim, self.action_dim, self.device).to(self.device)
            self.target_dqn = NoisyNetwork(self.state_dim, self.action_dim, self.device).to(self.device)

        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)

    def store_sample(self,state, action, reward, next_state, done):
        self.buffer.push(state,action,reward,next_state,done)

    def get_action(self, state, epsilon):

        self.dqn.sample_noise()                 # ε~ξ''

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_values = self.dqn(state)
            action = torch.argmax(q_values).item()
            #print(action)
        return action

    def train(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        if self.is_Dueling:
            loss = self.ddqn_loss(states, actions, rewards, next_states, dones)
        else:
            loss = self.dqn_loss(states, actions, rewards, next_states, dones)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.dqn.sample_noise()                 # ε~ξ
        self.target_dqn.sample_noise()          # ε~ξ'

    def dqn_loss(self, states, actions, rewards, next_states, dones):
        q_values = self.dqn(states)
        q_values = torch.gather(input=q_values, dim=1, index=actions.long())                    # [batch_size, 1]
        # y
        next_q_values = self.target_dqn(next_states).max(1)[0].detach()                         # [batch_size]

        target_q_values = rewards.squeeze(1) + (1-dones.squeeze(1))*self.gamma*next_q_values    # [batch_size]

        td_error = q_values.squeeze(1) - target_q_values
        loss = (td_error ** 2).mean()

        return loss


    def ddqn_loss(self, states, actions, rewards, next_states, dones):
        q_values = self.dqn(states)
        q_values = torch.gather(input=q_values, dim=1, index=actions.long())  # ε~ξ             # [batch_size, 1]
        with torch.no_grad():
            # y
            next_q_values_dist = self.dqn(next_states)
            # b
            max_actions = torch.argmax(next_q_values_dist, dim=1, keepdim=True)

            next_target_q_values_dist = self.target_dqn(next_states)
            next_max_q_values = torch.gather(input=next_target_q_values_dist, dim=1, index=max_actions) # [batch_size, 1]
            target_q_values = rewards + (1 - dones) * (self.gamma) * next_max_q_values  # ε~ξ'          # [batch_size, 1]

        td_error = q_values - target_q_values
        loss = (td_error ** 2).mean()

        return loss