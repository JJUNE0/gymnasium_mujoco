import gymnasium as gym

import torch
import torch.optim as optim

import random

from .network import Network, DuelingNetwork
from .replay_memory import ReplayMemory


class DQN():
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
        self.action_size = self.env.action_space.n
        print("action dim : ", self.action_size)
        self.tau = args.tau

        self.gamma = 0.99  # discount rate
        self.lr = 0.001

        self.is_dueling = args.is_dueling
        self.set_network(self.is_dueling)

        # default capcity : 1e6
        self.buffer = ReplayMemory(self.state_size, self.action_size, self.device)
        # 64
        self.batch_size = args.batch_size


    def set_network(self, is_Dueling):
        if is_Dueling:
            self.dqn = DuelingNetwork(self.state_size, self.action_size).to(self.device)
            self.target_dqn = DuelingNetwork(self.state_size, self.action_size).to(self.device)
        else:
            self.dqn = Network(self.state_size, self.action_size).to(self.device)
            self.target_dqn = Network(self.state_size, self.action_size).to(self.device)

        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)                            # [1, state_dim]
            q_value = self.dqn(state).detach()                                                       # [action_dim]
            return q_value.argmax().item()                                                           # [argmax action index]

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

        if self.is_dueling:
            loss = self.ddqn_loss(states, actions, next_states, rewards, dones)
        else:
            loss = self.dqn_loss(states, actions, next_states, rewards, dones)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def dqn_loss(self, states, actions, next_states, rewards, dones):
        # actions : [batch_size, selected_action]
        next_Qs = self.target_dqn(next_states).max(1)[0].detach()  # [batch_size]

        # gather(input, dim, index)
        curr_Qs = self.dqn(states).gather(1, actions.long()).squeeze(1)  # [batch_size] : selected action value

        y = rewards.squeeze(1) + self.gamma * next_Qs * (1 - dones.squeeze(1))  # [batch_size]

        loss = torch.nn.functional.mse_loss(y, curr_Qs)

        return loss

    def ddqn_loss(self, states, actions, next_states, rewards, dones):

        with torch.no_grad():
            # torch.argmax 에서 dim 을 기준으로 index 선택
            next_argmax_actions = torch.argmax(self.dqn(next_states), dim = 1, keepdim=True)                            # [batch_size , 1]
            next_target_Qs = torch.gather(self.target_dqn(next_states), dim=1, index=next_argmax_actions).squeeze(1)    # [batch_size]

            y = rewards.squeeze(1) + self.gamma * next_target_Qs * (1-dones.squeeze(1))                                 # [batch_size]

        curr_Qs= self.dqn(states).gather(1, actions.long()).squeeze(1)                                                  # [bathc_size]

        loss = torch.nn.functional.mse_loss(y, curr_Qs)

        return loss

