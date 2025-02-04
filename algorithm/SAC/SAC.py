import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from gymnasium_mujoco.algorithm.TD3.replay_memory import ReplayMemory

class Twin_Q_net(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dims=(256, 256), activation_fc=F.relu):
        super(Twin_Q_net, self).__init__()
        self.device = device

        self.activation_fc = activation_fc

        self.input_layer_A = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers_A = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_A = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_A.append(hidden_layer_A)
        self.output_layer_A = nn.Linear(hidden_dims[-1], 1)

        self.input_layer_B = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers_B = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_B = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_B.append(hidden_layer_B)
        self.output_layer_B = nn.Linear(hidden_dims[-1], 1)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)
        x = torch.cat([x, u], dim=1)

        x_A = self.activation_fc(self.input_layer_A(x))
        for i, hidden_layer_A in enumerate(self.hidden_layers_A):
            x_A = self.activation_fc(hidden_layer_A(x_A))
        x_A = self.output_layer_A(x_A)

        x_B = self.activation_fc(self.input_layer_B(x))
        for i, hidden_layer_B in enumerate(self.hidden_layers_B):
            x_B = self.activation_fc(hidden_layer_B(x_B))
        x_B = self.output_layer_B(x_B)

        return x_A, x_B


class GaussianPolicy(nn.Module):
    def __init__(self, args, state_dim, action_dim, action_bound,
                 hidden_dims=(256, 256), activation_fc=F.relu, device='cuda'):
        super(GaussianPolicy, self).__init__()
        self.device = device

        self.log_std_min = args.log_std_bound[0]
        self.log_std_max = args.log_std_bound[1]

        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)

        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        self.action_rescale = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)


    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        distribution = Normal(mean, log_std.exp())

        unbounded_action = distribution.rsample()
        # [Paper: Appendix C] Enforcing Action Bounds: [a_min, a_max] -> [-1, 1]
        bounded_action = torch.tanh(unbounded_action)
        action = bounded_action * self.action_rescale + self.action_rescale_bias

        # We must recover ture log_prob from true distribution by 'The Change of Variable Formula'.
        log_prob = distribution.log_prob(unbounded_action) - torch.log(self.action_rescale *
                                                                       (1 - bounded_action.pow(2).clamp(0, 1)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean) * self.action_rescale + self.action_rescale_bias

        # action : 정규 분포 sampling
        # mean   :
        return action, log_prob, mean



class SAC:
    def __init__(
        self,
        env: gym.Env,
        device,
        args,
    ):
        self.env = env
        self.args = args

        self.state_size = self.env.observation_space.shape[0]
        print("state dim : ", self.state_size)
        self.action_size = len(env.action_space.high)
        print("action dim : ", self.action_size)
        self.min_action = env.action_space.low[0]
        print("min action : ", self.min_action)
        self.max_action = env.action_space.high[0]
        print("max_action : ", self.max_action)
        self.action_bound = [self.min_action, self.max_action]

        self.device = device
        self.buffer = ReplayMemory(self.state_size, self.action_size, device, args.buffer_size)
        self.batch_size = args.batch_size

        self.gamma = args.gamma
        self.tau = args.tau

        self.actor = GaussianPolicy(args, self.state_size, self.action_size, self.action_bound, args.hidden_dims, F.relu, device).to(
            device)
        # self.target_actor = GaussianPolicy(args, state_dim, action_dim, action_bound, args.hidden_dims, F.relu, device).to(device)
        self.critic = Twin_Q_net(self.state_size, self.action_size, device, args.hidden_dims).to(device)
        self.target_critic = Twin_Q_net(self.state_size, self.action_size, device, args.hidden_dims).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # Automating Entropy Adjustment for Maximum Entropy RL
        if args.automating_temperature is True:
            self.target_entropy = -torch.prod(torch.Tensor((self.action_size,))).to(device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.temperature_lr)
        else:
            self.log_alpha = torch.log(torch.tensor(args.temperature, device=device, dtype=torch.float32))

        # hard_update(self.actor, self.target_actor)
        self.hard_update(self.critic, self.target_critic)

    def hard_update(self, network, target_network):
        with torch.no_grad():
            for param, target_param in zip(network.parameters(), target_network.parameters()):
                target_param.data.copy_(param.data)

    def soft_update(self, network, target_network, tau):
        with torch.no_grad():
            for param, target_param in zip(network.parameters(), target_network.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)



    def get_action(self, state, evaluation=True):
        with torch.no_grad():
            if evaluation:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def train_actor(self, states, next_states, args, train_alpha=True):
        self.actor_optimizer.zero_grad()
        actions, log_pis, mean_actions = self.actor.sample(states)
        q_values_A, q_values_B = self.critic(states, actions)
        q_values = torch.min(q_values_A, q_values_B)

        if args.use_CAPS:
            _, _, next_mean_actions = self.actor.sample(next_states)

            noise = torch.normal(torch.zeros_like(states), torch.ones_like(states))
            noisy_states = states + noise * self.args.eps_p
            _, _, actions_from_noisy_states = self.actor.sample(noisy_states)

            actor_loss = (self.log_alpha.exp().detach() * log_pis - q_values).mean()
            actor_loss += args.lambda_t * ((next_mean_actions - mean_actions) ** 2).mean()
            actor_loss += args.lambda_s * ((actions_from_noisy_states - mean_actions) ** 2).mean()
        else:
            actor_loss = (self.log_alpha.exp().detach() * log_pis - q_values).mean()

        actor_loss.backward()
        self.actor_optimizer.step()

        if train_alpha:
            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.)

        return actor_loss.item(), alpha_loss.item()

    def train_critic(self, states, actions, rewards, next_states, dones):
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            next_actions, next_log_pis, _ = self.actor.sample(next_states)
            next_q_values_A, next_q_values_B = self.target_critic(next_states, next_actions)
            next_q_values = torch.min(next_q_values_A, next_q_values_B) - self.log_alpha.exp() * next_log_pis
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        q_values_A, q_values_B = self.critic(states, actions)
        critic_loss = ((q_values_A - target_q_values) ** 2).mean() + ((q_values_B - target_q_values) ** 2).mean()

        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def train(self, args):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        critic_loss = self.train_critic(states, actions, rewards, next_states, dones)
        if self.args.automating_temperature is True:
            actor_loss, log_alpha_loss = self.train_actor(states, next_states, args, train_alpha=True)
        else:
            actor_loss, log_alpha_loss = self.train_actor(states, next_states, args, train_alpha=False)

        # soft_update(self.actor, self.target_actor, self.tau)
        self.soft_update(self.critic, self.target_critic, self.tau)

        return critic_loss, actor_loss, log_alpha_loss