import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter

class FactorisedNoisyLayer(nn.Module):
    def __init__(self, in_features, out_features, device, sigma_init=0.5):
        super(FactorisedNoisyLayer, self).__init__()

        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init / math.sqrt(in_features)

        # µ^w and σ^w
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = Parameter(torch.Tensor(out_features, in_features))

        # µ^b and σ^b
        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_sigma = Parameter(torch.Tensor(out_features))

        # stationary parameter(No learnable.) ε^w_{i,j}, ε^b_j
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_sigma.data.fill_(self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)

    def sample_noise(self):
        # random noise
        # ε^w_{i,j} = f(ε_i)f(ε_j)
        epsilon_in = self.f(torch.randn(self.in_features))
        # ε^b_j = f(ε_j)
        epsilon_out = self.f(torch.randn(self.out_features))
        """ factorized gaussian noise"""
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def remove_noise(self):
        self.weight_epsilon = torch.zeros(self.out_features, self.in_features).to(self.device)
        self.bias_epsilon = torch.zeros(self.out_features).to(self.device)

    def f(self, x) -> torch.Tensor:
        return x.sign().mul(x.abs().sqrt())

class NoisyNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 device,
                 hidden_dim : int = 128
                 ):
        super(NoisyNetwork, self).__init__()

        self.layer1 = FactorisedNoisyLayer(state_dim, hidden_dim, device)
        self.layer2 = FactorisedNoisyLayer(hidden_dim, hidden_dim, device)
        self.output = FactorisedNoisyLayer(hidden_dim, action_dim, device)

    def forward(self, state):
        layer1_q = F.relu(self.layer1(state))
        layer2_q = F.relu(self.layer2(layer1_q))
        q_values = self.output(layer2_q)
        return q_values

    def sample_noise(self):
        self.layer1.sample_noise()
        self.layer2.sample_noise()
        self.output.sample_noise()

class NoisyDuelingNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 device,
                 hidden_dim : int = 128
                 ):
        super(NoisyDuelingNetwork, self).__init__()

        self.layer1 = FactorisedNoisyLayer(state_dim, hidden_dim, device)

        self.layer_V = FactorisedNoisyLayer(hidden_dim, hidden_dim, device)
        self.layer_Adv = FactorisedNoisyLayer(hidden_dim, hidden_dim, device)

        self.output_V = FactorisedNoisyLayer(hidden_dim, action_dim, device)
        self.output_Adv = FactorisedNoisyLayer(hidden_dim, action_dim, device)

    def forward(self, state):
        x = F.relu(self.layer1(state))

        v = F.relu(self.layer_V(x))
        v = self.output_V(v)

        adv = F.relu(self.layer_Adv(x))
        adv = self.output_Adv(adv)

        #print("adv : ",adv.shape)
        adv_mean = torch.mean(adv, dim = 1, keepdim=True)

        Q = v + adv - adv_mean

        return Q

    def sample_noise(self):
        self.layer1.sample_noise()

        self.layer_V.sample_noise()
        self.layer_Adv.sample_noise()

        self.output_V.sample_noise()
        self.output_Adv.sample_noise()




