import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size = 128):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        layer1 = torch.relu(self.layer1(state))
        layer2 = torch.relu(self.layer2(layer1))
        Qvalue = self.value(layer2)
        return Qvalue

class DuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 128):
        super(DuelingNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, hidden_size)

        self.layer_V = nn.Linear(hidden_size, hidden_size)
        self.output_V = nn.Linear(hidden_size, 1)

        self.layer_Adv = nn.Linear(hidden_size, hidden_size)
        self.output_Adv = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.layer1(state))

        v = torch.relu(self.layer_V(x))
        v = self.output_V(v)
        #print("v : ",v.shape)
        adv = torch.relu(self.layer_Adv(x))
        adv = self.output_Adv(adv)
        #print("adv : ", adv.shape)

        adv_mean = torch.mean(adv, dim=1, keepdim=True)
        #print("adv mean : ",adv_mean.shape)
        # 형변환 알아서 잘 되나보네?
        Q = v + adv - adv_mean
        #print(Q.shape)


        return Q