import torch.nn as nn
import torch.nn.functional as F

class LOBMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes:list=[100], output_size=3):
        super().__init__()
        self.fc_list = nn.ModuleList()
        self.fc_list.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.fc_list.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.fc_list.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for fc in self.fc_list[0:-1]:
            x = F.leaky_relu(fc(x))
        x = self.fc_list[-1](x)
        return x