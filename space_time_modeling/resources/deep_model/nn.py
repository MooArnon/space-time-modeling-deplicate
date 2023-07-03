import torch.nn as nn
import torch.func  as F

class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NNModel, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x