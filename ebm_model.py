import torch.nn as nn


###### Energy-Based Model Implementation ######
class SmallEBM(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, num_layers=4, dropout=0.2):
        super(SmallEBM, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
            hidden_dim = max(hidden_dim // 2, 64)  # Gradually reduce dimensions
        
        # Final output layer
        layers.append(nn.Linear(current_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze()

    def energy(self, x):
        return self.forward(x)
