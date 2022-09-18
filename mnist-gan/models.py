import torch.nn as nn
import torch 

class Generator(nn.Module):
    def __init__(self, in_features=100, out_features=28*28, nker=128):
        super().__init__()

        def Block(in_features, out_features, normalize=True):
            layers = []
            layers.append(nn.Linear(in_features, out_features))
            
            if normalize:
                layers.append(nn.BatchNorm1d(out_features))
            
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)
    

        self.fc1 = Block(in_features, nker, normalize=False)
        self.fc2 = Block(nker, 2*nker)
        self.fc3 = Block(2*nker, 4*nker)
        self.fc4 = Block(4*nker, 8*nker)
        self.fc5 = nn.Sequential(*[nn.Linear(8*nker, out_features), nn.Tanh()])
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_features=28*28, out_features=512):
        super().__init__()
        
        self.fc1 = nn.Sequential(*[
            nn.Linear(in_features, out_features),
            nn.ReLU()
        ])

        self.fc2 = nn.Sequential(*[
            nn.Linear(out_features, 256),
            nn.ReLU()
        ])

        self.fc3 = nn.Sequential(*[
            nn.Linear(256, 1),
            nn.Sigmoid()
        ])
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



