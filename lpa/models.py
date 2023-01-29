import torch
import torch.nn as nn
import numpy as np

class NeuralNetwork(nn.Module):

    '''
    Class for a simple 2-layer NN
    '''

    def __init__(self, num_inputs, num_outputs):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )

    def forward(self, x):
        out = self.layers(x)
        return out
    
    def predict(self, x):
        out = self.layers(x)
        return torch.max(out, 1)[1].detach().numpy()
        