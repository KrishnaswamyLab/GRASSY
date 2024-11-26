from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

class GrassyAutoencoder(nn.Module):
    '''
    Simple autoencoder using an MLP to encode into latent space.
    '''
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GrassyAutoencoder, self).__init__()

        self.lins = nn.ModuleList()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        for _ in range(3):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, scattering_moments_BS):
        """
        B: Batch size
        S: scattering moments (number of features)
        H: hidden dimension
        Z: latent dim

        Input Dimension: B x S
        Output Dimension: B x Z
        """

        z_rep_BH = F.relu(self.fc1(scattering_moments_BS))
        for lin in self.lins:
            z_rep_BH = F.relu(lin(z_rep_BH))
        z_rep_BZ = self.fc3(z_rep_BH)

        return z_rep_BZ