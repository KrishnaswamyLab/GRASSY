from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

class PropertyPredictionModule(nn.Module):
    ''' 
    Simple MLP to predict properties from latent space.
    '''
    def __init__(self, latent_dim, hidden_dim, property_dim):
        super(PropertyPredictionModule, self).__init__()

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, property_dim)
    
    def forward(self, latent_representations_BL):
        """
        TODO: Verify the dimensions on the comments

        B: Batch size
        L: Latent representation
        H: hidden dimension
        P: Property dimension

        Input Dimension: B x L
        Output Dimension: B x P
        """

        z_rep_BH = F.relu(self.fc1(latent_representations_BL))
        z_rep_BH = F.relu(self.fc2(z_rep_BH))
        z_rep_BP = self.fc3(z_rep_BH)

        return z_rep_BP