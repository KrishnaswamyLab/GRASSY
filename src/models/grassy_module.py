from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule

from components.legs_model import LearnableScattering
from components.autoencoder import GrassyAutoencoder
from components.property_prediction_model import PropertyPredictionModule

class GrassyModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_autoencoder: int,
        latent_dim: int,
        hidden_dim_property_predictor: int,
        property_dim: int,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.input_dim = input_dim #number of unique atoms
        self.hidden_dim_autoencoder = hidden_dim_autoencoder #number of hidden dimensions for the autoencoder

        self.latent_dim = latent_dim #latent dimensions 
        self.hidden_dim_property_predictor = hidden_dim_property_predictor #number of hidden dimensions for the property prediction module

        self.property_dim = property_dim #number of properties to predict

        print("Initializing scattering..")
        self.scattering_network = LearnableScattering(self.input_dim, trainable_f=True)

        self.autoencoder = GrassyAutoencoder(self.scattering_network.out_shape(),
                                             self.hidden_dim_autoencoder,
                                             self.latent_dim)

        self.property_predictor = PropertyPredictionModule(self.latent_dim,
                                                           self.hidden_dim_property_predictor,
                                                           self.property_dim)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        ''' 
        Forward pass the signals through the LEGS(Learnable Scattering model)
        data is a dict 
                {
                    x: Signals of shape [Number of nodes(N),Number of unique atoms(A)], 
                    edge_index: Adjacency matrix
                }
        N: Number of nodes
        S: Number of scattering moments
        '''

        # Get the scattering moments
        scatter_moments_NS = self.scattering_network(data)

        # Project on to latent space
        latent_representations_scattering_moments = self.autoencoder(scatter_moments_NS)

        # Properties prediction
        properties = self.property_predictor(latent_representations_scattering_moments)
    
        return properties

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        '''
        Running evaluation on a single GPU
        - Generate Molecule tensor using predict()
        '''
        pass

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str) -> Dict[str, Any]:
        pass

    def predict(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int) -> Dict[str, Any]:
        pass

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        pass

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    @rank_zero_only
    def validation_helper(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        ''''
        Helper function for on_validation_epoch_end()
        - Save generated molecule as .sdf
        - Run eval_object()
        - Upload metrics to Weights and Biases (CODE from Rishabh)
        '''
        pass
