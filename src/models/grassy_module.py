from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from lightning.fabric.utilities.rank_zero import rank_zero_only

from torch import nn
import torch.nn.functional as F
from src.models.components.legs_model import LearnableScattering
from src.models.components.autoencoder import GrassyAutoencoder
from src.models.components.property_prediction_model import PropertyPredictionModule
from torchmetrics import MeanMetric, MeanSquaredError


class GrassyModule(LightningModule):

    def __init__(self, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler, net) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net

        self.input_dim = self.hparams.net.input_dim  #number of unique atoms
        self.hidden_dim_autoencoder = self.hparams.net.hidden_dim_autoencoder  #number of hidden dimensions for the autoencoder
        self.latent_dim = self.hparams.net.latent_dim  #latent dimensions
        self.hidden_dim_property_predictor = self.hparams.net.hidden_dim_property_predictor  #number of hidden dimensions for the property prediction module

        self.property_dim = self.hparams.net.property_dim  #number of properties to predict
        self.alpha = self.hparams.net.alpha  #weight for the reconstruction loss

        self.train_loss = MeanMetric()
        self.train_mse = MeanSquaredError(num_outputs=self.property_dim)

        print("Initializing scattering..")
        self.scattering_network = LearnableScattering(self.input_dim)

        # import pdb
        # pdb.set_trace()
        self.encoder = GrassyAutoencoder(self.scattering_network.out_shape(),
                                         self.hidden_dim_autoencoder,
                                         self.latent_dim)

        self.decoder = GrassyAutoencoder(self.latent_dim,
                                         self.hidden_dim_autoencoder,
                                         self.scattering_network.out_shape())

        self.property_predictor = PropertyPredictionModule(
            self.latent_dim, self.hidden_dim_property_predictor,
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
        latent_representations = self.encoder(scatter_moments_NS)

        # Reconstruction of the scattering moments
        scatter_moments_reconstructed = self.decoder(latent_representations)

        # Properties prediction
        properties = self.property_predictor(latent_representations)

        return scatter_moments_NS, properties, scatter_moments_reconstructed, latent_representations

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.

        """
        model_outputs = self.model_step(batch)
        loss = self.alpha * model_outputs[
            "train/scattering_reconstruction_loss"] + (
                1 -
                self.alpha) * model_outputs["train/property_prediction_loss"]
        # import pdb;pdb.set_trace()

        if torch.isnan(loss).any():
            print("NaN encountered")
            loss = torch.zeros_like(loss, requires_grad=True).to(loss.device)

        # update and log metrics
        total_loss = self.train_loss(loss)
        train_mse = self.train_mse(model_outputs["predicted_properties"],
                                   batch.y)

        self.log("train/scattering_reconstruction_loss",
                 self.alpha *
                 model_outputs["train/scattering_reconstruction_loss"],
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch.y.size(0))

        self.log("train/property_prediction_loss", (1 - self.alpha) *
                 model_outputs["train/property_prediction_loss"],
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch.y.size(0))

        self.log("train/loss",
                 total_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch.y.size(0))
        self.log("train/mse",
                 train_mse.mean(),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch.y.size(0))

        # return loss or backpropagation will fail
        return loss

    # def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
    #     '''
    #     Running evaluation on a single GPU
    #     - Generate Molecule tensor using predict()
    #     '''
    #     pass

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, properties = batch.x, batch.y
        # print (torch.isnan(properties).any())
        scattering_moments, predicted_properties, scatter_moments_reconstructed, _ = self.forward(
            batch)
        scattering_reconstruction_loss = self.scattering_reconstruction_loss(
            scatter_moments_reconstructed, scattering_moments)
        property_prediction_loss = self.property_prediction_loss(
            predicted_properties, properties)
        model_outputs = {
            "train/scattering_reconstruction_loss":
            scattering_reconstruction_loss,
            "train/property_prediction_loss": property_prediction_loss,
            "predicted_properties": predicted_properties,
            'scattering_moments': scattering_moments,
            'scatter_moments_reconstructed': scatter_moments_reconstructed
        }

        # import pdb;pdb.set_trace()

        return model_outputs

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                     batch_idx: int) -> Dict[str, Any]:
        scatter_moments_NS = self.scattering_network(batch)
        latents = self.encoder(scatter_moments_NS)
        props = self.property_predictor(latents)

        torch.save(latents.to("cpu"),
                   f"saved_tensors_12k_chaffer/latents_{batch_idx}.pt")
        torch.save(props.to("cpu"),
                   f"saved_tensors_12k_chaffer/props_{batch_idx}.pt")

        return latents, props

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer(params=self.parameters())

    # def on_validation_epoch_end(self) -> None:
    #     "Lightning hook that is called when a validation epoch ends."
    #     pass

    def scattering_reconstruction_loss(self, preds, targets) -> torch.Tensor:
        ''''
        Reconstruction loss for the autoencoder to reconstruct the scattering moments
        '''
        loss = nn.MSELoss()(preds, targets)
        return loss

    def property_prediction_loss(self, preds, targets) -> torch.Tensor:
        ''''
        Loss function for the property prediction module
        '''
        loss = nn.MSELoss()(preds, targets)
        return loss

    def contrastive_loss(self,
                         preds,
                         targets,
                         temperature=0.5) -> torch.Tensor:
        '''
        Contrastive loss for the latent space using Normalized Temp Cross entropy loss
        '''
        batch_size = preds.size(0)
        preds = F.normalize(preds, dim=1)
        targets = F.normalize(targets, dim=1)

        representations = torch.cat([preds, targets], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)

        # removing self-similarity
        mask = torch.eye(2 * batch_size, device=representations.device).bool()
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

        # labelling +ve pairs
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)],
                           dim=0).to(representations.device)
        labels = labels.repeat_interleave(
            similarity_matrix.size(1) // (2 * batch_size))

        # Scale similarities by temperature
        similarity_matrix = similarity_matrix / temperature

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    # Function used to garantee that only one GPU will log the metrics.
    def _log_scalar(
        self,
        key,
        value,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        batch_size=None,
        sync_dist=False,
        rank_zero_only=True  # NOTE: it's called here <- verify this with Alex
    ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(key,
                 value,
                 on_step=on_step,
                 on_epoch=on_epoch,
                 prog_bar=prog_bar,
                 batch_size=batch_size,
                 sync_dist=sync_dist,
                 rank_zero_only=rank_zero_only)

    # @rank_zero_only
    # def validation_helper(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
    #     ''''
    #     Helper function for on_validation_epoch_end()
    #     - Save generated molecule as .sdf
    #     - Run eval_object()
    #     - Upload metrics to Weights and Biases (CODE from Rishabh)
    #     '''
    #     pass
