from typing import Literal, List, Optional

import torch
import torch.nn as nn


class CountDecoder(nn.Module):
    """
    Fully connected count decoder.

    Takes the embedding space z as input, and has fully connected layers
    to decode the parameters of the underlying count distributions.

    Adapted from https://github.com/Lotfollahi-lab/nichecompass/blob/main/src/nichecompass/nn/decoders.py#L172C1-L268C24;
    20.01.2026.

    Parameters
    ----------
    n_input:
        Dimensionality of embedding.
    n_output:
        Number of output nodes from the decoder (number of genes).
    n_layers:
        Number of fully connected layers used for decoding.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 n_layers: int):
        super().__init__()
        print(f"FC COUNT DECODER -> "
              f"n_input: {n_input}, "
              f"n_output: {n_output}")

        self.n_input = n_input

        if n_layers == 1:
            self.nb_means_normalized_decoder = nn.Sequential(
                nn.Linear(self.n_input, n_output, bias=False),
                nn.Softmax(dim=-1))
        elif n_layers == 2:
            self.nb_means_normalized_decoder = nn.Sequential(
                nn.Linear(self.n_input, self.n_input, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_input, n_output, bias=False),
                nn.Softmax(dim=-1))

    def forward(self,
                z: torch.Tensor,
                log_library_size: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Forward pass of the fully connected count decoder.

        Parameters
        ----------
        z:
            Tensor containing the observation-level embeddings.
        log_library_size:
            Tensor containing the log library size of the observations.

        Returns
        ----------
        nb_means:
            The mean parameters of the negative binomial distribution.
        """
        nb_means_normalized = self.nb_means_normalized_decoder(input=z)
        nb_means = torch.exp(log_library_size) * nb_means_normalized
        return nb_means

    @staticmethod
    def loss(x: torch.Tensor,
             mu: torch.Tensor,
             theta: torch.Tensor,
             eps: float=1e-8) -> torch.Tensor:
        """
        Compute count reconstruction loss according to a negative binomial model,
        which is often used to model count data such as scRNA-seq.

        Adapted from
        https://github.com/Lotfollahi-lab/nichecompass/blob/main/src/nichecompass/modules/losses.py#L169C1-L212C19;
        20.01.2026.

        Parameters
        ----------
        x:
            Ground truth log counts (dim: batch_size, n_genes).
        mu:
            Mean of the negative binomial with positive support.
            (dim: batch_size x n_genes)
        theta:
            Inverse dispersion parameter with positive support.
            (dim: n_genes)
        eps:
            Numerical stability constant.

        Returns
        ----------
        nb_loss:
            Count reconstruction loss using a negative binomial model.
        """
        log_theta_mu_eps = torch.log(theta + mu + eps)
        log_likelihood_nb = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1))
        nb_loss = torch.mean(-log_likelihood_nb.sum(-1))
        return nb_loss