import typing as tp
import numpy as np
import torch
import torch.nn as nn
import modules.transformer as Stransformer

class SemanticTransformerVAE(nn.Module):
    def __init__(self, params):
        super(SemanticTransformerVAE, self).__init__()
        self.semantic_transformer = Stransformer.ProjectedTransformer(**params._semantic_transformer_kwargs)
        self.dense_mu    = nn.Linear(512, 512)
        self.dense_var   = nn.Linear(512, 512)
    
    def reparameterize(self, mu, logvar):
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        semantic_embed = self.semantic_transformer(x)[0].transpose(1, 2)
        mu = self.dense_mu(semantic_embed)
        logvar = self.dense_var(semantic_embed)
        # sample latent attributes
        vae_sample = self.reparameterize(mu, logvar).transpose(1, 2)
        return vae_sample, mu, logvar
