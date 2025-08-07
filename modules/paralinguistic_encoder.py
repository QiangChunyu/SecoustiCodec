from typing import Dict, Iterable, Optional
from math import sqrt

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class ParalinguisticEncoder(nn.Module):
    def __init__(self,params):
        super(ParalinguisticEncoder, self).__init__()
        self.conv2d_1    = nn.Conv2d(params.n_mels, 32, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.batchnorm_1 = nn.BatchNorm2d(32)
        self.conv2d_2    = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.batchnorm_2 = nn.BatchNorm2d(32)
        self.conv2d_3    = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.batchnorm_3 = nn.BatchNorm2d(64)
        self.conv2d_4    = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.batchnorm_4 = nn.BatchNorm2d(64)
        self.conv2d_5    = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.batchnorm_5 = nn.BatchNorm2d(128)
        self.conv2d_6    = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.batchnorm_6 = nn.BatchNorm2d(128)
        self.senet       = SEResNetLayer(128)
        self.gru         = nn.GRU(128,64, batch_first=True)
        self.dense_mu    = nn.Linear(64, params.embed_dim)
        self.dense_var   = nn.Linear(64, params.embed_dim)
    
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
        x = x.transpose(1, 2).unsqueeze(-1)
        x = self.conv2d_1(x)
        x = self.batchnorm_1(x)
        x = F.relu(x)
        x = self.conv2d_2(x)
        x = self.batchnorm_2(x)
        x = F.relu(x)
        x = self.conv2d_3(x)
        x = self.batchnorm_3(x)
        x = F.relu(x)
        x = self.conv2d_4(x)
        x = self.batchnorm_4(x)
        x = F.relu(x)
        x = self.conv2d_5(x)
        x = self.batchnorm_5(x)
        x = F.relu(x)
        x = self.conv2d_6(x)
        x = self.batchnorm_6(x)
        x = F.relu(x)
        x = self.senet(x)
        x = x.transpose(1, 2).squeeze(-1)
        _, hidden_state = self.gru(x)
        hidden_state = hidden_state.squeeze(0)
        mu = self.dense_mu(hidden_state)
        logvar = self.dense_var(hidden_state)

        # sample latent attributes
        vae_sample = self.reparameterize(mu, logvar)
        return vae_sample, mu, logvar

class SEResNetLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEResNetLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)+x
