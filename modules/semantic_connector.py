import typing as tp

import numpy as np
import torch.nn as nn

from modules.conv import StreamingConv1d, StreamingConvTranspose1d
from modules.streaming import StreamingContainer, StreamingAdd
from utils.compile import torch_compile_lazy
from modules.resample import ConvTrUpsample1d

import modules.transformer as Stransformer


class Semantic_Connector(nn.Module):
    def __init__(self, params):
        super().__init__()
        #self.semantic_upsample    = ConvTrUpsample1d(2, dimension=512, learnt=True, causal=True, channel_wise=True)
        self.semantic_transformer = Stransformer.ProjectedTransformer(**params._semantic_connector_kwargs)

    def forward(self, x, vae_sample):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = x +  vae_sample.unsqueeze(-1)
        #x = self.semantic_upsample(x)
        x = self.semantic_transformer(x)[0]
        return x