import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gdl


class AE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform, pool=None):
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)
        self.pool = pool

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.en_layers.append(
                    gdl.SpiralEnblock(self.in_channels, self.out_channels[idx],
                                      self.spiral_indices[idx], self.pool))
            else:
                self.en_layers.append(
                    gdl.SpiralEnblock(self.out_channels[idx - 1], self.out_channels[idx],
                                      self.spiral_indices[idx], self.pool))
        for idx in range(len(self.latent_channels)):
            if idx == 0:
                self.en_layers.append(
                    nn.Linear(self.num_vert * self.out_channels[-1], self.latent_channels[idx]))
            else:
                self.en_layers.append(
                    nn.Linear(self.latent_channels[idx - 1], self.latent_channels[idx]))

        # decoder
        self.de_layers = nn.ModuleList()
        for idx in range(len(self.latent_channels)):
            if idx == len(self.latent_channels) - 1:
                self.de_layers.append(
                    nn.Linear(self.latent_channels[-idx - 1], self.num_vert * self.out_channels[-1]))
            else:
                self.de_layers.append(
                    nn.Linear(self.latent_channels[-idx - 1], self.latent_channels[-idx - 2]))

        for idx in range(len(self.out_channels)):
            if idx == len(self.out_channels) - 1:
                self.de_layers.append(
                    gdl.SpiralDeblock(self.out_channels[-idx - 1],
                                      self.in_channels,
                                      self.spiral_indices[-idx - 1], act=False))
            else:
                self.de_layers.append(
                    gdl.SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 2],
                                      self.spiral_indices[-idx - 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if not isinstance(layer, nn.Linear):
                x = layer(x, self.down_transform[i])
            else:
                if not isinstance(self.en_layers[i - 1], nn.Linear):
                    x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - len(self.up_transform) - 1
        for i, layer in enumerate(self.de_layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if not isinstance(self.de_layers[i + 1], nn.Linear):
                    x = x.view(-1, self.num_vert, self.out_channels[-1])
            else:
                x = layer(x, self.up_transform[num_features - i])
        return x

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return z, out




