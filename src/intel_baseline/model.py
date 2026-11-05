import torch
from torch import nn


class IntelAutoencoder(nn.Module):
    def __init__(self, n_bins, bottleneck=48, hidden=1024):
        super().__init__()
        self.standardizer = nn.BatchNorm1d(n_bins, affine=False)
        channels = [n_bins, hidden, hidden, hidden, hidden, bottleneck]
        layers = []
        for source, target in zip(channels, channels[1:]):
            layers.extend([nn.Conv1d(source, target, 3), nn.LeakyReLU(0.01)])
        self.encoder = nn.Sequential(*layers)
        channels = list(reversed(channels))
        layers = []
        for index, (source, target) in enumerate(zip(channels, channels[1:])):
            layers.extend([nn.ConvTranspose1d(source, target, 3),
                           nn.PReLU(init=0.25) if index == 4 else nn.LeakyReLU(0.01)])
        self.decoder = nn.Sequential(*layers)

    def forward(self, features):
        if features.ndim != 3 or features.shape[-1] < 11:
            raise ValueError("Expected [batch, frequency, time] with at least eleven frames")
        target = self.standardizer(features)
        return self.decoder(self.encoder(target)), target.detach()


def masked_mse(reconstruction, target, lengths):
    mask = torch.arange(target.shape[-1], device=target.device)[None, :] < lengths[:, None]
    squared = (reconstruction - target).square()
    return (squared * mask[:, None, :]).sum() / (mask.sum() * target.shape[1])
