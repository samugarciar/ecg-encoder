"""Autoencoder convolucional 1D para latidos de ECG (D-009, D-022)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _down_block(c_in: int, c_out: int, kernel_size: int) -> nn.Sequential:
    """Convolución con paso 2: extrae patrones locales y reduce el largo a la mitad."""
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size, stride=2, padding=kernel_size // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
    )


def _up_block(c_in: int, c_out: int, kernel_size: int) -> nn.Sequential:
    """Duplica el largo por interpolación lineal y luego aplica una convolución."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
        nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
    )


class ConvAutoencoder(nn.Module):
    """Comprime un latido de `input_length` muestras a `latent_dim` valores y lo reconstruye.

    Entrada y salida con forma (lote, 1, input_length). Cada nivel del encoder reduce el
    largo a la mitad y el decoder recorre el camino inverso; el cuello es una capa densa.
    """

    def __init__(
        self,
        input_length: int = 256,
        channels: Sequence[int] = (16, 32, 64, 128),
        kernel_size: int = 7,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        n_levels = len(channels)
        if n_levels == 0 or input_length % 2**n_levels:
            raise ValueError(
                f"input_length={input_length} debe ser divisible por 2**{n_levels} "
                "(cada nivel reduce el largo a la mitad)"
            )
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size={kernel_size} debe ser impar para conservar el largo")

        self.input_length = input_length
        self.channels = tuple(channels)
        self.latent_dim = latent_dim
        self.bottleneck_length = input_length // 2**n_levels

        widths = (1, *self.channels)
        self.encoder = nn.Sequential(
            *(_down_block(widths[i], widths[i + 1], kernel_size) for i in range(n_levels))
        )
        flat = self.channels[-1] * self.bottleneck_length
        self.to_latent = nn.Linear(flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat)

        reverse = widths[::-1]  # por defecto: 128, 64, 32, 16, 1
        self.decoder = nn.Sequential(
            *(_up_block(reverse[i], reverse[i + 1], kernel_size) for i in range(n_levels - 1)),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(reverse[-2], 1, kernel_size, padding=kernel_size // 2),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latido (lote, 1, L) → representación latente (lote, latent_dim)."""
        return self.to_latent(self.encoder(x).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Representación latente (lote, latent_dim) → latido reconstruido (lote, 1, L)."""
        h = self.from_latent(z).view(-1, self.channels[-1], self.bottleneck_length)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def count_parameters(model: nn.Module) -> int:
    """Número de parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
