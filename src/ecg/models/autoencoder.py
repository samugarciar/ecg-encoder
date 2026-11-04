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


def upsample2x_linear(x: torch.Tensor) -> torch.Tensor:
    """Interpolación lineal ×2 (align_corners=False) escrita con desplazamientos y sumas.

    Cada muestra se convierte en dos: 3/4 de ella más 1/4 de su vecina izquierda, y 3/4 de ella
    más 1/4 de su vecina derecha, repitiendo la muestra del borde. Es la misma cuenta que hace
    `nn.Upsample(scale_factor=2, mode="linear", align_corners=False)`.
    """
    left = torch.cat([x[..., :1], x[..., :-1]], dim=-1)
    right = torch.cat([x[..., 1:], x[..., -1:]], dim=-1)
    return torch.stack([0.75 * x + 0.25 * left, 0.75 * x + 0.25 * right], dim=-1).flatten(-2)


class Upsample2x(nn.Upsample):
    """Duplica el largo por interpolación lineal; en GPU usa `upsample2x_linear`.

    En CUDA, el kernel de PyTorch para la interpolación lineal se llevaba el 60 % del tiempo
    de entrenamiento; la versión con sumas es 4 veces más rápida. En CPU es al revés, así que
    ahí se usa la de PyTorch. El resultado es el mismo y no tiene pesos: los runs anteriores
    se cargan igual.
    """

    def __init__(self) -> None:
        super().__init__(scale_factor=2, mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upsample2x_linear(x) if x.is_cuda else super().forward(x)


def _up_block(c_in: int, c_out: int, kernel_size: int) -> nn.Sequential:
    """Duplica el largo por interpolación lineal y luego aplica una convolución."""
    return nn.Sequential(
        Upsample2x(),
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
            Upsample2x(),
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
