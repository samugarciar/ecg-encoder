"""Pruebas de la arquitectura del autoencoder y del preprocesamiento (D-008, D-020, D-022)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg.models.autoencoder import ConvAutoencoder, count_parameters, upsample2x_linear
from ecg.models.preprocessing import normal_mask, prepare_inputs, zscore_beats

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere GPU CUDA")


def test_default_architecture_shapes_and_size() -> None:
    """La arquitectura base reconstruye la forma de entrada, con cuello de 32 y 284 961 pesos."""
    model = ConvAutoencoder()
    x = torch.randn(3, 1, 256)
    assert model.encode(x).shape == (3, 32)
    assert model(x).shape == (3, 1, 256)
    assert count_parameters(model) == 284_961


def test_encoder_halves_length_per_level() -> None:
    model = ConvAutoencoder()
    h = torch.randn(2, 1, 256)
    lengths = []
    for block in model.encoder:
        h = block(h)
        lengths.append((h.shape[1], h.shape[2]))
    assert lengths == [(16, 128), (32, 64), (64, 32), (128, 16)]


@pytest.mark.parametrize("shape", [(4, 128, 16), (2, 16, 128), (3, 1, 5), (1, 1, 1)])
def test_upsample2x_linear_matches_pytorch(shape: tuple[int, ...]) -> None:
    """La interpolación con sumas da lo mismo que la de PyTorch, incluidos los bordes."""
    x = torch.randn(shape, dtype=torch.float64)
    expected = torch.nn.functional.interpolate(x, scale_factor=2, mode="linear",
                                               align_corners=False)
    assert torch.allclose(upsample2x_linear(x), expected, atol=1e-12)


@needs_cuda
def test_gpu_model_matches_cpu_model() -> None:
    """El mismo modelo da la misma reconstrucción en CPU y en GPU, con los mismos pesos."""
    torch.manual_seed(0)
    model = ConvAutoencoder().eval()
    x = torch.randn(8, 1, 256)
    with torch.no_grad():
        cpu_out = model(x)
        gpu_out = model.to("cuda")(x.to("cuda")).cpu()
    assert torch.allclose(cpu_out, gpu_out, atol=1e-4)


@pytest.mark.parametrize("kwargs", [{"input_length": 250}, {"kernel_size": 6}, {"channels": ()}])
def test_invalid_architecture_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ConvAutoencoder(**kwargs)


def test_zscore_beats_per_row() -> None:
    x = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 10.0, 10.0]], dtype=np.float32)
    z = zscore_beats(x)
    assert z.dtype == np.float32
    assert np.allclose(z[0].mean(), 0.0, atol=1e-6)
    assert np.allclose(z[0].std(), 1.0, atol=1e-6)
    assert np.all(np.isfinite(z[1])) and np.allclose(z[1], 0.0)  # latido plano: sin dividir por 0


def test_prepare_inputs_rejects_unknown_normalization() -> None:
    with pytest.raises(ValueError):
        prepare_inputs(np.zeros((2, 4)), "minmax")


def test_normal_mask_excludes_noisy_when_asked() -> None:
    beats = {
        "split": np.array(["train", "train", "train", "val"]),
        "label": np.array([0, 0, 1, 0], dtype=np.int8),
        "mlii_noisy": np.array([False, True, False, False]),
    }
    assert normal_mask(beats, "train", exclude_noisy=True).tolist() == [True, False, False, False]
    assert normal_mask(beats, "train", exclude_noisy=False).tolist() == [True, True, False, False]
