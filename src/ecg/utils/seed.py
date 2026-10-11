"""Control centralizado de semillas y aleatoriedad.

Regla 9 de AGENTS.md: Toda aleatoriedad usa src/ecg/utils/seed.py.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fija la semilla en todos los generadores pseudo-aleatorios (random, numpy, torch)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
