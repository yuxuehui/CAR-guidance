import torch
import random
import pathlib
import numpy as np
from dataclasses import dataclass

@dataclass
class DATA_DIRS:
    ROOT = pathlib.Path(__file__).parents[1] / "demos"
    PFP = ROOT / "sim"
    PFP_REAL = ROOT / "real"

@dataclass
class REPO_DIRS:
    ROOT = pathlib.Path(__file__).parents[1]
    CKPT = ROOT / "ckpt"
    URDFS = ROOT / "urdfs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
