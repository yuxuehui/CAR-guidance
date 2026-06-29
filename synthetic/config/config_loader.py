"""Configuration loading for the synthetic experiment.

Parses ``fm_config.yaml`` into the typed :class:`TrainConfig` dataclass. PyYAML
is used when available; otherwise a small dependency-free fallback parser
handles the flat ``key: value`` config used here.
"""

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _strip_yaml_comment(s: str) -> str:
    """Strip # comments, respecting simple quotes."""
    in_single = False
    in_double = False
    for i, ch in enumerate(s):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return s[:i].rstrip()
    return s.strip()


def _coerce_scalar(v: str):
    """Best-effort scalar parsing for simple YAML key: value configs."""
    v = v.strip()
    if not v:
        return ""
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        # int first (avoid turning "1" into 1.0)
        if low.startswith("0") and len(low) > 1 and low[1].isdigit():
            # keep as string for things like "01"
            return v
        return int(v)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return v


def _simple_yaml_load(text: str) -> dict:
    """
    Minimal YAML loader for flat key: value configs (no nesting/lists).
    This is a dependency-free fallback when PyYAML isn't available.
    """
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = _strip_yaml_comment(val)
        out[key] = _coerce_scalar(val)
    return out

@dataclass
class TrainConfig:
    # Required fields (no defaults)
    lr: float
    batch_size: int
    iterations: int
    hidden_dim: int
    guidance_scale: float
    classifier_epochs: int
    classifier_lr: float
    device: str
    vf_model: str
    estimate_x1: bool
    step_size: float
    batch_size_plot_traj: int
    start_guidance_threshold: float

    # Guidance configuration (g_cov_g, guidance_matching, car_guidance)
    guidance_fn: str = "g_cov_g"
    guidance_train_steps: int = 10000
    guidance_batch_size: int = 1024
    guidance_lr: float = 0.0001

    # Conflict-gate / online-training configuration
    energy_temperature: float = 1.0
    conflict_threshold: float = 0.495
    conflict_temperature: float = 0.15
    blend_function: str = "smootherstep"  # "smootherstep" or "sigmoid"
    x1_conflict_threshold: Optional[float] = None  # early-stop when x1_conflict_ratio < threshold
    active_ratio_threshold: Optional[float] = None  # early-stop when trajectory conflict ratio < threshold
    loss_early_stop_threshold: Optional[float] = None  # early-stop when loss < threshold

    guidance_ckpt_dir: str = "checkpoints/guidance"
    result_output_dir: str = "result_new"  # base output directory (relative to script dir)

    # Visualization configuration
    use_reward_landscape: bool = True  # plot reward (higher=better) instead of energy
    integrate_gradient_to_energy: bool = False  # integrate gradient fields to scalar energy for plotting

def load_config(path: str = "fm_config.yaml") -> TrainConfig:
    """
    Load YAML config into TrainConfig dataclass.
    
    Args:
        path: Path to config file. If relative, it will be resolved relative to
              the directory containing this config_loader.py file.
    
    Returns:
        TrainConfig: Loaded configuration object
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        Exception: If config file is invalid YAML
    """
    # Convert to Path object for easier manipulation
    config_path = Path(path)
    
    # If path is not absolute, resolve it relative to this file's directory
    if not config_path.is_absolute():
        # Get the directory where this config_loader.py file is located
        config_dir = Path(__file__).parent.absolute()
        config_path = config_dir / path
    
    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Resolved from: {path}\n"
            f"Config directory: {config_path.parent}"
        )
    
    # Load and parse YAML (PyYAML if available; otherwise a minimal fallback)
    with open(config_path, "r", encoding="utf-8") as f:
        if yaml is not None:
            cfg_dict = yaml.safe_load(f)
        else:
            cfg_dict = _simple_yaml_load(f.read())
    
    if cfg_dict is None:
        raise ValueError(f"Config file is empty or invalid: {config_path}")
    
    return TrainConfig(**cfg_dict)
