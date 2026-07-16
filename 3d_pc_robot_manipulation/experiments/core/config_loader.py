import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf

def load_config(config_path: str, base_config_path: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    if base_config_path:

        base_config = OmegaConf.load(base_config_path)
        exp_config = OmegaConf.load(config_path)

        config = OmegaConf.merge(base_config, exp_config)
    else:
        config = OmegaConf.load(config_path)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(config)

    config_dict = OmegaConf.to_container(config, resolve=True)

    validate_config(config_dict)

    return config_dict

def validate_config(config: Dict[str, Any]) -> None:
    required_keys = ['model', 'data', 'evaluation']

    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置缺少必需的键: {key}")

    model_config = config['model']
    if 'ckpt_path' not in model_config:
        raise ValueError("模型配置缺少 'ckpt_path'")

    data_config = config['data']

    if 'test_data_path' not in data_config and 'demos_path' not in data_config:
        raise ValueError("数据配置缺少 'test_data_path' 或 'demos_path'")

    eval_config = config['evaluation']
    if 'metrics' not in eval_config:
        raise ValueError("评估配置缺少 'metrics'")

def merge_configs(base_config: Dict[str, Any], exp_config: Dict[str, Any]) -> Dict[str, Any]:
    merged = base_config.copy()

    for key, value in exp_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
