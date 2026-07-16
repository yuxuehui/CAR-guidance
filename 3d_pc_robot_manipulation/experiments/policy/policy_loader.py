import torch
from pathlib import Path
from typing import Optional, Dict, Any

from .guided_policy import GuidedFMPolicy
from ..guidance import StaticGuidance, NoneGuidance

def load_policy(ckpt_path: str,
                guidance=None,
                config: Optional[Dict[str, Any]] = None) -> tuple:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).parent.parent.parent
    utils_path = repo_root / "utils"
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))

    from utils.utils_tool import load_model_from_checkpoint_concat_goal

    ckpt_path = Path(ckpt_path)

    if ckpt_path.is_absolute() or '/' in str(ckpt_path):

        ckpt_dir = ckpt_path.parent
        ckpt_name = ckpt_dir.name
        ckpt_episode = ckpt_path.stem
    else:

        ckpt_name = str(ckpt_path.parent) if ckpt_path.parent != Path('.') else ckpt_path.stem
        ckpt_episode = ckpt_path.stem

    if config:
        model_config = config.get('model', {})
        num_k_infer = model_config.get('num_k_infer', 10)
        flow_schedule = model_config.get('flow_schedule', 'linear')
        exp_scale = model_config.get('exp_scale', None)
    else:
        num_k_infer = 10
        flow_schedule = 'linear'
        exp_scale = None

    base_policy, model_config_dict = load_model_from_checkpoint_concat_goal(
        ckpt_name=ckpt_name,
        ckpt_episode=ckpt_episode,
        num_k_infer=num_k_infer,
        flow_schedule=flow_schedule,
        exp_scale=exp_scale,
        use_ema=True
    )

    if guidance is None:

        guidance = NoneGuidance()

    guided_policy = GuidedFMPolicy(base_policy, guidance)

    return guided_policy, model_config_dict

def create_guidance_from_config(guidance_config: Dict[str, Any]) -> Optional[Any]:
    guidance_type = guidance_config.get('type', 'none')

    if guidance_type == 'none':
        return NoneGuidance()
    elif guidance_type == 'static':
        return StaticGuidance(guidance_config)
    else:
        raise ValueError(f"未知的 guidance 类型: {guidance_type}")
