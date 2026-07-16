from typing import Dict, Any, List
import torch
import numpy as np
from pathlib import Path

from ..core.base_experiment import BaseExperiment
from ..policy.policy_loader import load_policy, create_guidance_from_config
from ..guidance import StaticGuidance

class StaticGuidanceExperiment(BaseExperiment):

    def setup(self):
        self.logger.info("设置静态能量场实验...")

        model_config = self.config['model']
        ckpt_path = model_config['ckpt_path']
        config_path = model_config.get('config_path', None)

        self.logger.info(f"加载模型: {ckpt_path}")

        guidance_config = self.config.get('guidance', {})

        if 'norm_pcd_center' in model_config:
            guidance_config['norm_pcd_center'] = model_config['norm_pcd_center']

        self.guidance = StaticGuidance(guidance_config)

        self.logger.info(f"创建静态能量场 guidance:")
        self.logger.info(f"  能量中心数量: {len(guidance_config.get('energy_centers', []))}")
        self.logger.info(f"  高斯标准差 (sigma): {guidance_config.get('sigma', 0.1)}")
        self.logger.info(f"  能量强度: {guidance_config.get('energy_scales', [])}")

        self.policy, self.model_config_dict = load_policy(ckpt_path, guidance=self.guidance, config=self.config)

        if self.model_config_dict:
            norm_pcd_center = self.model_config_dict.get('model_config', {}).get('norm_pcd_center', None)
            if norm_pcd_center:

                guidance_config['norm_pcd_center'] = norm_pcd_center
                self.guidance.norm_pcd_center = norm_pcd_center
                self.logger.info(f"模型 norm_pcd_center: {norm_pcd_center}")

        self.logger.info("静态能量场实验设置完成")

    def load_data(self):
        from ..utils.data_utils import load_test_data

        data_config = self.config['data']
        test_data_path = data_config['test_data_path']

        self.logger.info(f"加载测试数据: {test_data_path}")

        test_data = load_test_data(test_data_path, data_config)

        self.logger.info(f"加载了 {len(test_data)} 个测试样本")

        return test_data

    def generate_energy_centers(self, base_trajectory: np.ndarray) -> List[List[float]]:

        guidance_config = self.config.get('guidance', {})
        return guidance_config.get('energy_centers', [])
