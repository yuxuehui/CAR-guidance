from typing import Dict, Any, List
import torch
import numpy as np
from pathlib import Path

from ..core.base_experiment import BaseExperiment
from ..policy.policy_loader import load_policy, create_guidance_from_config
from ..guidance import NoneGuidance

class BaselineExperiment(BaseExperiment):

    def setup(self):
        self.logger.info("设置基线实验（无 guidance）...")

        model_config = self.config['model']
        ckpt_path = model_config['ckpt_path']

        self.logger.info(f"加载模型: {ckpt_path}")

        guidance = NoneGuidance()
        self.policy, self.model_config_dict = load_policy(ckpt_path, guidance=guidance, config=self.config)

        if self.model_config_dict:
            self.norm_pcd_center = self.model_config_dict.get('model_config', {}).get('norm_pcd_center', None)
            if self.norm_pcd_center:
                self.logger.info(f"模型 norm_pcd_center: {self.norm_pcd_center}")

        self.logger.info("基线实验设置完成")

    def load_data(self):
        from ..utils.data_utils import load_test_data

        data_config = self.config['data']
        test_data_path = data_config['test_data_path']

        self.logger.info(f"加载测试数据: {test_data_path}")

        test_data = load_test_data(test_data_path, data_config)

        self.logger.info(f"加载了 {len(test_data)} 个测试样本")

        return test_data
