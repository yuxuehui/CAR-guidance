import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import time
import json
from pathlib import Path

from datasets.traj_dataset import get_traj_dataloader
from models.flow_guide import TrajFlowModel, create_traj_flow_model

class FlowModelTrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = create_traj_flow_model(config['model'])
        self.model.to(self.device)

        self.train_dataloader = get_traj_dataloader(
            hdf5_path=config['data']['train_path'],
            horizon=config['model']['horizon'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            shuffle=True
        )

        if config['data'].get('val_path'):
            self.val_dataloader = get_traj_dataloader(
                hdf5_path=config['data']['val_path'],
                horizon=config['model']['horizon'],
                batch_size=config['training']['batch_size'],
                num_workers=config['training']['num_workers'],
                shuffle=False
            )
        else:
            self.val_dataloader = None

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['min_lr']
        )

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        print(f"✅ 训练器初始化完成")
        print(f"   - 设备: {self.device}")
        print(f"   - 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   - 训练数据量: {len(self.train_dataloader)} batches")
        if self.val_dataloader:
            print(f"   - 验证数据量: {len(self.val_dataloader)} batches")

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'global_step': self.global_step,
            'config': self.config
        }

        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')

        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pth')
            print(f"💾 保存最佳模型，验证损失: {loss:.6f}")

        if epoch % self.config['training']['save_interval'] == 0:
            torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pth')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        print(f"📂 从第 {self.current_epoch} 轮继续训练")

    def train_epoch(self):
        self.model.train()
        epoch_losses = []

        epoch_loss_dict = {
            'total_loss': [], 'dynamics_loss': []
        }

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):

            observations = batch['observations'].to(self.device)
            conditions = {k: v.to(self.device) for k, v in batch['conditions'].items()}
            wall_locations = batch['wall_locations'].to(self.device)
            mask = batch['mask'].to(self.device)

            loss, loss_dict = self.model.compute_loss(
                x_target=observations,
                conditions=conditions,
                wall_locations=wall_locations,
                mask=mask
            )

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip_norm']
            )

            self.optimizer.step()

            epoch_losses.append(loss.item())
            for key, value in loss_dict.items():
                epoch_loss_dict[key].append(value)

            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            if self.global_step % self.config['training']['log_interval'] == 0:
                print(f"Step {self.global_step}: Loss={loss.item():.6f}, LR={self.optimizer.param_groups[0]['lr']:.2e}")

            self.global_step += 1

        avg_losses = {key: np.mean(values) for key, values in epoch_loss_dict.items()}
        return avg_losses

    def validate(self):
        if self.val_dataloader is None:
            return None

        self.model.eval()
        val_losses = []
        val_loss_dict = {
            'total_loss': [], 'dynamics_loss': [], 'position_loss': [], 'velocity_loss': [],
            'boundary_loss': [], 'smoothness_loss': [], 'collision_loss': []
        }

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):

                observations = batch['observations'].to(self.device)
                conditions = {k: v.to(self.device) for k, v in batch['conditions'].items()}
                wall_locations = batch['wall_locations'].to(self.device)
                mask = batch['mask'].to(self.device)

                loss, loss_dict = self.model.compute_loss(
                    x_target=observations,
                    conditions=conditions,
                    wall_locations=wall_locations,
                    mask=mask
                )

                val_losses.append(loss.item())
                for key, value in loss_dict.items():
                    val_loss_dict[key].append(value)

        avg_val_losses = {key: np.mean(values) for key, values in val_loss_dict.items()}

        return avg_val_losses

    def train(self):
        print("🚀 开始训练...")

        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            start_time = time.time()

            train_losses = self.train_epoch()

            val_losses = self.validate()

            self.scheduler.step()

            epoch_time = time.time() - start_time

            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            print(f"训练损失: {train_losses['total_loss']:.6f}")
            if val_losses:
                print(f"验证损失: {val_losses['total_loss']:.6f}")
            print(f"耗时: {epoch_time:.2f}s")
            print("-" * 50)

            current_loss = val_losses['total_loss'] if val_losses else train_losses['total_loss']
            is_best = current_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_loss

            self.save_checkpoint(epoch + 1, current_loss, is_best)

            if self.config['training'].get('early_stopping') and val_losses:

                pass

        print("✅ 训练完成！")

def main():
    parser = argparse.ArgumentParser(description='训练Traj_Flow_Model')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    trainer = FlowModelTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()

if __name__ == "__main__":
    main()
