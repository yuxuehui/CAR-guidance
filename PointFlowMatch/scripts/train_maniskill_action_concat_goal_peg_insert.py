import os

from pathlib import Path
from torch.utils.data import DataLoader
from composer.trainer import Trainer
from composer.callbacks import LRMonitor
from composer.core import State, Callback
from composer.models import ComposerModel
from composer.algorithms import EMA
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
import torch
import torch.optim as optim

from pfp import DEVICE, DATA_DIRS, set_seeds
from pfp.data.dataset_maniskill_action import RobotDatasetManiskillAction
from pfp.backbones.pointnet_concat_goal import PointNetBackboneConcatGoal
from pfp.policy.fm_policy_maniskill_action_rot_no_transform import FMPolicyActionRotNoTransform

class EpochLossLogger(Callback):

    def __init__(self, train_log_file: str, eval_log_file: str = None):
        self.train_log_file = train_log_file
        self.eval_log_file = eval_log_file
        self.epoch_losses = []

    def epoch_end(self, state: State, logger):
        epoch = int(state.timestamp.epoch)

        train_loss = state.loss.item() if state.loss is not None else 0.0

        eval_loss = None
        if hasattr(state, 'eval_metrics') and state.eval_metrics is not None:

            if isinstance(state.eval_metrics, dict):

                for metric_name, metric_value in state.eval_metrics.items():
                    if "loss" in metric_name.lower() and "total" in metric_name.lower():
                        eval_loss = metric_value.item() if hasattr(metric_value, 'item') else float(metric_value)
                        break

                if eval_loss is None:
                    for metric_name, metric_value in state.eval_metrics.items():
                        if "loss/eval" in metric_name.lower() and "total" in metric_name.lower():
                            eval_loss = metric_value.item() if hasattr(metric_value, 'item') else float(metric_value)
                            break

        self.epoch_losses.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
        })

        with open(self.train_log_file, 'a') as f:
            f.write(f"epoch={epoch}, loss={train_loss:.6f}\n")

        if self.eval_log_file is not None and eval_loss is not None:
            with open(self.eval_log_file, 'a') as f:
                f.write(f"epoch={epoch}, loss={eval_loss:.6f}\n")

        if eval_loss is not None:
            print(f"[Epoch {epoch}] train_loss={train_loss:.6f}, eval_loss={eval_loss:.6f}")
        else:
            print(f"[Epoch {epoch}] train_loss={train_loss:.6f}")

def main():

    data_path_train = "/home/cfc/桌面/PointFlowMatch/data/peg_demo_hole_pose_no_rot_transform_train"
    data_path_valid = "/home/cfc/桌面/PointFlowMatch/data/peg_demo_hole_pose_no_rot_transform_valid"

    seed = 1234
    epochs = 1500
    log_wandb = False
    run_name = "maniskill_train_peg_insertion_action_concat_goal_action_length_1"

    obs_features_dim = 256

    y_dim = 7

    robot_state_dim = 8

    goal_pos_dim = 3
    x_dim = obs_features_dim + robot_state_dim + goal_pos_dim
    n_obs_steps = 2
    n_pred_steps = 4
    use_ema = True
    save_each_n_epochs = 25

    dataset_config = {
        "n_obs_steps": n_obs_steps,
        "n_pred_steps": n_pred_steps,
        "subs_factor": 1,
        "use_pc_color": True,
        "n_points": 4096,
    }

    dataloader_config = {
        "batch_size": 64,
        "num_workers": 0,

    }

    optimizer_config = {
        "lr": 3.0e-5,
        "betas": [0.95, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 1.0e-6,
    }

    lr_scheduler_config = {
        "name": "cosine",
        "num_warmup_steps": 1000,
    }

    model_config = {
        "num_k_infer": 10,
        "time_conditioning": True,

        "norm_pcd_center": [0.0, 0.0, 0.0],
        "augment_data": False,
        "noise_type": "gaussian",
        "noise_scale": 1.0,
        "loss_type": "l2",
        "flow_schedule": "exp",
        "exp_scale": 4.0,
        "snr_sampler": "uniform",
        "loss_weights": {
            "xyz": 10.0,

            "rot": 10.0,
            "grip": 8.0,
        },
    }

    backbone_config = {
        "embed_dim": obs_features_dim,
        "input_channels": 6 if dataset_config["use_pc_color"] else 3,
        "input_transform": False,
        "use_group_norm": False,

    }

    diffusion_net_config = {
        "input_dim": y_dim,
        "global_cond_dim": x_dim * n_obs_steps,
        "diffusion_step_embed_dim": 256 if model_config["time_conditioning"] else 0,
        "down_dims": [256, 512, 1024],
        "kernel_size": 5,
        "n_groups": 8,
        "cond_predict_scale": True,

    }

    print("=" * 50)
    print("开始训练配置")
    print("=" * 50)
    set_seeds(seed)

    print("构建模型...")

    obs_encoder = PointNetBackboneConcatGoal(**backbone_config)

    diffusion_net = ConditionalUnet1D(**diffusion_net_config)

    composer_model: ComposerModel = FMPolicyActionRotNoTransform(
        x_dim=x_dim,
        y_dim=y_dim,
        n_obs_steps=n_obs_steps,
        n_pred_steps=n_pred_steps,
        obs_encoder=obs_encoder,
        diffusion_net=diffusion_net,
        **model_config,
    )

    composer_model.to(DEVICE)
    print(f"模型已移到设备: {DEVICE}")

    print(f"加载训练数据集: {data_path_train}")

    dataset_kwargs = {
        "data_path": data_path_train,
        "n_obs_steps": dataset_config["n_obs_steps"],
        "n_pred_steps": dataset_config["n_pred_steps"],
        "subs_factor": dataset_config.get("subs_factor", 1),
        "target_type": "action",
    }

    dataset_train = RobotDatasetManiskillAction(**dataset_kwargs)

    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=dataloader_config["batch_size"],
        num_workers=dataloader_config["num_workers"],
        persistent_workers=True if dataloader_config["num_workers"] > 0 else False,
        pin_memory=dataloader_config.get("pin_memory", False),
    )

    print(f"加载验证数据集: {data_path_valid}")
    dataset_kwargs["data_path"] = data_path_valid
    dataset_valid = RobotDatasetManiskillAction(**dataset_kwargs)

    dataloader_valid = DataLoader(
        dataset_valid,
        shuffle=False,
        batch_size=dataloader_config["batch_size"],
        num_workers=dataloader_config["num_workers"],
        persistent_workers=True if dataloader_config["num_workers"] > 0 else False,
        pin_memory=dataloader_config.get("pin_memory", False),
    )

    print(f"训练样本数: {len(dataset_train)}")
    print(f"验证样本数: {len(dataset_valid)}")

    optimizer = optim.AdamW(
        composer_model.parameters(),
        **optimizer_config
    )

    lr_scheduler = get_scheduler(
        lr_scheduler_config["name"],
        optimizer=optimizer,
        num_warmup_steps=lr_scheduler_config["num_warmup_steps"],
        num_training_steps=len(dataloader_train) * epochs,
    )

    log_dir = Path("ckpt") / run_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    train_log_file = log_dir / "epoch_loss.txt"
    eval_log_file = log_dir / "eval_loss.txt"
    epoch_loss_logger = EpochLossLogger(str(train_log_file), str(eval_log_file))

    with open(train_log_file, 'w') as f:
        f.write("# Training Loss Log\n")
        f.write(f"# run_name: {run_name}\n")
        f.write(f"# epochs: {epochs}\n")
        f.write("# format: epoch, loss\n")
        f.write("-" * 40 + "\n")

    with open(eval_log_file, 'w') as f:
        f.write("# Evaluation Loss Log\n")
        f.write(f"# run_name: {run_name}\n")
        f.write(f"# epochs: {epochs}\n")
        f.write("# format: epoch, loss\n")
        f.write("-" * 40 + "\n")

    trainer = Trainer(
        model=composer_model,
        train_dataloader=dataloader_train,
        eval_dataloader=dataloader_valid,
        max_duration=epochs,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        step_schedulers_every_batch=True,
        device="gpu" if DEVICE.type == "cuda" else "cpu",
        callbacks=[LRMonitor(), epoch_loss_logger],
        save_folder="ckpt/{run_name}",
        save_interval=f"{save_each_n_epochs}ep",
        save_num_checkpoints_to_keep=60,
        algorithms=[EMA()] if use_ema else None,
        run_name=run_name,
        autoresume=True if run_name is not None else False,
        spin_dataloaders=False,
    )

    import json
    config_to_save = {
        "seed": seed,
        "epochs": epochs,
        "obs_features_dim": obs_features_dim,
        "y_dim": y_dim,
        "x_dim": x_dim,
        "n_obs_steps": n_obs_steps,
        "n_pred_steps": n_pred_steps,
        "dataset_config": dataset_config,
        "dataloader_config": dataloader_config,
        "optimizer_config": optimizer_config,
        "lr_scheduler_config": lr_scheduler_config,
        "model_config": model_config,
        "backbone_config": backbone_config,
        "diffusion_net_config": diffusion_net_config,
        "data_path_train": data_path_train,

    }

    ckpt_dir = Path("ckpt") / trainer.state.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_path = ckpt_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)
    print(f"配置已保存到: {config_path}")

    print("=" * 50)
    print("开始训练...")
    print("=" * 50)

    trainer.fit()

    run_name_final = trainer.state.run_name
    print(f"训练完成！Run name: {run_name_final}")

    trainer.close()

    print("训练脚本执行完毕！")

if __name__ == "__main__":
    main()
