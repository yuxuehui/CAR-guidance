"""
一个简单的2D离散流匹配模型
使用 t^2 调度器的简单2D离散FM模型的训练

数据集: 2D离散棋盘
模型 (概率去噪器): MLP
"""

import time
import torch
from torch import nn, Tensor

# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL

# visualization
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# 设置设备
if torch.cuda.is_available():
    device = 'cuda:0'
    print('使用GPU')
else:
    device = 'cpu'
    print('使用CPU')

# 设置随机种子
torch.manual_seed(42)

def inf_train_gen(n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu") -> Tensor:
    """生成训练数据的函数"""
    assert n_grid_points % 4 == 0, "网格点数必须能被4整除"
    
    n_grid_points = n_grid_points // 4
    
    x1 = torch.randint(low=0, high=n_grid_points * 4, size=(batch_size,), device=device)
    samples_x2 = torch.randint(low=0, high=n_grid_points, size=(batch_size,), device=device)
    
    x2 = (
        samples_x2
        + 2 * n_grid_points
        - torch.randint(low=0, high=2, size=(batch_size,), device=device) * 2 * n_grid_points
        + (torch.floor(x1 / n_grid_points) % 2) * n_grid_points
    )
    
    x_end = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1)

    return x_end.long()

# 激活函数类
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x

# 模型类
class MLP(nn.Module):
    def __init__(
        self, input_dim: int = 128, time_dim: int = 1, hidden_dim=128, length=2):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.time_embedding = nn.Linear(1, time_dim)
        self.token_embedding = torch.nn.Embedding(self.input_dim, hidden_dim)

        self.main = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim * length + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, self.input_dim * length),
        )

    def forward(self, x, t):
        t = self.time_embedding(t.unsqueeze(-1))
        x = self.token_embedding(x)

        B, N, d = x.shape
        x = x.reshape(B, N * d)
        
        h = torch.cat([x, t], dim=1)
        h = self.main(h)

        h = h.reshape(B, N, self.input_dim)

        return h

def main():
    """主训练函数"""
    source_distribution = "uniform"

    # 训练参数
    lr = 0.001
    batch_size = 4096
    iterations = 30001
    print_every = 3000

    vocab_size = 128
    hidden_dim = 128

    epsilon = 1e-3

    if source_distribution == "uniform":
        added_token = 0
    elif source_distribution == "mask":
        mask_token = vocab_size  # tokens starting from zero
        added_token = 1
    else:
        raise NotImplementedError
        
    # 额外的mask token
    vocab_size += added_token

    # 初始化概率去噪器模型
    probability_denoiser = MLP(input_dim=vocab_size, time_dim=1, hidden_dim=hidden_dim).to(device)

    # 实例化凸路径对象
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)

    # 初始化优化器
    optim = torch.optim.Adam(probability_denoiser.parameters(), lr=lr) 

    loss_fn = MixturePathGeneralizedKL(path=path)

    # 训练
    start_time = time.time()

    steps = 0
    losses = []
    for i in range(iterations):
        optim.zero_grad() 

        # 采样数据
        x_1 = inf_train_gen(n_grid_points=vocab_size - added_token, batch_size=batch_size, device=device)
        
        if source_distribution == "uniform":
            x_0 = torch.randint_like(x_1, high=vocab_size)
        elif source_distribution == "mask":
            x_0 = torch.zeros_like(x_1) + mask_token
        else:
            raise NotImplementedError

        # 采样时间
        t = torch.rand(x_1.shape[0]).to(device) * (1 - epsilon)

        # 采样概率路径
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

        print("path_sample.x_t形状是", path_sample.x_t.shape)
        print("path_sample.t形状是", path_sample.t.shape)
        print("x_1形状是", x_1.shape)
        print("x_0形状是", x_0.shape)
        print("t形状是", t.shape)
        logits = probability_denoiser(x=path_sample.x_t, t=path_sample.t)

        print("logits形状是", logits.shape)
        break

        # 离散流匹配广义KL损失
        loss = loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)

        # 优化器步骤
        loss.backward()
        optim.step()
        
        # 记录损失
        if (i+1) % print_every == 0:
            elapsed = time.time() - start_time
            print('| 迭代 {:6d} | {:5.2f} ms/step | 损失 {:8.3f} ' 
                  .format(i+1, elapsed*1000/print_every, loss.item())) 
            start_time = time.time()

    return 0
    # 保存模型
    torch.save(probability_denoiser.state_dict(), 'probability_denoiser.pth')

if __name__ == "__main__":
    main()
