"""MLP reward classifier and its training loop, used to build guidance fields."""

import os
import torch
from torch import nn
from torch.optim import Optimizer


class Classifier(nn.Module):
    """Basic MLP classifier used for guidance fields."""

    def __init__(self, dim=2, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(model, data, labels, model_name, epochs=4000, lr=1e-3, save_dir=None, device=None):
    """Train a classifier and optionally save its weights."""
    if device is not None:
        model = model.to(device)
        data = data.to(device)
        labels = labels.to(device)

    optimizer: Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(epochs // 8, 1) == 0:
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean()
            print(f"{model_name} Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}")

    if save_dir:
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name.replace(' ', '_').lower()}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[SAVED CLASSIFIER] {save_path}")

