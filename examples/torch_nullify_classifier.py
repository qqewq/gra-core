# examples/torch_nullify_classifier.py

"""
EN:
Toy PyTorch classifier with GRA foam regularizer on hidden states.

RU:
Игрушечный PyTorch-классификатор с GRA-регуляризатором пены
по скрытым состояниям.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from gra_core.torch_nullification import (
    multilevel_phi_torch,
    homogeneous_projector_torch,
)


class SmallMLP(nn.Module):
    def __init__(self, dim_in=20, dim_hidden=16, dim_out=2):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x, return_hidden: bool = False):
        h = self.relu(self.fc1(x))
        logits = self.fc2(h)
        if return_hidden:
            return logits, h
        return logits


def make_toy_data(n_samples=512, dim=20):
    """
    EN: Simple synthetic binary classification dataset.
    RU: Простой синтетический датасет для бинарной классификации.
    """
    x = torch.randn(n_samples, dim)
    w = torch.randn(dim)
    y = (x @ w > 0).long()
    return x, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_toy_data()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SmallMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    projector = homogeneous_projector_torch(dim=16, device=device)
    lambda_foam = 1e-3
    levels = [0]  # single level for hidden layer

    for epoch in range(10):
        total_loss = 0.0
        total_task = 0.0
        total_foam = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits, h = model(xb, return_hidden=True)
            task_loss = ce(logits, yb)

            # EN: treat each hidden vector as a state at level 0
            # RU: каждое скрытое состояние считаем состоянием уровня 0
            psi_dict = {0: [v for v in h]}  # list of (dim_hidden,) tensors
            projectors = {0: projector}

            foam_loss = multilevel_phi_torch(psi_dict, projectors, levels)

            loss = task_loss + lambda_foam * foam_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_task += task_loss.item() * xb.size(0)
            total_foam += foam_loss.item() * xb.size(0)

        n = len(dataset)
        print(
            f"Epoch {epoch:02d} | "
            f"loss={total_loss/n:.4f} | "
            f"task={total_task/n:.4f} | "
            f"foam={total_foam/n:.4f}"
        )


if __name__ == "__main__":
    main()
