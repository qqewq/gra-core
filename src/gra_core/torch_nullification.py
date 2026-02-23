from __future__ import annotations

from typing import Dict, Sequence, Callable

import torch
from torch import Tensor


def phi_level_torch(
    psi_level: Sequence[Tensor],
    projector: Callable[[Tensor], Tensor],
) -> Tensor:
    """
    EN:
    PyTorch version of the level-l foam functional:
        Phi^{(l)} = sum_{i != j} | <psi_i | P | psi_j> |^2

    RU:
    PyTorch-версия функционала пены уровня l:
        Phi^{(l)} = sum_{i != j} | <psi_i | P | psi_j> |^2
    """
    # psi_level: list/tuple of (d,) vectors -> stack to (N, d)
    psi = torch.stack(psi_level, dim=0)  # (N, d)

    # Project each vector
    v_proj = torch.stack([projector(v) for v in psi], dim=0)  # (N, d)

    # Pairwise inner products: (N, d) x (d, N) -> (N, N)
    inner = v_proj.conj() @ v_proj.T

    # Zero out diagonal (we only want i != j)
    inner_no_diag = inner - torch.diag(torch.diag(inner))

    # Sum of squared magnitudes
    phi = torch.sum(torch.abs(inner_no_diag) ** 2)
    return phi


def homogeneous_projector_torch(dim: int, device=None, dtype=None):
    """
    EN:
    Simple projector onto the 1D subspace spanned by [1, 1, ..., 1].

    RU:
    Простой проектор на одномерное подпространство,
    натянутое на [1, 1, ..., 1].
    """
    if dtype is None:
        dtype = torch.float32
    u = torch.ones(dim, device=device, dtype=dtype)
    u = u / (torch.norm(u) + 1e-12)

    def project(v: Tensor) -> Tensor:
        coeff = torch.dot(v, u)
        return coeff * u

    return project


def multilevel_phi_torch(
    psi: Dict[int, Sequence[Tensor]],
    projectors: Dict[int, Callable[[Tensor], Tensor]],
    levels: Sequence[int],
    lambdas: Dict[int, float] | None = None,
) -> Tensor:
    """
    EN:
    Multilevel foam functional (PyTorch):
        J = sum_l Lambda_l * Phi^{(l)}.

    RU:
    Многоуровневый функционал пены (PyTorch):
        J = sum_l Lambda_l * Phi^{(l)}.
    """
    # pick device from any tensor
    any_level = next(iter(psi.values()))
    any_vec = any_level[0]
    device = any_vec.device

    if lambdas is None:
        lambdas = {l: 1.0 for l in levels}

    total = torch.tensor(0.0, device=device)
    for l in levels:
        if l not in psi or l not in projectors:
            continue
        phi_l = phi_level_torch(psi[l], projectors[l])
        total = total + lambdas.get(l, 1.0) * phi_l
    return total
