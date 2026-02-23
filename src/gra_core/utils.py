# src/gra_core/utils.py

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np
Array = np.ndarray


def l2_normalize(v: Array) -> Array:
    """
    EN:
    L2-normalize a vector (no-op for zero vector).

    RU:
    L2-нормировка вектора (ничего не делает для нулевого вектора).
    """
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n


def random_states(
    dim: int,
    n: int,
    seed: int | None = None,
) -> List[Array]:
    """
    EN:
    Generate n random L2-normalized state vectors of dimension dim.

    RU:
    Сгенерировать n случайных L2-нормированных векторов-состояний размерности dim.
    """
    rng = np.random.default_rng(seed)
    states: List[Array] = []
    for _ in range(n):
        v = rng.standard_normal(dim)
        states.append(l2_normalize(v))
    return states


def homogeneous_projector(dim: int) -> Callable[[Array], Array]:
    """
    EN:
    Build a simple projector onto the 1D subspace spanned by [1, 1, ..., 1].

    RU:
    Построить простой проектор на одномерное подпространство,
    натянутое на вектор [1, 1, ..., 1].
    """
    u = np.ones(dim, dtype=np.float64)
    u = l2_normalize(u)

    def project(v: Array) -> Array:
        coeff = float(np.dot(v, u))
        return coeff * u

    return project


def make_level_dict(
    levels: Sequence[int],
    dim: int,
    n_per_level: int,
    seed: int | None = None,
) -> Dict[int, List[Array]]:
    """
    EN:
    Helper to create a dict[level] -> list of random states.

    RU:
    Вспомогательная функция для создания dict[level] -> список случайных состояний.
    """
    rng = np.random.default_rng(seed)
    psi: Dict[int, List[Array]] = {}
    for l in levels:
        # независимый seed per level для воспроизводимости
        level_seed = None if seed is None else int(seed + l)
        psi[l] = random_states(dim=dim, n=n_per_level, seed=level_seed)
    return psi
