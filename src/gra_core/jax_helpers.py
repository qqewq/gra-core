# src/gra_core/jax_helpers.py

from __future__ import annotations
from typing import Dict, Sequence, Callable

import jax.numpy as jnp
from jax import Array

from .jax_nullification import multilevel_phi_jax


def build_psi_from_params(params: Dict) -> Dict[int, Sequence[Array]]:
    """
    EN:
    Very simple example: treat each parameter vector as a 'state'
    at level 0. You can replace this with your own logic (e.g.
    activations, specific layers, etc).

    RU:
    Очень простой пример: рассматриваем каждый параметр-вектор как
    состояние уровня 0. В реальном коде можно подменить на активации
    или отдельные слои.
    """
    states = []
    for leaf in params.values():
        arr = jnp.ravel(leaf)
        states.append(arr / (jnp.linalg.norm(arr) + 1e-12))
    return {0: states}


def foam_regularizer_from_params(
    params: Dict,
    projectors: Dict[int, Callable[[Array], Array]],
    levels,
    lambdas,
) -> Array:
    """
    EN: Convenience wrapper to compute J for given params.
    RU: Удобный враппер для вычисления J по параметрам.
    """
    psi = build_psi_from_params(params)
    return multilevel_phi_jax(psi, projectors, levels, lambdas)
