# src/gra_core/jax_nullification.py

from __future__ import annotations

from typing import Callable, Dict, Sequence

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def phi_level_jax(
    psi_level: Sequence[Array],
    projector: Callable[[Array], Array],
) -> Array:
    """
    EN:
    JAX version of the level-l foam functional:
        Phi^{(l)} = sum_{i != j} | <psi_i | P | psi_j> |^2

    RU:
    JAX-версия функционала пены уровня l:
        Phi^{(l)} = sum_{i != j} | <psi_i | P | psi_j> |^2
    """
    # psi_level: list/tuple of (d,) vectors -> stack to (N, d)
    psi = jnp.stack(psi_level, axis=0)  # shape (N, d)

    def project(v: Array) -> Array:
        return projector(v)

    # (N, d) -> (N, d) projected
    v_proj = jax.vmap(project)(psi)

    # Pairwise inner products: (N, d) x (d, N) -> (N, N)
    inner = jnp.matmul(jnp.conjugate(v_proj), jnp.transpose(v_proj))

    # Zero out diagonal (we only want i != j)
    inner_no_diag = inner - jnp.diag(jnp.diag(inner))

    # Sum of squared magnitudes
    phi = jnp.sum(jnp.abs(inner_no_diag) ** 2)
    return phi


def homogeneous_projector_jax(dim: int) -> Callable[[Array], Array]:
    """
    EN:
    Simple projector onto the 1D subspace spanned by [1, 1, ..., 1].

    RU:
    Простой проектор на одномерное подпространство,
    натянутое на [1, 1, ..., 1].
    """
    u = jnp.ones((dim,), dtype=jnp.float32)
    u = u / (jnp.linalg.norm(u) + 1e-12)

    def project(v: Array) -> Array:
        coeff = jnp.dot(v, u)
        return coeff * u

    return project


def multilevel_phi_jax(
    psi: Dict[int, Sequence[Array]],
    projectors: Dict[int, Callable[[Array], Array]],
    levels: Sequence[int],
    lambdas: Dict[int, float] | None = None,
) -> Array:
    """
    EN:
    Multilevel foam functional:
        J = sum_l Lambda_l * Phi^{(l)}

    RU:
    Многоуровневый функционал пены:
        J = sum_l Lambda_l * Phi^{(l)}
    """
    if lambdas is None:
        lambdas = {l: 1.0 for l in levels}

    total = 0.0
    for l in levels:
        if l not in psi or l not in projectors:
            continue
        phi_l = phi_level_jax(psi[l], projectors[l])
        total = total + lambdas.get(l, 1.0) * phi_l
    return total


@jax.jit
def gradient_step_states(
    psi: Dict[int, Array],
    projectors: Dict[int, Callable[[Array], Array]],
    levels: Sequence[int],
    lambdas: Dict[int, float],
    lr: float,
) -> Dict[int, Array]:
    """
    EN:
    One JAX gradient step on states psi to minimize multilevel foam J.

    psi:
      dict[level] -> (N_l, d) JAX arrays

    RU:
    Один шаг градиентного спуска по состояниям psi для минимизации
    многоуровневой пены J.

    psi:
      dict[level] -> (N_l, d) JAX-массивы
    """

    def loss(flat_psi: Dict[int, Array]) -> Array:
        local_psi = {l: flat_psi[l] for l in levels}
        return multilevel_phi_jax(local_psi, projectors, levels, lambdas)

    grads = jax.grad(loss)(psi)
    new_psi: Dict[int, Array] = {}
    for l in levels:
        new_psi[l] = psi[l] - lr * grads[l]
    return new_psi
