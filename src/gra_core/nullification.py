# src/gra_core/nullification.py

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np
Array = np.ndarray


class GraNullifier:
    """
    EN:
    GraNullifier: multiverse nullification of "foam" between states.

    Idea:
    - For each level l we have a set of states psi[l] = [|Psi^{(a)}>]
    - There is a projector P_l: v -> v_proj onto the solution subspace of goal G_l
    - Level-l "foam":
        Phi^{(l)} = sum_{a != b} | <Psi^{(a)} | P_l | Psi^{(b)}> |^2
    - The nullifier iteratively moves states towards a subspace where
      off-diagonal cross terms vanish (Phi^{(l)} → 0).

    This is a toy but honest implementation using
    "move towards projector" as a pseudo-gradient.

    RU:
    GraNullifier: мультиверсное обнуление «пены» между состояниями.

    Идея:
    - Для каждого уровня l есть набор состояний psi[l] = [|Psi^{(a)}>]
    - Есть проектор P_l: v -> v_proj на подпространство решений цели G_l
    - «Пена» уровня l:
        Phi^{(l)} = sum_{a != b} | <Psi^{(a)} | P_l | Psi^{(b)}> |^2
    - Nullifier итеративно двигает состояния к подпространству, где
      недиагональные перекрестные члены затухают (Phi^{(l)} → 0).

    Это игрушечная, но честная реализация через
    «движение к проектору» как псевдо-градиент.
    """

    def __init__(
        self,
        levels: int = 1,
        alpha: float = 0.9,
        lambda0: float = 1.0,
        lr: float = 1e-2,
        eps: float = 1e-6,
        max_iter: int = 100,
    ) -> None:
        """
        EN:
        Parameters:
        - levels: maximum level index l (0..levels)
        - alpha: geometric decay coefficient across levels
        - lambda0: base weight (reserved for full J_multiverse functional)
        - lr: step size of the pseudo-gradient update
        - eps: threshold below which Phi^{(l)} is considered "nullified"
        - max_iter: max iterations per level

        RU:
        Параметры:
        - levels: максимальный индекс уровня l (0..levels)
        - alpha: коэффициент геометрического затухания по уровням
        - lambda0: базовый вес (зарезервирован для полного функционала J_multiverse)
        - lr: шаг псевдо-градиентного обновления
        - eps: порог, ниже которого Phi^{(l)} считаем «обнулённой»
        - max_iter: максимум итераций на уровень
        """
        self.levels = levels
        self.alpha = alpha
        self.lambda0 = lambda0
        self.lr = lr
        self.eps = eps
        self.max_iter = max_iter

    @staticmethod
    def _normalize(v: Array) -> Array:
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

    def _phi_level(
        self,
        psi_level: Sequence[Array],
        projector: Callable[[Array], Array],
    ) -> float:
        """
        EN:
        Compute Phi^{(l)} for a given level:

            Phi^{(l)} = sum_{i != j} | <psi_i | P | psi_j> |^2

        where P is the level-l projector.

        RU:
        Вычисляет Phi^{(l)} для данного уровня:

            Phi^{(l)} = sum_{i != j} | <psi_i | P | psi_j> |^2

        где P — проектор уровня l.
        """
        phi = 0.0
        L = len(psi_level)
        if L <= 1:
            return 0.0

        for i in range(L):
            v_i = projector(psi_level[i])
            for j in range(L):
                if i == j:
                    continue
                v_j = projector(psi_level[j])
                amp = np.vdot(v_i, v_j)  # complex inner product
                phi += float(np.abs(amp) ** 2)

        return float(phi)

    def nullify(
        self,
        psi: Mapping[int, Sequence[Array]],
        projectors: Mapping[int, Callable[[Array], Array]],
    ) -> Dict[int, List[Array]]:
        """
        EN:
        Perform multiverse nullification for levels 0..self.levels.

        parameters:
        - psi: dict[level] -> list of state vectors (np.ndarray)
        - projectors: dict[level] -> callable(v) -> projected v

        returns:
        - new psi where foam Phi^{(l)} is reduced for each level
          (tends to 0 if convergence is reached).

        RU:
        Выполняет мультиверсное обнуление для уровней 0..self.levels.

        параметры:
        - psi: dict[level] -> список векторов-состояний (np.ndarray)
        - projectors: dict[level] -> функция(v) -> v_proj (проектор уровня l)

        возвращает:
        - новое psi, в котором пена Phi^{(l)} для каждого уровня уменьшена
          (при сходимости стремится к 0).
        """
        # copy to avoid in-place modification of input
        psi_new: Dict[int, List[Array]] = {
            l: [np.array(v, dtype=np.float64, copy=True) for v in vs]
            for l, vs in psi.items()
        }

        for l in range(self.levels + 1):
            if l not in psi_new or l not in projectors:
                continue

            P_l = projectors[l]

            for _ in range(self.max_iter):
                phi = self._phi_level(psi_new[l], P_l)
                if phi < self.eps:
                    break

                updated_level: List[Array] = []
                for v in psi_new[l]:
                    v_proj = P_l(v)
                    # EN: pseudo-gradient step towards the solution subspace
                    # RU: псевдо-градиентный шаг в сторону подпространства решения
                    v_new = v - self.lr * (v - v_proj)
                    v_new = self._normalize(v_new)
                    updated_level.append(v_new)

                psi_new[l] = updated_level

        return psi_new
