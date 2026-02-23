# src/gra_core/bio_aging.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
Array = np.ndarray


@dataclass
class AgingState:
    """
    EN:
    AgingState represents a vector of biological markers
    (e.g., clocks, physiological indices, etc.).

    RU:
    AgingState представляет вектор биологических маркеров
    (напр., часы старения, физиологические индексы и т.п.).
    """

    values: Array  # shape: (d,)

    def copy(self) -> "AgingState":
        return AgingState(values=self.values.copy())


@dataclass
class AgingResetParameters:
    """
    EN:
    Parameters controlling a toy aging dynamics and reset:

    - drift_rate: how fast markers drift away from youthful reference
    - noise_scale: stochastic fluctuations (environment / measurement)
    - reset_strength: how strongly GRA-reset pulls state back
      to the youthful reference.

    RU:
    Параметры игрушечной динамики старения и сброса:

    - drift_rate: скорость дрейфа маркеров от молодого референса
    - noise_scale: стохастические флуктуации (среда / шум измерений)
    - reset_strength: сила притяжения GRA-сброса к молодому референсу.
    """

    drift_rate: float = 0.01
    noise_scale: float = 0.05
    reset_strength: float = 0.2


class AgingResetModel:
    """
    EN:
    AgingResetModel is a toy dynamical system:
    - state: vector of biological markers
    - reference: "youthful" or target state
    - dynamics: drift + noise
    - reset: GRA-style nullification towards the reference

    It is NOT a biomedical model; it is a sandbox vertical
    to demonstrate how GRA-nullification can be interpreted
    as "aging reset" in a multi-marker space.

    RU:
    AgingResetModel — игрушечная динамическая система:
    - state: вектор биологических маркеров
    - reference: «молодое» или целевое состояние
    - dynamics: дрейф + шум
    - reset: GRA-обнуление в сторону референса

    Это НЕ биомедицинская модель; это демонстрационный vertical,
    показывающий, как GRA-обнуление можно трактовать как
    «reset старения» в пространстве мульти-маркеров.
    """

    def __init__(
        self,
        reference: Sequence[float],
        params: AgingResetParameters | None = None,
    ) -> None:
        """
        EN:
        parameters:
        - reference: youthful / target marker vector
        - params: optional AgingResetParameters

        RU:
        параметры:
        - reference: вектор «молодых» / целевых маркеров
        - params: опционально AgingResetParameters
        """
        self.reference = np.array(reference, dtype=np.float64)
        self.dim = self.reference.shape[0]
        self.params = params or AgingResetParameters()

    def step_dynamics(self, state: AgingState) -> AgingState:
        """
        EN:
        Pure aging dynamics (no reset):
        state_{t+1} = state_t + drift + noise

        RU:
        Чистая динамика старения (без сброса):
        state_{t+1} = state_t + дрейф + шум
        """
        drift = self.params.drift_rate * (np.ones(self.dim))
        noise = self.params.noise_scale * np.random.randn(self.dim)
        new_values = state.values + drift + noise
        return AgingState(values=new_values)

    def apply_reset(self, state: AgingState) -> AgingState:
        """
        EN:
        GRA-style reset step:
        move state towards the youthful reference.

        RU:
        GRA-подобный шаг сброса:
        движение состояния в сторону молодого референса.
        """
        direction = self.reference - state.values
        new_values = state.values + self.params.reset_strength * direction
        return AgingState(values=new_values)

    def simulate(
        self,
        initial: AgingState,
        steps: int = 50,
        reset_schedule: Dict[int, bool] | None = None,
    ) -> List[AgingState]:
        """
        EN:
        Simulate aging dynamics with optional resets.

        parameters:
        - initial: starting state
        - steps: number of time steps
        - reset_schedule: dict[step_index] -> bool;
          if True at step t, apply reset AFTER dynamics

        returns:
        - list of states (length = steps + 1), including initial

        RU:
        Моделирует динамику старения с опциональными сбросами.

        параметры:
        - initial: начальное состояние
        - steps: число шагов по времени
        - reset_schedule: словарь step_index -> bool;
          если True на шаге t, то сброс применяется ПОСЛЕ динамики

        возвращает:
        - список состояний (длины steps + 1), включая начальное
        """
        if reset_schedule is None:
            reset_schedule = {}

        history: List[AgingState] = [initial.copy()]
        state = initial.copy()

        for t in range(steps):
            # 1) natural aging dynamics
            state = self.step_dynamics(state)

            # 2) optional reset
            if reset_schedule.get(t, False):
                state = self.apply_reset(state)

            history.append(state.copy())

        return history
