# src/gra_core/agents.py

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple


PolicyFn = Callable[[Mapping[str, Any], Dict[str, Any]],
                    Tuple[Mapping[str, Any], Dict[str, Any]]]
ResetFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class GraAgent:
    """
    EN:
    GraAgent is a minimal agent with:
    - name: identifier
    - policy: callable(state, memory) -> (action, new_memory)
    - reset_fn: optional GRA-based cognitive reset for its memory

    It is designed to be "GRA-friendly":
    you can plug meta-reset / nullification into reset_fn.

    RU:
    GraAgent — минимальный агент с:
    - name: идентификатор
    - policy: функция(state, memory) -> (action, new_memory)
    - reset_fn: опциональный GRA-сброс когнитивного состояния (памяти)

    Специально сделан «GRA-дружественным»:
    в reset_fn можно подключать мета-сброс / обнуление.
    """

    def __init__(
        self,
        name: str,
        policy: PolicyFn,
        reset_fn: Optional[ResetFn] = None,
    ) -> None:
        """
        EN:
        Create a new agent.

        parameters:
        - name: agent identifier
        - policy: main decision function
        - reset_fn: optional cognitive reset function

        RU:
        Создать нового агента.

        параметры:
        - name: идентификатор агента
        - policy: основная функция принятия решений
        - reset_fn: опциональная функция когнитивного сброса
        """
        self.name = name
        self.policy = policy
        self.reset_fn = reset_fn
        self.memory: Dict[str, Any] = {}

    def step(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        EN:
        Perform one step:
        - takes global/local state
        - returns action
        - updates internal memory

        RU:
        Выполнить один шаг:
        - принимает глобальное/локальное состояние
        - возвращает действие
        - обновляет внутреннюю память
        """
        action, new_memory = self.policy(state, self.memory)
        self.memory = dict(new_memory)
        return action

    def reset_cognition(self) -> None:
        """
        EN:
        Apply cognitive reset (if reset_fn is provided).

        RU:
        Применить когнитивный сброс (если задан reset_fn).
        """
        if self.reset_fn is not None:
            self.memory = self.reset_fn(self.memory)


class GraMultiAgentSystem:
    """
    EN:
    GraMultiAgentSystem orchestrates a set of GraAgent instances.

    - agents: mapping name -> GraAgent
    - meta_reset: optional callable(agents) to apply higher-level
      GRA meta-reset across the whole system.

    RU:
    GraMultiAgentSystem управляет набором экземпляров GraAgent.

    - agents: соответствие name -> GraAgent
    - meta_reset: опциональная функция(agents) для применения
      мета-сброса GRA ко всей системе в целом.
    """

    def __init__(
        self,
        agents: Mapping[str, GraAgent],
        meta_reset: Optional[Callable[[Mapping[str, GraAgent]], None]] = None,
    ) -> None:
        """
        EN:
        parameters:
        - agents: dict of agent_name -> GraAgent
        - meta_reset: optional global reset function

        RU:
        параметры:
        - agents: словарь agent_name -> GraAgent
        - meta_reset: опциональная функция глобального сброса
        """
        self.agents: Dict[str, GraAgent] = dict(agents)
        self.meta_reset = meta_reset

    def step(self, global_state: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
        """
        EN:
        Execute one synchronized step for all agents.

        parameters:
        - global_state: shared state visible to all agents

        returns:
        - dict[name] -> action

        RU:
        Выполнить один синхронизированный шаг для всех агентов.

        параметры:
        - global_state: общее состояние, видимое всем агентам

        возвращает:
        - словарь name -> action
        """
        actions: Dict[str, Mapping[str, Any]] = {}
        for name, agent in self.agents.items():
            actions[name] = agent.step(global_state)
        return actions

    def global_reset(self) -> None:
        """
        EN:
        Apply cognitive reset to all agents, then optional meta_reset.

        RU:
        Применить когнитивный сброс ко всем агентам,
        затем опциональный мета-сброс meta_reset.
        """
        for agent in self.agents.values():
            agent.reset_cognition()
        if self.meta_reset is not None:
            self.meta_reset(self.agents)
