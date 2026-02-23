# src/gra_core/fractal_lingua.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class FractalNode:
    """
    EN:
    FractalNode represents a single node in the fractal context graph.

    - level: abstraction level (0 = local, higher = more abstract)
    - label: human-readable concept / description
    - children: indices of child nodes (within the same or lower levels)

    RU:
    FractalNode — это один узел фрактального контекстного графа.

    - level: уровень абстракции (0 = локальный, выше = более абстрактный)
    - label: текстовое обозначение концепта
    - children: индексы дочерних узлов (на этом или нижних уровнях)
    """

    id: int
    level: int
    label: str
    children: List[int] = field(default_factory=list)


@dataclass
class FractalContext:
    """
    EN:
    FractalContext is a lightweight, multi-level map of concepts.
    You can treat it as a very small knowledge graph:
    - nodes: all concepts and abstractions
    - by_level(): quick grouped view by abstraction levels.

    RU:
    FractalContext — лёгкая многоуровневая карта концептов.
    Это упрощённый knowledge graph:
    - nodes: все концепты и абстракции
    - by_level(): быстрый группированный вид по уровням абстракции.
    """

    query: str
    nodes: Dict[int, FractalNode]  # id -> node

    def by_level(self) -> Dict[int, List[FractalNode]]:
        """
        EN:
        Group nodes by their abstraction level.

        RU:
        Группировка узлов по уровню абстракции.
        """
        levels: Dict[int, List[FractalNode]] = {}
        for node in self.nodes.values():
            levels.setdefault(node.level, []).append(node)
        return levels


class FractalContextEngine:
    """
    EN:
    FractalContextEngine builds a multi-level (fractal) context map.

    Minimal version:
    - level 0: original query as a single root node
    - level 1: base concepts (tokens or provided list)
    - higher levels: abstract aggregating nodes (placeholders for real logic)

    You can later plug in:
    - embeddings + clustering
    - knowledge graphs
    - GRA invariants and multiverse structure.

    RU:
    FractalContextEngine строит многоуровневую (фрактальную) карту контекста.

    Минимальная версия:
    - уровень 0: исходный запрос как корневой узел
    - уровень 1: базовые концепты (токены или заданный список)
    - более высокие уровни: абстрактные агрегирующие узлы (заглушки под реальную логику)

    В дальнейшем сюда можно подключить:
    - эмбеддинги + кластеризацию
    - графы знаний
    - инварианты GRA и мультиверсную структуру.
    """

    def __init__(self, depth: int = 3) -> None:
        """
        EN:
        depth: number of abstraction levels (>= 1).

        RU:
        depth: количество уровней абстракции (>= 1).
        """
        assert depth >= 1
        self.depth = depth

    def build_fractal_context(
        self,
        query: str,
        base_concepts: Optional[Sequence[str]] = None,
    ) -> FractalContext:
        """
        EN:
        Build a fractal context for a given text query.

        parameters:
        - query: original text query / description
        - base_concepts: optional list of level-1 concepts
          (if None, we use simple whitespace tokenization)

        returns:
        - FractalContext with nodes across levels 0..depth-1

        RU:
        Строит фрактальный контекст для заданного текстового запроса.

        параметры:
        - query: исходный текстовый запрос / описание
        - base_concepts: опциональный список концептов уровня 1
          (если None — используется простое разбиение по пробелам)

        возвращает:
        - FractalContext с узлами на уровнях 0..depth-1
        """
        nodes: Dict[int, FractalNode] = {}
        node_id = 0

        # Level 0: root node = full query
        root = FractalNode(id=node_id, level=0, label=query)
        nodes[node_id] = root
        node_id += 1

        # Level 1: base concepts
        if base_concepts is None:
            tokens = [t for t in query.split() if t.strip()]
        else:
            tokens = list(base_concepts)

        level1_ids: List[int] = []
        for tok in tokens:
            n = FractalNode(id=node_id, level=1, label=tok)
            nodes[node_id] = n
            level1_ids.append(node_id)
            node_id += 1

        # connect root -> level1
        nodes[0].children = level1_ids

        # Levels 2..depth-1: simple abstract aggregators (placeholder logic)
        prev_level_ids = level1_ids
        for level in range(2, self.depth):
            new_ids: List[int] = []
            if not prev_level_ids:
                break

            label = f"abstract_level_{level}"
            n = FractalNode(id=node_id, level=level, label=label)
            n.children = prev_level_ids.copy()
            nodes[node_id] = n
            new_ids.append(node_id)
            node_id += 1

            prev_level_ids = new_ids

        return FractalContext(query=query, nodes=nodes)
