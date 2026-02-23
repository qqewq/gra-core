Сейчас это только начало README до середины первого кодового блока. Полный рабочий `README.md` должен продолжаться дальше и закрывать блок ```bash```.

Возьми весь файл в таком виде:

```markdown
# GRA-Core: Multiverse Nullification & Fractal Agents  
# GRA-Core: Мультиверсное обнуление и фрактальные агенты

> English below · Русский ниже

---

## ???? English

### What is GRA-Core?

`gra-core` is a minimal implementation of the **GRA Multiverse Nullification**
architecture: a multi-level foam nullification engine for complex systems,
plus a runtime for **fractal language context** and **loop-free agents**.

The project unifies several research codebases into one importable toolkit:

- Multiverse nullification functional (NumPy + JAX)
- Fractal language / context engine
- Agent runtime with GRA-style meta-reset
- Vertical module for biological aging models

This repository is intended as a **reference implementation** for experiments
in stability, AI safety, and multi-agent systems.

### Installation

```bash
pip install gra-core
```

Dev mode:

```bash
git clone https://github.com/qqewq/gra-core.git
cd gra-core
pip install -e .
```

### Quickstart

#### 1. Multiverse nullification (NumPy)

```python
import numpy as np
from gra_core import GraNullifier
from gra_core.utils import homogeneous_projector, make_level_dict

dim = 4
n_per_level = 5
levels = [github](https://github.com/TheQuantScientist/Multi-Agent)

psi = make_level_dict(levels=levels, dim=dim, n_per_level=n_per_level, seed=42)
P = homogeneous_projector(dim)
projectors = {0: P, 1: P}

nullifier = GraNullifier(levels=1, alpha=0.9)
psi_star = nullifier.nullify(psi, projectors)
```

See `examples/nullify_toy_model.ipynb` for a full, bilingual demo.

#### 2. JAX integration

```python
import jax
import jax.numpy as jnp

from gra_core.jax_nullification import (
    homogeneous_projector_jax,
    multilevel_phi_jax,
    gradient_step_states,
)

dim = 4
n_per_level = 5
levels = [github](https://github.com/TheQuantScientist/Multi-Agent)

key = jax.random.PRNGKey(0)

def random_level_states(key, n, d):
    v = jax.random.normal(key, (n, d))
    v = v / (jnp.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

key0, key1 = jax.random.split(key)
psi = {
    0: random_level_states(key0, n_per_level, dim),
    1: random_level_states(key1, n_per_level, dim),
}

P = homogeneous_projector_jax(dim)
projectors = {0: P, 1: P}
lambdas = {0: 1.0, 1: 1.0}

def loss_fn(psi_dict):
    return multilevel_phi_jax(psi_dict, projectors, levels, lambdas)

psi_current = {l: psi[l] for l in levels}
for _ in range(50):
    psi_current = gradient_step_states(
        psi_current, projectors, levels, lambdas, lr=0.1
    )
```

See `examples/jax_nullify_toy.ipynb` for a full JAX notebook.

#### 3. GRA agents (loop-free)

```python
from gra_core import GraAgent, GraMultiAgentSystem

def counting_policy(state, memory):
    count = memory.get("count", 0) + 1
    return {"echo": state, "step": count}, {"count": count}

def reset_memory(memory):
    return {}

agent_a = GraAgent("A", counting_policy, reset_fn=reset_memory)
agent_b = GraAgent("B", counting_policy, reset_fn=reset_memory)

system = GraMultiAgentSystem({"A": agent_a, "B": agent_b})

state = {"x": 42}
actions = system.step(state)
system.global_reset()
```

See `examples/gra_agent_loop_free.ipynb`.

#### 4. Fractal context engine

```python
from gra_core import FractalContextEngine

engine = FractalContextEngine(depth=4)
ctx = engine.build_fractal_context("quantum gravity and information")
levels = ctx.by_level()
```

See `examples/fractal_context_demo.ipynb`.

### Source repositories

- `GRA-Multiverse-Nullification` – formal multiverse nullification  
- `Lingua-GRA-Fractal-AGI` – fractal language / cognition layer  
- `moltnew-gra-agents` – multi-agent runtime  
- `GRA-Aging-Reset` – biological vertical (aging models)

`gra-core` provides a unified, lightweight interface on top of them.

### License

MIT License (see `LICENSE`).

---

## ???? Русский

### Что такое GRA-Core?

`gra-core` — минимальная реализация архитектуры **GRA мультиверсного обнуления**:
многоуровневый движок обнуления «пены» в сложных системах, плюс рантайм для
**фрактального языкового контекста** и **агентов без зацикливания**.

Репозиторий объединяет несколько исследовательских кодовых баз в единый
инструментарий:

- Мультиверсный функционал обнуления (NumPy + JAX)
- Фрактальный языковой / контекстный движок
- Агентный рантайм с GRA-подобным мета-сбросом
- Вертикальный модуль для биомоделей старения

Проект задуман как **эталонная реализация** для экспериментов со стабильностью,
AI safety и мультиагентными системами.

### Установка

```bash
pip install gra-core
```

Режим разработки:

```bash
git clone https://github.com/qqewq/gra-core.git
cd gra-core
pip install -e .
```

### Быстрый старт

#### 1. Мультиверсное обнуление (NumPy)

```python
import numpy as np
from gra_core import GraNullifier
from gra_core.utils import homogeneous_projector, make_level_dict

dim = 4
n_per_level = 5
levels = [github](https://github.com/TheQuantScientist/Multi-Agent)

psi = make_level_dict(levels=levels, dim=dim, n_per_level=n_per_level, seed=42)
P = homogeneous_projector(dim)
projectors = {0: P, 1: P}

nullifier = GraNullifier(levels=1, alpha=0.9)
psi_star = nullifier.nullify(psi, projectors)
```

Полный двуязычный пример — в `examples/nullify_toy_model.ipynb`.

#### 2. JAX-интеграция

```python
import jax
import jax.numpy as jnp

from gra_core.jax_nullification import (
    homogeneous_projector_jax,
    multilevel_phi_jax,
    gradient_step_states,
)

dim = 4
n_per_level = 5
levels = [github](https://github.com/TheQuantScientist/Multi-Agent)

key = jax.random.PRNGKey(0)

def random_level_states(key, n, d):
    v = jax.random.normal(key, (n, d))
    v = v / (jnp.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

key0, key1 = jax.random.split(key)
psi = {
    0: random_level_states(key0, n_per_level, dim),
    1: random_level_states(key1, n_per_level, dim),
}

P = homogeneous_projector_jax(dim)
projectors = {0: P, 1: P}
lambdas = {0: 1.0, 1: 1.0}

def loss_fn(psi_dict):
    return multilevel_phi_jax(psi_dict, projectors, levels, lambdas)

psi_current = {l: psi[l] for l in levels}
for _ in range(50):
    psi_current = gradient_step_states(
        psi_current, projectors, levels, lambdas, lr=0.1
    )
```

Полный JAX-ноутбук — `examples/jax_nullify_toy.ipynb`.

#### 3. GRA-агенты (без зацикливания)

```python
from gra_core import GraAgent, GraMultiAgentSystem

def counting_policy(state, memory):
    count = memory.get("count", 0) + 1
    return {"echo": state, "step": count}, {"count": count}

def reset_memory(memory):
    return {}

agent_a = GraAgent("A", counting_policy, reset_fn=reset_memory)
agent_b = GraAgent("B", counting_policy, reset_fn=reset_memory)

system = GraMultiAgentSystem({"A": agent_a, "B": agent_b})

state = {"x": 42}
actions = system.step(state)
system.global_reset()
```

См. `examples/gra_agent_loop_free.ipynb`.

#### 4. Фрактальный контекст

```python
from gra_core import FractalContextEngine

engine = FractalContextEngine(depth=4)
ctx = engine.build_fractal_context("квантовая гравитация и информация")
levels = ctx.by_level()
```

См. `examples/fractal_context_demo.ipynb`.

### Исходные репозитории

- `GRA-Multiverse-Nullification` — формализм мультиверсного обнуления  
- `Lingua-GRA-Fractal-AGI` — фрактальный языковой / когнитивный слой  
- `moltnew-gra-agents` — агентный рантайм  
- `GRA-Aging-Reset` — биологический vertical (старение)

`gra-core` даёт единый облегчённый интерфейс поверх них.

### Лицензия

MIT License (см. `LICENSE`).