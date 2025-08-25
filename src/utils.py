from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import random


def ensure_timestamped_dir(root: Path, leaf: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    p = root / ts / leaf
    p.mkdir(parents=True, exist_ok=True)
    return p


def cartesian_product(grid: Dict[str, Sequence]) -> List[Dict[str, object]]:
    from itertools import product

    keys = list(grid.keys())
    combos = []
    for values in product(*[grid[k] for k in keys]):
        combos.append(dict(zip(keys, values)))
    return combos


def random_sample_grid(
    grid: Dict[str, Sequence],
    num_samples: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    """
    Randomly sample parameter combinations from a grid.

    Args:
        grid: Mapping from parameter name to a sequence of possible values.
        num_samples: Number of combinations to sample.
        rng: A seeded random.Random instance for reproducibility.
    """
    combos = cartesian_product(grid)
    if num_samples >= len(combos):
        return combos
    return rng.sample(combos, k=num_samples)
