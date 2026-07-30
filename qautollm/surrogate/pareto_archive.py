"""Non-dominated archive + hypervolume (Brief §5.4). Minimization
convention throughout (Brief §5.5: objectives are ``[-F1, -faithfulness,
latency]``, lower is better for every entry).

Hypervolume is computed via BoTorch's ``Hypervolume`` (already a pinned
dependency for EHVI) rather than a hand-rolled sweep algorithm — P6
("hypervolume validated against pymoo reference values") is where a
from-scratch/optimized implementation, if ever needed for speed, gets
checked against pymoo as ground truth; for now, delegating to a well-tested
library implementation *is* the correctness guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if ``a`` is at least as good as ``b`` in every objective and
    strictly better in at least one (minimization convention).
    """
    return bool(np.all(a <= b) and np.any(a < b))


@dataclass
class ParetoArchive:
    points: list[np.ndarray] = field(default_factory=list)
    tags: list[Any] = field(default_factory=list)

    def update(self, objectives: np.ndarray, tag: Any = None) -> bool:
        """Insert ``objectives`` if it isn't dominated by anything already
        in the archive, pruning anything it now dominates. Returns whether
        it was added.
        """
        objectives = np.asarray(objectives, dtype=np.float64)
        for existing in self.points:
            if dominates(existing, objectives):
                return False

        keep = [i for i, p in enumerate(self.points) if not dominates(objectives, p)]
        self.points = [self.points[i] for i in keep]
        self.tags = [self.tags[i] for i in keep]
        self.points.append(objectives.copy())
        self.tags.append(tag)
        return True

    @property
    def front(self) -> np.ndarray:
        if not self.points:
            return np.zeros((0, 0))
        return np.stack(self.points)

    def hypervolume(self, reference_point: np.ndarray) -> float:
        """Hypervolume dominated between the front and ``reference_point``
        (both in minimization convention; ``reference_point`` must be
        elementwise >= every point in the archive).
        """
        if not self.points:
            return 0.0
        ref_max = -np.asarray(reference_point, dtype=np.float64)
        pts_max = -self.front
        hv = Hypervolume(ref_point=torch.tensor(ref_max, dtype=torch.float64))
        return float(hv.compute(torch.tensor(pts_max, dtype=torch.float64)))
