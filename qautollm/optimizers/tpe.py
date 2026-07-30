"""Optuna TPE wrapper (Brief §7's ``optimizer.name: tpe``).

Resume strategy: pickle the live ``optuna.Study`` (base64-encoded so it fits
the JSON checkpoint). This was *not* the first design tried here — an
earlier version replayed (params, objectives) history through
``add_trial()`` to rebuild the study, on the assumption that
``TPESampler`` is a pure function of (seed, completed-trial data). That
assumption is false: ``TPESampler`` also consumes its internal RNG on every
live ``ask()`` call, independent of what the trial is later told, so a
study rebuilt purely from completed-trial data diverges from a study that
was actually asked. Verified with a scripted pickle/continue comparison
before landing this; only pickling the real ``Study`` object reproduces
bit-identical future suggestions after resume.

Every observation — whether it came from the shared init design or from a
TPE-guided ``ask()`` — is still recorded into the study (via ``add_trial``
for init-design points, since those never went through ``ask()``) so the
sampler learns from the full history, matching every other optimizer's
paired comparison.
"""

from __future__ import annotations

import base64
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState

from qautollm.core.registry import OPTIMIZERS
from qautollm.optimizers.base import OptimizerInterface
from qautollm.space.encoding import CanonicalConfig, encode
from qautollm.space.search_space import SearchSpace

if TYPE_CHECKING:
    from qautollm.core.config import OptimizerConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

_DIRECTIONS = ["minimize", "minimize", "minimize"]  # matches Brief §5.5's objective vector


@OPTIMIZERS.register("tpe")
class TPEOptimizer(OptimizerInterface):
    def __init__(
        self,
        search_space: SearchSpace,
        seed: int,
        init_design: list[CanonicalConfig],
        optimizer_config: OptimizerConfig | None = None,
    ) -> None:
        super().__init__(search_space, seed, init_design, optimizer_config)
        self._study = self._new_study()
        self._pending_trial: optuna.trial.Trial | None = None

    def _new_study(self) -> optuna.study.Study:
        return optuna.create_study(
            directions=_DIRECTIONS,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

    def suggest(self) -> CanonicalConfig:
        init = self._next_init_point()
        if init is not None:
            self._pending_trial = None
            return init

        trial = self._study.ask()
        raw = self._sample_raw(trial)
        self._pending_trial = trial
        return encode(raw, self.search_space)

    def _sample_raw(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        raw: dict[str, Any] = {}
        for v in self.search_space.variables:
            if not v.is_conditional:
                raw[v.name] = trial.suggest_categorical(v.name, list(v.values))
        for v in self.search_space.variables:
            if v.is_conditional and v.condition.is_active(raw):
                raw[v.name] = trial.suggest_categorical(v.name, list(v.values))
        return raw

    def update(self, config: CanonicalConfig, objectives: np.ndarray) -> None:
        values = [float(x) for x in objectives]
        if self._pending_trial is not None:
            self._study.tell(self._pending_trial, values)
        else:
            self._study.add_trial(self._completed_trial(config.raw, values))
        self._pending_trial = None

    def _completed_trial(
        self, raw: dict[str, Any], values: list[float]
    ) -> optuna.trial.FrozenTrial:
        distributions = {
            v.name: CategoricalDistribution(list(v.values))
            for v in self.search_space.variables
            if v.name in raw
        }
        return optuna.trial.create_trial(
            params=raw, distributions=distributions, values=values, state=TrialState.COMPLETE
        )

    def state_dict(self) -> dict[str, Any]:
        blob = base64.b64encode(pickle.dumps(self._study)).decode("ascii")
        return {"init_cursor": self._init_cursor, "study_pickle_b64": blob}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._init_cursor = state["init_cursor"]
        self._study = pickle.loads(base64.b64decode(state["study_pickle_b64"]))
        self._pending_trial = None
