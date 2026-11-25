"""Train with and without domain randomisation, then test on unseen cars."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from driftsim.ars import ArsConfig, train
from driftsim.car import NOMINAL, CarParams, randomized
from driftsim.task import ParamSampler, rollout

TRAIN_SPREAD, TRAIN_DELAY = 0.30, 3
REAL_SPREAD, REAL_DELAY = 0.50, 6  # wider and laggier than anything seen in training
EVAL_SEED = 10_000


def nominal_sampler(_rng: random.Random) -> CarParams:
    return NOMINAL


def randomized_sampler(rng: random.Random) -> CarParams:
    return randomized(NOMINAL, TRAIN_SPREAD, TRAIN_DELAY, rng)


def real_sampler(rng: random.Random) -> CarParams:
    """Stand-in for the physical car: a wider spread than training covered."""
    return randomized(NOMINAL, REAL_SPREAD, REAL_DELAY, rng)


@dataclass(frozen=True)
class Score:
    rmse: float  # mean cross-track error, metres
    failure_rate: float  # share of cars that left the track


def evaluate(theta: np.ndarray, sampler: ParamSampler, cars: int = 40) -> Score:
    results = []
    for i in range(cars):
        rng = random.Random(EVAL_SEED + i)
        results.append(rollout(theta.tolist(), sampler(rng), rng))
    return Score(
        rmse=float(np.mean([r.rmse for r in results])),
        failure_rate=float(np.mean([r.failed for r in results])),
    )


@dataclass(frozen=True)
class Outcome:
    trained_on: str
    on_nominal: Score
    on_real: Score


def run(seeds: int = 5, cfg: ArsConfig | None = None) -> list[Outcome]:
    """Average each strategy over several training seeds, since one run is noisy."""
    cfg = cfg or ArsConfig()
    outcomes = []
    for name, sampler in (
        ("nominal sim only", nominal_sampler),
        ("randomised sims", randomized_sampler),
    ):
        nominal, real = [], []
        for seed in range(seeds):
            theta, _ = train(sampler, cfg, seed)
            nominal.append(evaluate(theta, nominal_sampler, cars=1))
            real.append(evaluate(theta, real_sampler))
        outcomes.append(Outcome(name, mean_score(nominal), mean_score(real)))
    return outcomes


def mean_score(scores: list[Score]) -> Score:
    return Score(
        rmse=float(np.mean([s.rmse for s in scores])),
        failure_rate=float(np.mean([s.failure_rate for s in scores])),
    )
