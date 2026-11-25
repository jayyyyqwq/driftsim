"""Augmented Random Search (Mania et al., 2018), for a linear policy.

Gradient-free: nudge the policy in random directions, keep what lowers the
cost. A good fit here because the policy is ten numbers and a rollout is cheap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from driftsim.task import THETA_DIM, ParamSampler, rollout


@dataclass(frozen=True)
class ArsConfig:
    iterations: int = 40
    directions: int = 8
    top: int = 4
    step_size: float = 0.3
    noise: float = 0.15
    episodes_per_eval: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.top <= self.directions:
            raise ValueError("top must be between 1 and directions")


def mean_cost(theta: np.ndarray, sampler: ParamSampler, seeds: list[int]) -> float:
    """Average cost over episodes. Each seed fixes both the car and the start,
    so a direction and its mirror image are compared on identical conditions."""
    total = 0.0
    for seed in seeds:
        rng = random.Random(seed)
        total += rollout(theta.tolist(), sampler(rng), rng).cost
    return total / len(seeds)


def train(
    sampler: ParamSampler, cfg: ArsConfig | None = None, seed: int = 0
) -> tuple[np.ndarray, list[float]]:
    """Returns the final policy and the cost of the current policy per iteration."""
    cfg = cfg or ArsConfig()
    rng = np.random.default_rng(seed)
    theta = np.zeros(THETA_DIM)
    history: list[float] = []

    for _ in range(cfg.iterations):
        seeds = [int(s) for s in rng.integers(0, 2**31, cfg.episodes_per_eval)]
        deltas = rng.normal(size=(cfg.directions, THETA_DIM))
        plus = np.array([mean_cost(theta + cfg.noise * d, sampler, seeds) for d in deltas])
        minus = np.array([mean_cost(theta - cfg.noise * d, sampler, seeds) for d in deltas])

        # costs, so lower is better: the best directions have the smallest min(plus, minus)
        best = np.argsort(np.minimum(plus, minus))[: cfg.top]
        spread = np.concatenate([plus[best], minus[best]]).std() or 1.0
        step = ((minus[best] - plus[best])[:, None] * deltas[best]).sum(axis=0)
        theta = theta + cfg.step_size / (cfg.top * spread) * step
        history.append(mean_cost(theta, sampler, seeds))
    return theta, history
