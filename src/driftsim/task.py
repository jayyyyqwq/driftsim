"""The driving task: follow a wavy line at a target speed."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from driftsim.car import DT, Car, CarParams

AMPLITUDE = 1.0  # m
WAVELENGTH = 12.0  # m
TARGET_SPEED = 2.0  # m/s
STEPS = 200
OFF_TRACK = 2.0  # m of cross-track error that ends the episode
OBS_DIM = 5
ACT_DIM = 2
THETA_DIM = OBS_DIM * ACT_DIM

ParamSampler = Callable[[random.Random], CarParams]


def path_y(x: float) -> float:
    return AMPLITUDE * math.sin(2 * math.pi * x / WAVELENGTH)


def path_heading(x: float) -> float:
    slope = AMPLITUDE * 2 * math.pi / WAVELENGTH * math.cos(2 * math.pi * x / WAVELENGTH)
    return math.atan(slope)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def observe(car: Car) -> list[float]:
    """Cross-track error, heading error, speed error, wheel angle, bias."""
    return [
        car.y - path_y(car.x),
        wrap(car.heading - path_heading(car.x)),
        car.speed - TARGET_SPEED,
        car.steer,
        1.0,
    ]


@dataclass(frozen=True)
class Result:
    cost: float  # lower is better
    rmse: float  # cross-track error, metres
    failed: bool  # left the track


def rollout(theta: Sequence[float], params: CarParams, start_noise: random.Random) -> Result:
    """Run one episode with the linear policy ``action = theta @ observation``."""
    if len(theta) != THETA_DIM:
        raise ValueError(f"theta needs {THETA_DIM} numbers, got {len(theta)}")
    car = Car(params, y=start_noise.uniform(-0.3, 0.3), heading=start_noise.uniform(-0.2, 0.2))
    cost = squared_error = 0.0
    for step in range(STEPS):
        obs = observe(car)
        steer = sum(theta[j] * obs[j] for j in range(OBS_DIM))
        throttle = sum(theta[OBS_DIM + j] * obs[j] for j in range(OBS_DIM))
        car.step(steer, throttle)

        e, he, ve = obs[0], obs[1], obs[2]
        cost += min(e * e, 4.0) + 0.5 * he * he + 0.1 * ve * ve
        squared_error += e * e
        if abs(e) > OFF_TRACK or not math.isfinite(car.x + car.y):
            remaining = STEPS - step - 1
            cost += remaining * 4.0
            squared_error += remaining * OFF_TRACK**2
            return Result(cost * DT, math.sqrt(squared_error / STEPS), True)
    return Result(cost * DT, math.sqrt(squared_error / STEPS), False)
