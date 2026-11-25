"""A kinematic bicycle car with steering lag, drag and command delay."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, replace

DT = 0.05


@dataclass(frozen=True)
class CarParams:
    wheelbase: float = 0.30  # m
    max_steer: float = 0.5  # rad
    steer_tau: float = 0.15  # s, how slowly the wheels follow the command
    throttle_gain: float = 3.0  # m/s^2 per unit throttle
    drag: float = 0.8  # 1/s
    delay_steps: int = 0  # command latency in simulation steps


NOMINAL = CarParams()


def randomized(base: CarParams, spread: float, max_delay: int, rng: random.Random) -> CarParams:
    """Scale each continuous parameter by a uniform factor in [1-spread, 1+spread]."""
    if not 0.0 <= spread < 1.0 or max_delay < 0:
        raise ValueError("need 0 <= spread < 1 and max_delay >= 0")

    def jitter(value: float) -> float:
        return value * rng.uniform(1.0 - spread, 1.0 + spread)

    return replace(
        base,
        wheelbase=jitter(base.wheelbase),
        steer_tau=jitter(base.steer_tau),
        throttle_gain=jitter(base.throttle_gain),
        drag=jitter(base.drag),
        delay_steps=rng.randint(0, max_delay),
    )


class Car:
    def __init__(self, params: CarParams, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        self.p = params
        self.x, self.y, self.heading = x, y, heading
        self.speed = 0.0
        self.steer = 0.0  # actual wheel angle, rad
        self._pending: deque[tuple[float, float]] = deque([(0.0, 0.0)] * params.delay_steps)

    def step(self, steer_cmd: float, throttle_cmd: float, dt: float = DT) -> None:
        """Advance one step. Commands are clipped to [-1, 1] and delayed."""
        cmd = (max(-1.0, min(1.0, steer_cmd)), max(-1.0, min(1.0, throttle_cmd)))
        self._pending.append(cmd)
        steer_cmd, throttle_cmd = self._pending.popleft()

        p = self.p
        self.steer += (steer_cmd * p.max_steer - self.steer) / p.steer_tau * dt
        self.speed += (p.throttle_gain * throttle_cmd - p.drag * self.speed) * dt
        self.heading += self.speed / p.wheelbase * math.tan(self.steer) * dt
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
