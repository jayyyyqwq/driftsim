import math
import random

import numpy as np
import pytest

from driftsim.ars import ArsConfig, mean_cost, train
from driftsim.car import DT, NOMINAL, Car, CarParams, randomized
from driftsim.cli import format_table, main
from driftsim.experiment import (
    Score,
    evaluate,
    mean_score,
    nominal_sampler,
    randomized_sampler,
    real_sampler,
)
from driftsim.task import STEPS, THETA_DIM, observe, path_heading, path_y, rollout, wrap

GOOD_THETA = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0]  # hand-picked, see below


def test_straight_line_with_no_steering() -> None:
    car = Car(NOMINAL)
    for _ in range(100):
        car.step(0.0, 0.5)
    assert car.y == pytest.approx(0.0, abs=1e-9) and car.heading == 0.0 and car.x > 0


def test_speed_settles_at_gain_over_drag() -> None:
    car = Car(NOMINAL)
    for _ in range(600):
        car.step(0.0, 1.0)
    assert car.speed == pytest.approx(NOMINAL.throttle_gain / NOMINAL.drag, rel=1e-3)


def test_steady_turn_matches_bicycle_geometry() -> None:
    car = Car(NOMINAL)
    for _ in range(400):
        car.step(1.0, 1.0)
    rate_before = car.heading
    car.step(1.0, 1.0)
    measured = (car.heading - rate_before) / DT
    expected = car.speed / NOMINAL.wheelbase * math.tan(NOMINAL.max_steer)
    assert measured == pytest.approx(expected, rel=1e-2)


def test_steering_lags_the_command() -> None:
    car = Car(NOMINAL)
    car.step(1.0, 0.0)
    assert 0 < car.steer < NOMINAL.max_steer * 0.5


def test_commands_are_delayed_and_clipped() -> None:
    car = Car(CarParams(delay_steps=3))
    for _ in range(3):
        car.step(1.0, 1.0)
        assert car.speed == 0.0 and car.steer == 0.0
    car.step(1.0, 1.0)
    assert car.speed > 0.0
    wild = Car(NOMINAL)
    wild.step(50.0, 50.0)
    tame = Car(NOMINAL)
    tame.step(1.0, 1.0)
    assert wild.speed == tame.speed and wild.steer == tame.steer


def test_randomized_stays_in_range_and_is_repeatable() -> None:
    for seed in range(50):
        p = randomized(NOMINAL, 0.3, 2, random.Random(seed))
        assert 0.7 * NOMINAL.wheelbase <= p.wheelbase <= 1.3 * NOMINAL.wheelbase
        assert 0 <= p.delay_steps <= 2
    a = randomized(NOMINAL, 0.3, 2, random.Random(1))
    assert a == randomized(NOMINAL, 0.3, 2, random.Random(1)) and a != NOMINAL
    with pytest.raises(ValueError):
        randomized(NOMINAL, 1.0, 0, random.Random(0))


def test_observation_is_zero_error_on_the_line() -> None:
    car = Car(NOMINAL, x=3.0, y=path_y(3.0), heading=path_heading(3.0))
    obs = observe(car)
    assert obs[0] == pytest.approx(0.0) and obs[1] == pytest.approx(0.0)
    assert obs[4] == 1.0


def test_wrap_keeps_angles_in_range() -> None:
    assert abs(wrap(3 * math.pi)) == pytest.approx(math.pi)
    assert wrap(0.5) == 0.5


def test_rollout_validates_and_is_deterministic() -> None:
    with pytest.raises(ValueError):
        rollout([0.0] * 3, NOMINAL, random.Random(0))
    a = rollout(GOOD_THETA, NOMINAL, random.Random(5))
    assert a == rollout(GOOD_THETA, NOMINAL, random.Random(5))


def test_zero_policy_goes_nowhere_and_a_bad_policy_leaves_the_track() -> None:
    idle = rollout([0.0] * THETA_DIM, NOMINAL, random.Random(0))
    assert not idle.failed
    # a gentle constant steer with throttle on sweeps a wide arc away from the line
    veer = rollout([0, 0, 0, 0, 0.1, 0, 0, 0, 0, 1.0], NOMINAL, random.Random(0))
    assert veer.failed and veer.cost > idle.cost


def test_ars_config_validation() -> None:
    with pytest.raises(ValueError):
        ArsConfig(directions=2, top=3)


def test_training_beats_the_zero_policy() -> None:
    cfg = ArsConfig(iterations=25)
    theta, history = train(nominal_sampler, cfg, seed=0)
    assert theta.shape == (THETA_DIM,) and len(history) == 25
    seeds = list(range(20))
    trained = mean_cost(theta, nominal_sampler, seeds)
    assert trained < mean_cost(np.zeros(THETA_DIM), nominal_sampler, seeds)


def test_training_is_repeatable() -> None:
    cfg = ArsConfig(iterations=5)
    a, _ = train(randomized_sampler, cfg, seed=3)
    b, _ = train(randomized_sampler, cfg, seed=3)
    assert np.array_equal(a, b)


def test_evaluate_and_samplers() -> None:
    theta, _ = train(nominal_sampler, ArsConfig(iterations=10), seed=0)
    score = evaluate(theta, real_sampler, cars=6)
    assert score.rmse > 0 and 0.0 <= score.failure_rate <= 1.0
    assert mean_score([Score(1.0, 0.0), Score(3.0, 1.0)]) == Score(2.0, 0.5)


def test_cli_run_and_table(capsys) -> None:
    assert main(["run", "--iterations", "3", "--seeds", "1"]) == 0
    out = capsys.readouterr().out
    assert "nominal sim only" in out and "randomised sims" in out
    assert "RMS cross-track" in format_table([])
    assert STEPS == 200
