"""Compare both training strategies on the same delayed simulated cars."""

from dataclasses import replace
from random import Random

from driftsim.ars import ArsConfig, train
from driftsim.car import NOMINAL, CarParams, randomized
from driftsim.experiment import evaluate, nominal_sampler, randomized_sampler


def heldout(rng: Random) -> CarParams:
    car = randomized(NOMINAL, spread=0.30, max_delay=0, rng=rng)
    return replace(car, delay_steps=5)


def main() -> None:
    print("seed,strategy,mean_rmse_m,departures,cars")
    for seed in range(5):
        for strategy, sampler in (("nominal", nominal_sampler), ("randomized", randomized_sampler)):
            policy, _ = train(sampler, ArsConfig(iterations=40), seed)
            score = evaluate(policy, heldout, cars=40)
            print(f"{seed},{strategy},{score.rmse:.3f},{round(score.failure_rate * 40)},40")


if __name__ == "__main__":
    main()
