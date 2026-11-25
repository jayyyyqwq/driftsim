"""Command line: ``driftsim demo`` and ``driftsim run``."""

from __future__ import annotations

import argparse

from driftsim.ars import ArsConfig
from driftsim.experiment import Outcome, run


def format_table(outcomes: list[Outcome]) -> str:
    lines = [
        f"{'trained in':<18} {'nominal car':>14} {'unseen cars (never trained on)':>34}",
        f"{'':<18} {'error':>14} {'error':>16} {'left the track':>17}",
    ]
    for o in outcomes:
        lines.append(
            f"{o.trained_on:<18} {o.on_nominal.rmse:>12.2f} m {o.on_real.rmse:>14.2f} m "
            f"{o.on_real.failure_rate:>16.0%}"
        )
    lines.append("\nerror = RMS cross-track distance from the line; lower is better.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="driftsim", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run the experiment with default settings")
    r = sub.add_parser("run", help="run with your own settings")
    r.add_argument("--iterations", type=int, default=40)
    r.add_argument("--seeds", type=int, default=5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        outcomes = run()
    else:
        outcomes = run(args.seeds, ArsConfig(iterations=args.iterations))
    print(format_table(outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
