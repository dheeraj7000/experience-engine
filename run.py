#!/usr/bin/env python3
"""Experiment entry point.

    python run.py --family toy_bug --configs a0 a1 a2 --seeds 1 2 3
    python run.py --provider config          # use config/models.yaml (real models)
    python run.py --provider dry_run         # no model needed (plumbing demo)

With dry_run every episode fails (the stub model does nothing), so success
curves are flat — but the FULL pipeline still runs: A2 clusters failures,
diagnoses, induces experiences, validates on held-out, and promotes policies.
Real learning curves require a real online model (e.g. Ollama).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from providers import DryRunProvider, load_roles
from harness import Runner, RunConfig, Reporter
from families import get_family


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an Experience Engine experiment.")
    ap.add_argument("--family", default="toy_bug")
    ap.add_argument("--configs", nargs="+", default=["a0", "a1", "a2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--provider", choices=["dry_run", "config"], default="dry_run")
    ap.add_argument("--models", default="config/models.yaml")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args()

    if args.provider == "config":
        roles = load_roles(args.models)
        online = roles.get("online") or DryRunProvider()
        offline = roles.get("offline")
    else:
        online, offline = DryRunProvider(), DryRunProvider()

    family = get_family(args.family)
    runner = Runner(online=online, offline=offline)
    cfg = RunConfig(
        configs=args.configs, seeds=args.seeds,
        checkpoint_every=args.checkpoint_every, max_steps=args.max_steps,
        out_dir=args.out_dir,
    )
    records = runner.run(family, cfg)

    reporter = Reporter(records)
    report_path = Path(args.out_dir) / family.family_id / "report.json"
    metrics = reporter.write(report_path)
    print(reporter.ascii_curves())
    print(f"\nReport: {report_path}")
    for cfg_name in args.configs:
        if cfg_name in metrics:
            m = metrics[cfg_name]
            print(f"  {cfg_name}: mean_success={m['mean_success']:.3f} "
                  f"slope={m['learning_slope']:+.4f} "
                  f"repeated_failures={m['repeated_failure_rate']:.2f} "
                  f"mean_tokens={m['mean_tokens']:.0f}")


if __name__ == "__main__":
    main()
