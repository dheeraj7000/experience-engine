#!/usr/bin/env python3
"""Run the Experience Engine against LifelongAgentBench.

Usage:
    # Quick test (5 variants, dry_run provider, no Docker needed):
    python -m benchmarks.lifelongagentbench.run_benchmark --env-type db_bench \
        --max-variants 5 --provider dry_run

    # Full benchmark with real model (requires Ollama + Docker):
    python -m benchmarks.lifelongagentbench.run_benchmark --env-type db_bench \
        --provider config --n-episodes 20 --checkpoint-every 5

    # All environments:
    python -m benchmarks.lifelongagentbench.run_benchmark --provider config
"""
from __future__ import annotations

import argparse
from pathlib import Path

from providers import DryRunProvider, load_roles
from harness import Runner, RunConfig, Reporter
from benchmarks.lifelongagentbench.adapter import (
    LifelongAgentBenchFamily, setup_docker_containers, teardown_docker_containers,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Experience Engine on LifelongAgentBench.")
    ap.add_argument("--env-type", choices=["db_bench", "os_interaction", "knowledge_graph"],
                    default=None, help="Filter to one environment type (default: all)")
    ap.add_argument("--configs", nargs="+", default=["a0", "a1", "a2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--provider", choices=["dry_run", "config"], default="dry_run")
    ap.add_argument("--models", default="config/models.yaml")
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--max-variants", type=int, default=None,
                    help="Limit number of variants loaded (for quick testing)")
    ap.add_argument("--cache-dir", default=".cache/lifelongagentbench")
    ap.add_argument("--out-dir", default="runs/lifelongagentbench")
    ap.add_argument("--setup-docker", action="store_true",
                    help="Start Docker containers before running")
    ap.add_argument("--teardown-docker", action="store_true",
                    help="Stop Docker containers after running")
    args = ap.parse_args()

    # Setup Docker if requested.
    if args.setup_docker:
        print("Setting up Docker containers...")
        status = setup_docker_containers()
        for name, ok in status.items():
            print(f"  {name}: {'✓' if ok else '✗'}")

    # Build provider.
    if args.provider == "config":
        roles = load_roles(args.models)
        online = roles.get("online") or DryRunProvider()
        offline = roles.get("offline")
    else:
        online, offline = DryRunProvider(), DryRunProvider()

    # Build family.
    family = LifelongAgentBenchFamily(
        env_type=args.env_type,
        cache_dir=args.cache_dir,
        max_variants=args.max_variants,
    )

    # Run.
    runner = Runner(online=online, offline=offline)
    cfg = RunConfig(
        configs=args.configs, seeds=args.seeds,
        checkpoint_every=args.checkpoint_every,
        max_steps=args.max_steps,
        n_episodes=args.n_episodes,
        out_dir=args.out_dir,
    )

    print(f"\nRunning LifelongAgentBench ({family.family_id})")
    print(f"  Configs: {args.configs}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Provider: {args.provider}")
    print(f"  Max variants: {args.max_variants or 'all'}")
    print()

    records = runner.run(family, cfg)

    # Report.
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

    # Teardown Docker if requested.
    if args.teardown_docker:
        print("\nTearing down Docker containers...")
        teardown_docker_containers()


if __name__ == "__main__":
    main()
