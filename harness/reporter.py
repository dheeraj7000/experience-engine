"""Reporter — turns episode records into the money-plot metrics + a manifest.

Headline metric: the compounding curve (success vs accumulated episodes) for
A0/A1/A2. Also: learning-efficiency slope, repeated-failure rate, and token
overhead. Writes JSON and prints a compact ASCII curve for quick eyeballing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .runner import EpisodeRecord


class Reporter:
    def __init__(self, records: list[EpisodeRecord]) -> None:
        self.records = records

    def metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for config in sorted({r.config for r in self.records}):
            rs = [r for r in self.records if r.config == config]
            by_index = self._mean_by_index(rs)
            xs = sorted(by_index)
            ys = [by_index[i] for i in xs]
            out[config] = {
                "n_episodes": len(rs),
                "curve": [{"index": i, "success": by_index[i]} for i in xs],
                "learning_slope": self._slope(xs, ys),
                "final_success": ys[-1] if ys else 0.0,
                "mean_success": round(float(np.mean([r.success for r in rs])), 4) if rs else 0.0,
                "repeated_failure_rate": self._repeated_failure_rate(rs),
                "mean_tokens": round(float(np.mean([r.tokens for r in rs])), 1) if rs else 0.0,
            }
        out["overhead_vs_a1"] = self._overhead(out)
        return out

    def write(self, path: str | Path) -> dict[str, Any]:
        m = self.metrics()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(m, indent=2))
        return m

    def ascii_curves(self, width: int = 40) -> str:
        m = self.metrics()
        lines = ["Compounding curve — success vs episode index (mean over seeds)"]
        for config in sorted(k for k in m if k.startswith("a")):
            curve = m[config]["curve"]
            if not curve:
                continue
            bar = "".join("#" if p["success"] >= 0.5 else "." for p in curve)[:width]
            slope = m[config]["learning_slope"]
            lines.append(f"  {config}: {bar}  slope={slope:+.4f} final={m[config]['final_success']:.2f}")
        return "\n".join(lines)

    # ---- internals --------------------------------------------------------
    @staticmethod
    def _mean_by_index(rs: list[EpisodeRecord]) -> dict[int, float]:
        buckets: dict[int, list[float]] = {}
        for r in rs:
            buckets.setdefault(r.index, []).append(r.success)
        return {i: round(float(np.mean(v)), 4) for i, v in buckets.items()}

    @staticmethod
    def _slope(xs: list[int], ys: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        return round(float(np.polyfit(xs, ys, 1)[0]), 4)

    @staticmethod
    def _repeated_failure_rate(rs: list[EpisodeRecord]) -> float:
        seen: set[str] = set()
        repeated = 0
        failures = 0
        for r in sorted(rs, key=lambda x: (x.seed, x.index)):
            if r.success >= 1.0:
                continue
            failures += 1
            if r.signature in seen:
                repeated += 1
            seen.add(r.signature)
        return round(repeated / failures, 4) if failures else 0.0

    @staticmethod
    def _overhead(out: dict[str, Any]) -> dict[str, Any]:
        if "a1" not in out or "a2" not in out:
            return {}
        a1t = out["a1"]["mean_tokens"] or 1.0
        return {"a2_token_ratio_vs_a1": round(out["a2"]["mean_tokens"] / a1t, 3)}
