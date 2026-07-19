"""Build providers from config; resolve model roles from models.yaml."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .base import ModelProvider
from .dry_run import DryRunProvider
from .openai_compat import OpenAICompatProvider


def _expand(value: Any) -> Any:
    """Expand ${ENV_VAR} in strings so secrets stay out of the yaml file."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def build_provider(cfg: dict) -> ModelProvider:
    kind = cfg.get("provider", "dry_run")
    if kind == "dry_run":
        return DryRunProvider()
    if kind == "openai_compat":
        return OpenAICompatProvider(
            base_url=_expand(cfg["base_url"]),
            model=_expand(cfg["model"]),
            api_key=_expand(cfg.get("api_key", "")),
            rpm_limit=cfg.get("rpm_limit"),
        )
    raise ValueError(f"unknown provider: {kind!r}")


def load_roles(path: str | Path) -> dict[str, ModelProvider]:
    """Return {'online': provider, 'offline': provider} from a models.yaml."""
    data = yaml.safe_load(Path(path).read_text())
    roles = data.get("roles", {})
    return {name: build_provider(rcfg) for name, rcfg in roles.items()}
