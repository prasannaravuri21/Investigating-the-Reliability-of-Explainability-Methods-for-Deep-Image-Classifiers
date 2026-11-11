"""Load configs/default.yaml and let scripts override single keys."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .utils import PROJECT_ROOT, set_gpu_duty_cycle


class Config(dict):
    """Dict with dotted access: cfg["road"]["ks"] or cfg.get_path("road.ks")."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        keys = dotted.split(".")
        node: Any = self
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

    def resolve(self, dotted: str) -> Path:
        """Resolve a configured relative path against the project root."""
        value = self.get_path(dotted)
        if value is None:
            raise KeyError(dotted)
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Read the YAML config, then apply `key.subkey=value` overrides."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = Config(yaml.safe_load(fh))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.subkey=value, got {item!r}")
        key, raw = item.split("=", 1)
        cfg.set_path(key.strip(), yaml.safe_load(raw))
    # Every script loads the config first, so the GPU cap applies everywhere.
    set_gpu_duty_cycle(cfg.get_path("gpu_max_utilisation", 1.0))
    return cfg
