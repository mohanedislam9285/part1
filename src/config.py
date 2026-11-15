"""Project paths used by the notebook.

Configuration loading and validation are implemented in
:mod:`pm25_project.config`; this module gives the notebook stable paths.
"""

from __future__ import annotations

from pathlib import Path

from pm25_project.config import load_config, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "project.yaml"

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

SEED = 42


def load_project_config(path: str | Path | None = None):
    """Load ``project.yaml``, validate it, and return ``(config, project_root)``."""
    cfg, root = load_config(path or CONFIG_PATH)
    validate_config(cfg)
    return cfg, root


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_PATH",
    "DATA_PROCESSED",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "SEED",
    "load_project_config",
]
