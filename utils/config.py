"""
utils/config.py
===============
Thin helpers for loading model configuration from YAML files.

Every model's ``from_config()`` class method uses ``load_config()`` to read
its YAML file, making it trivial to swap hyperparameters without touching code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Default directory where config files live (relative to project root).
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(
    config_path: str | Path,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load a YAML configuration file and return it as a plain dictionary.

    Parameters
    ----------
    config_path : str or Path
        Name or full path of the YAML file.  If only a filename is given
        (e.g. ``"dkl.yaml"``), it is resolved relative to *config_dir*.
    config_dir : str or Path, optional
        Directory to search for *config_path*.  Defaults to ``./configs/``.

    Returns
    -------
    dict
        Parsed YAML content.
    """
    config_path = Path(config_path)

    # If the caller passed just a filename, resolve it inside the config dir.
    if not config_path.is_absolute() and not config_path.exists():
        base = Path(config_dir) if config_dir is not None else _DEFAULT_CONFIG_DIR
        config_path = base / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected a YAML mapping at the top level, got {type(cfg)}")

    return cfg
