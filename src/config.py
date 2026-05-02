"""Configuration loading for the Qur'an Recitation Evaluation system.

All hyperparameters live in ``configs/*.yaml``; this module is the single
entry point for reading them. Training, inference, and Streamlit code should
call :func:`load_config` rather than hardcoding values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs") / "default.yaml"
REQUIRED_SECTIONS: tuple[str, ...] = ("audio", "model", "training", "paths", "app")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML config. Defaults to ``configs/default.yaml``
            relative to the current working directory.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        ValueError: If the top-level YAML document is not a mapping or if any
            required top-level section is missing.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must be a mapping, got {type(data).__name__}")

    missing = [section for section in REQUIRED_SECTIONS if section not in data]
    if missing:
        raise ValueError(f"Config at {config_path} is missing required sections: {missing}")

    return data
