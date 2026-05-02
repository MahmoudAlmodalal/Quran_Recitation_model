"""Tests for ``configs/default.yaml`` and :mod:`src.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import REQUIRED_SECTIONS, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def test_default_config_exists() -> None:
    """The default config file ships with the repository."""
    assert DEFAULT_CONFIG.is_file()


def test_default_config_loads_cleanly() -> None:
    """PyYAML parses the file without error and yields a mapping."""
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)


def test_loader_returns_required_sections() -> None:
    """All five top-level sections are present after loading."""
    cfg = load_config(DEFAULT_CONFIG)
    for section in REQUIRED_SECTIONS:
        assert section in cfg, f"missing section: {section}"


def test_audio_section_values() -> None:
    """Audio defaults match the model contract (16 kHz, 30 s)."""
    cfg = load_config(DEFAULT_CONFIG)
    audio = cfg["audio"]
    assert audio["sample_rate"] == 16_000
    assert float(audio["max_length_s"]) == 30.0
    assert audio["n_mels"] == 80


def test_model_section_values() -> None:
    """Model defaults match the frozen Whisper-base wrapper."""
    cfg = load_config(DEFAULT_CONFIG)
    model = cfg["model"]
    assert model["encoder_id"] == "tarteel-ai/whisper-base-ar-quran"
    assert model["embedding_dim"] == 512
    assert model["n_levels"] == 3
    assert 0.0 <= float(model["dropout"]) <= 1.0


def test_training_loss_lambdas() -> None:
    """All four loss-weight keys are present and non-negative."""
    cfg = load_config(DEFAULT_CONFIG)
    lambdas = cfg["training"]["loss_lambdas"]
    assert set(lambdas) == {"level", "pronunciation", "tajweed", "fluency"}
    assert all(float(v) >= 0.0 for v in lambdas.values())


def test_paths_are_relative() -> None:
    """Paths must be relative so the project is portable across machines."""
    cfg = load_config(DEFAULT_CONFIG)
    for key, value in cfg["paths"].items():
        assert not Path(value).is_absolute(), f"{key} must be relative: {value}"


def test_app_level_labels() -> None:
    """Level labels match the three-class contract."""
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg["app"]["level_labels"] == ["Beginner", "Intermediate", "Advanced"]


def test_loader_rejects_non_mapping(tmp_path: Path) -> None:
    """A YAML file whose root is a list is rejected with ``ValueError``."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_loader_rejects_missing_section(tmp_path: Path) -> None:
    """Missing a required top-level section raises ``ValueError``."""
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("audio: {}\nmodel: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(incomplete)


def test_loader_raises_on_missing_file(tmp_path: Path) -> None:
    """A non-existent path raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")
