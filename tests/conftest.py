"""Shared pytest fixtures and path setup.

Adds the project root to ``sys.path`` so tests can ``import src.<module>``
without requiring an editable install, and exposes audio fixtures used
across the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def synthetic_wav_factory(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory that writes small 16 kHz sine-wave WAVs to ``tmp_path``.

    Each generated file is a mono PCM_16 WAV. Defaults produce a 0.25 s
    clip (~8 KB) so a directory of dozens stays well under the 100 KB
    fixture budget specified in the testing-standards skill.

    Example:
        >>> wav_path = synthetic_wav_factory("clip.wav", seconds=1.0, freq=220)
    """

    def _make(
        name: str = "clip.wav",
        sample_rate: int = 16_000,
        seconds: float = 0.25,
        freq: float = 440.0,
        amplitude: float = 0.1,
    ) -> Path:
        n_samples = int(round(sample_rate * seconds))
        t = np.linspace(0.0, seconds, n_samples, endpoint=False, dtype=np.float32)
        waveform = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), waveform, sample_rate, subtype="PCM_16")
        return path

    return _make
