"""Tests for ``scripts/download_tarteel.py``.

Synthetic-only: never touches the network or the real HuggingFace
dataset. Audio fixtures come from ``synthetic_wav_factory`` in
``conftest.py``; the dataset iterator is replaced with an in-memory
generator wherever ``stream_and_save`` is exercised.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "download_tarteel.py"

_spec = importlib.util.spec_from_file_location("download_tarteel", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
download_tarteel = importlib.util.module_from_spec(_spec)
sys.modules["download_tarteel"] = download_tarteel
_spec.loader.exec_module(download_tarteel)


# --- parse_surah_ayah --------------------------------------------------------


def test_parse_surah_ayah_legacy_format() -> None:
    """Legacy Everyayah filenames decode to (surah, ayah)."""
    assert download_tarteel.parse_surah_ayah("AbdulSamad/001005.mp3") == (1, 5)
    assert download_tarteel.parse_surah_ayah("114006.wav") == (114, 6)


def test_parse_surah_ayah_unparseable() -> None:
    """Anything that doesn't start with two 3-digit groups returns nulls."""
    assert download_tarteel.parse_surah_ayah("weird_name.wav") == (None, None)
    assert download_tarteel.parse_surah_ayah("") == (None, None)
    assert download_tarteel.parse_surah_ayah("12345.mp3") == (None, None)


def test_parse_surah_ayah_out_of_range() -> None:
    """Digits that don't map to a real surah/ayah are rejected."""
    assert download_tarteel.parse_surah_ayah("999999.mp3") == (None, None)
    assert download_tarteel.parse_surah_ayah("000001.mp3") == (None, None)
    assert download_tarteel.parse_surah_ayah("001000.mp3") == (None, None)


# --- safe_reciter_name -------------------------------------------------------


def test_safe_reciter_name() -> None:
    """ASCII names lose punctuation and spaces; output is lowercase."""
    assert download_tarteel.safe_reciter_name("Mishary Rashid Alafasy") == "mishary_rashid_alafasy"
    assert download_tarteel.safe_reciter_name("AbdulSamad") == "abdulsamad"
    assert download_tarteel.safe_reciter_name("abdul-rahman al-sudais") == "abdul_rahman_al_sudais"


def test_safe_reciter_name_idempotent() -> None:
    """Running the function on its output yields the same string."""
    once = download_tarteel.safe_reciter_name("Mishary Rashid Alafasy")
    twice = download_tarteel.safe_reciter_name(once)
    assert once == twice


def test_safe_reciter_name_handles_arabic_and_empty() -> None:
    """Arabic-only / empty inputs fall back to ``"unknown"``."""
    assert download_tarteel.safe_reciter_name("") == "unknown"
    assert download_tarteel.safe_reciter_name("إسلام صبحي") == "unknown"


# --- write_manifest_csv ------------------------------------------------------


def _row(**overrides: object) -> dict[str, object]:
    """Build a manifest row with sensible defaults; override any field."""
    base: dict[str, object] = {
        "clip_id": "tarteel_abdulsamad_001_005",
        "path": "data/raw/tarteel/abdulsamad/001005.wav",
        "reciter": "AbdulSamad",
        "surah": 1,
        "ayah": 5,
        "duration_s": 6.5,
        "sample_rate": 16_000,
    }
    base.update(overrides)
    return base


def test_write_manifest_csv_columns(tmp_path: Path) -> None:
    """The CSV header matches MANIFEST_COLUMNS exactly, in order."""
    manifest = tmp_path / "manifest.csv"
    download_tarteel.write_manifest_csv([_row()], manifest)
    df = pd.read_csv(manifest)
    assert tuple(df.columns) == download_tarteel.MANIFEST_COLUMNS


def test_write_manifest_csv_dtypes(tmp_path: Path) -> None:
    """surah/ayah become Int64 (nullable); rest are str/float/int."""
    manifest = tmp_path / "manifest.csv"
    download_tarteel.write_manifest_csv(
        [
            _row(),
            _row(clip_id="tarteel_x_unparsed_abc", path="x/y.wav", surah=None, ayah=None),
        ],
        manifest,
    )
    df = pd.read_csv(manifest, dtype={"surah": "Int64", "ayah": "Int64"})
    assert df["clip_id"].dtype == object
    assert df["path"].dtype == object
    assert df["reciter"].dtype == object
    assert str(df["surah"].dtype) == "Int64"
    assert str(df["ayah"].dtype) == "Int64"
    assert df["duration_s"].dtype == float
    assert df["sample_rate"].dtype == np.int64
    assert pd.isna(df.loc[1, "surah"])


def test_write_manifest_preserves_original_reciter(tmp_path: Path) -> None:
    """The ``reciter`` column keeps its original casing/spaces verbatim."""
    manifest = tmp_path / "manifest.csv"
    download_tarteel.write_manifest_csv(
        [_row(reciter="Mishary Rashid Alafasy")],
        manifest,
    )
    df = pd.read_csv(manifest)
    assert df.iloc[0]["reciter"] == "Mishary Rashid Alafasy"


def test_write_manifest_rejects_missing_columns(tmp_path: Path) -> None:
    """A row missing a required key is reported, not silently coerced."""
    manifest = tmp_path / "manifest.csv"
    bad = _row()
    del bad["sample_rate"]
    with pytest.raises(ValueError, match="missing columns"):
        download_tarteel.write_manifest_csv([bad], manifest)


def test_write_manifest_rejects_empty(tmp_path: Path) -> None:
    """An empty manifest is rejected; AC requires ≥ 1500 rows downstream."""
    with pytest.raises(ValueError, match="empty"):
        download_tarteel.write_manifest_csv([], tmp_path / "manifest.csv")


# --- verify_random_clips -----------------------------------------------------


def _build_manifest_for_fixtures(paths: list[Path], manifest_path: Path) -> None:
    rows = [_row(path=str(p), clip_id=f"tarteel_test_{i:03d}") for i, p in enumerate(paths)]
    download_tarteel.write_manifest_csv(rows, manifest_path)


def test_verify_random_clips_passes_on_clean_wavs(
    tmp_path: Path, synthetic_wav_factory: object
) -> None:
    """A manifest of clean sine-wave WAVs verifies cleanly."""
    paths = [synthetic_wav_factory(f"clip_{i:02d}.wav") for i in range(12)]  # type: ignore[operator]
    manifest = tmp_path / "manifest.csv"
    _build_manifest_for_fixtures(paths, manifest)
    n_ok = download_tarteel.verify_random_clips(manifest, n=10, seed=42)
    assert n_ok == 10


def test_verify_random_clips_raises_on_corrupt(
    tmp_path: Path, synthetic_wav_factory: object
) -> None:
    """A non-decodable file in the sample raises AssertionError."""
    good_paths = [synthetic_wav_factory(f"clip_{i:02d}.wav") for i in range(2)]  # type: ignore[operator]
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"this is not a wav file at all")
    paths = good_paths + [bad]
    manifest = tmp_path / "manifest.csv"
    _build_manifest_for_fixtures(paths, manifest)
    with pytest.raises(AssertionError):
        download_tarteel.verify_random_clips(manifest, n=3, seed=0)


def test_verify_random_clips_detects_sample_rate_mismatch(
    tmp_path: Path, synthetic_wav_factory: object
) -> None:
    """If the manifest's sample_rate disagrees with the WAV, we fail loudly."""
    # WAV is 16 kHz but manifest claims 8 kHz.
    wav = synthetic_wav_factory("clip.wav")  # type: ignore[operator]
    manifest = tmp_path / "manifest.csv"
    download_tarteel.write_manifest_csv([_row(path=str(wav), sample_rate=8_000)], manifest)
    with pytest.raises(AssertionError, match="sample-rate mismatch"):
        download_tarteel.verify_random_clips(manifest, n=1, seed=0)


# --- stream_and_save ---------------------------------------------------------


def _fake_examples(reciters: list[str], per_reciter: int) -> Iterator[dict[str, object]]:
    """Yield deterministic fake HF examples (no network, no audio decoding).

    Examples are emitted round-robin across reciters so the iterator
    mimics the per-reciter spread that ``ds.shuffle()`` produces in
    production. A naive sequential order would let the first reciter
    saturate ``per_reciter_cap`` before any other reciter is seen,
    which is the opposite of what the production code assumes.
    """
    sr = download_tarteel.EXPECTED_SAMPLE_RATE
    # 0.05 s of near-silence is enough for soundfile to write a valid PCM_16 WAV.
    array = np.zeros(int(sr * 0.05), dtype=np.float32) + 1e-3
    for ayah in range(1, per_reciter + 1):
        for reciter in reciters:
            yield {
                "audio": {
                    "path": f"{reciter}/001{ayah:03d}.mp3",
                    "array": array,
                    "sampling_rate": sr,
                },
                "reciter": reciter,
                "duration": 0.05,
            }


def test_stream_and_save_stops_at_target(tmp_path: Path) -> None:
    """Stops at ``target_clips`` and respects ``per_reciter_cap``."""
    reciters = [f"reciter_{i:02d}" for i in range(8)]
    examples = _fake_examples(reciters, per_reciter=20)

    rows = download_tarteel.stream_and_save(
        target_clips=50,
        output_dir=tmp_path / "tarteel",
        seed=42,
        per_reciter_cap=10,
        dataset_iter=examples,
    )

    assert len(rows) == 50
    by_reciter = pd.Series([r["reciter"] for r in rows]).value_counts()
    assert (by_reciter <= 10).all()
    assert by_reciter.size >= download_tarteel.MIN_DISTINCT_RECITERS


def test_stream_and_save_writes_audio_to_disk(tmp_path: Path) -> None:
    """Each accepted example produces a WAV under ``<output>/<safe_reciter>/``."""
    reciters = ["AbdulSamad", "Mishary Rashid Alafasy"] + [f"r_{i}" for i in range(6)]
    examples = _fake_examples(reciters, per_reciter=4)
    rows = download_tarteel.stream_and_save(
        target_clips=24,
        output_dir=tmp_path / "tarteel",
        seed=7,
        per_reciter_cap=4,
        dataset_iter=examples,
    )
    for row in rows:
        assert Path(row["path"]).exists(), row
    abdulsamad_dir = tmp_path / "tarteel" / "abdulsamad"
    mishary_dir = tmp_path / "tarteel" / "mishary_rashid_alafasy"
    assert abdulsamad_dir.is_dir()
    assert mishary_dir.is_dir()


def test_stream_and_save_raises_when_too_few_reciters(tmp_path: Path) -> None:
    """Stream of <8 reciters fails the diversity check loudly."""
    reciters = [f"only_{i}" for i in range(3)]
    examples = _fake_examples(reciters, per_reciter=20)
    with pytest.raises(RuntimeError, match="distinct reciters"):
        download_tarteel.stream_and_save(
            target_clips=30,
            output_dir=tmp_path / "tarteel",
            seed=0,
            per_reciter_cap=20,
            dataset_iter=examples,
        )


def test_stream_and_save_raises_when_stream_exhausted(tmp_path: Path) -> None:
    """If the iterator runs out before target, we surface a RuntimeError."""
    reciters = [f"r_{i}" for i in range(8)]
    examples = _fake_examples(reciters, per_reciter=2)  # only 16 total
    with pytest.raises(RuntimeError, match="stream exhausted"):
        download_tarteel.stream_and_save(
            target_clips=100,
            output_dir=tmp_path / "tarteel",
            seed=0,
            per_reciter_cap=20,
            dataset_iter=examples,
        )
