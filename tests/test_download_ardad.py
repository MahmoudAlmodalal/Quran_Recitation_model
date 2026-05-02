"""Tests for ``scripts/download_ardad.py``.

Synthetic-only: never touches the network or the real Kaggle API. Audio
fixtures come from ``synthetic_wav_factory`` in ``conftest.py``; the
extract root is a ``tmp_path`` sub-tree wherever ``build_manifest_rows``
or ``iter_audio_files`` is exercised. The ``kaggle`` SDK is never
imported because ``download_archive`` lazy-imports it inside the
function body.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "download_ardad.py"

_spec = importlib.util.spec_from_file_location("download_ardad", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
download_ardad = importlib.util.module_from_spec(_spec)
sys.modules["download_ardad"] = download_ardad
_spec.loader.exec_module(download_ardad)


# --- parse_ardad_filename ----------------------------------------------------


def test_parse_ardad_filename_canonical() -> None:
    """A standard ``<reciter>/<NNN>.<ext>`` path decodes both fields."""
    assert download_ardad.parse_ardad_filename(Path("Alafasy/001.mp3")) == ("Alafasy", 1)
    assert download_ardad.parse_ardad_filename(Path("AbdulBasit/114.wav")) == ("AbdulBasit", 114)


def test_parse_ardad_filename_with_leading_zeros() -> None:
    """Leading-zero stems decode correctly to ``int``."""
    assert download_ardad.parse_ardad_filename(Path("Husary/036.mp3")) == ("Husary", 36)
    assert download_ardad.parse_ardad_filename(Path("Sudais/078.flac")) == ("Sudais", 78)


def test_parse_ardad_filename_no_digits_yields_none_surah() -> None:
    """Stems without digits keep the reciter but ``surah`` is ``None``."""
    assert download_ardad.parse_ardad_filename(Path("R/notes.mp3")) == ("R", None)
    assert download_ardad.parse_ardad_filename(Path("R/track.wav")) == ("R", None)


def test_parse_ardad_filename_out_of_range_yields_none_surah() -> None:
    """Numbers outside the 1..114 surah range are rejected."""
    assert download_ardad.parse_ardad_filename(Path("R/200.mp3")) == ("R", None)
    assert download_ardad.parse_ardad_filename(Path("R/000.mp3")) == ("R", None)
    assert download_ardad.parse_ardad_filename(Path("R/115.mp3")) == ("R", None)


def test_parse_ardad_filename_preserves_reciter_verbatim() -> None:
    """Reciter equals the immediate parent dir name, including spaces."""
    rec, surah = download_ardad.parse_ardad_filename(Path("Mishary Rashid/001.mp3"))
    assert rec == "Mishary Rashid"
    assert surah == 1


# --- safe_reciter_name -------------------------------------------------------


def test_safe_reciter_name_ascii() -> None:
    """ASCII names lose punctuation and spaces; output is lowercase."""
    assert download_ardad.safe_reciter_name("Mishary Rashid Alafasy") == "mishary_rashid_alafasy"
    assert download_ardad.safe_reciter_name("AbdulSamad") == "abdulsamad"
    assert download_ardad.safe_reciter_name("abdul-rahman al-sudais") == "abdul_rahman_al_sudais"


def test_safe_reciter_name_idempotent() -> None:
    """Running the function on its output yields the same string."""
    once = download_ardad.safe_reciter_name("Mishary Rashid Alafasy")
    twice = download_ardad.safe_reciter_name(once)
    assert once == twice


def test_safe_reciter_name_handles_arabic_and_empty() -> None:
    """Arabic-only / empty inputs fall back to ``"unknown"``."""
    assert download_ardad.safe_reciter_name("") == "unknown"
    assert download_ardad.safe_reciter_name("إسلام صبحي") == "unknown"


# --- write_manifest_csv ------------------------------------------------------


def _row(**overrides: object) -> dict[str, object]:
    """Build a manifest row with sensible Ar-DAD defaults; override any field."""
    base: dict[str, object] = {
        "clip_id": "ardad_alafasy_001",
        "path": "data/raw/ardad/Alafasy/001.mp3",
        "reciter": "Alafasy",
        "surah": 1,
        "ayah": None,
        "duration_s": 213.4,
        "sample_rate": 22_050,
    }
    base.update(overrides)
    return base


def test_write_manifest_csv_columns(tmp_path: Path) -> None:
    """The CSV header matches ``MANIFEST_COLUMNS`` exactly, in order."""
    manifest = tmp_path / "manifest.csv"
    download_ardad.write_manifest_csv([_row()], manifest)
    df = pd.read_csv(manifest)
    assert tuple(df.columns) == download_ardad.MANIFEST_COLUMNS


def test_write_manifest_csv_dtypes_with_null_ayah(tmp_path: Path) -> None:
    """``ayah`` is always nullable Int64 with NaN; other dtypes are stable."""
    manifest = tmp_path / "manifest.csv"
    download_ardad.write_manifest_csv(
        [
            _row(),
            _row(clip_id="ardad_husary_036", surah=36, reciter="Husary", path="x/036.mp3"),
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
    assert pd.isna(df["ayah"]).all()


def test_write_manifest_preserves_original_reciter(tmp_path: Path) -> None:
    """The ``reciter`` column keeps its original casing/spaces verbatim."""
    manifest = tmp_path / "manifest.csv"
    download_ardad.write_manifest_csv(
        [_row(reciter="Mishary Rashid Alafasy")],
        manifest,
    )
    df = pd.read_csv(manifest)
    assert df.iloc[0]["reciter"] == "Mishary Rashid Alafasy"


def test_write_manifest_rejects_missing_columns(tmp_path: Path) -> None:
    """A row missing a required key is reported, not silently coerced."""
    bad = _row()
    del bad["sample_rate"]
    with pytest.raises(ValueError, match="missing columns"):
        download_ardad.write_manifest_csv([bad], tmp_path / "manifest.csv")


def test_write_manifest_rejects_empty(tmp_path: Path) -> None:
    """An empty manifest is rejected loudly."""
    with pytest.raises(ValueError, match="empty"):
        download_ardad.write_manifest_csv([], tmp_path / "manifest.csv")


# --- verify_random_clips -----------------------------------------------------


def _build_manifest_for_fixtures(paths: list[Path], manifest_path: Path) -> None:
    """Write a manifest pointing at ``paths`` (one row each, surah=None)."""
    rows = [
        _row(path=str(p), clip_id=f"ardad_test_{i:03d}", surah=None, sample_rate=16_000)
        for i, p in enumerate(paths)
    ]
    download_ardad.write_manifest_csv(rows, manifest_path)


def test_verify_random_clips_passes_on_clean_wavs(
    tmp_path: Path, synthetic_wav_factory: Callable[..., Path]
) -> None:
    """A manifest of clean sine-wave WAVs verifies cleanly."""
    paths = [synthetic_wav_factory(f"clip_{i:02d}.wav") for i in range(12)]
    manifest = tmp_path / "manifest.csv"
    _build_manifest_for_fixtures(paths, manifest)
    n_ok = download_ardad.verify_random_clips(manifest, n=10, seed=42)
    assert n_ok == 10


def test_verify_random_clips_raises_on_corrupt(
    tmp_path: Path, synthetic_wav_factory: Callable[..., Path]
) -> None:
    """A non-decodable file in the sample raises ``AssertionError``."""
    good_paths = [synthetic_wav_factory(f"clip_{i:02d}.wav") for i in range(2)]
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"this is not a wav file at all")
    paths = good_paths + [bad]
    manifest = tmp_path / "manifest.csv"
    _build_manifest_for_fixtures(paths, manifest)
    with pytest.raises(AssertionError):
        download_ardad.verify_random_clips(manifest, n=3, seed=0)


def test_verify_random_clips_detects_sample_rate_mismatch(
    tmp_path: Path, synthetic_wav_factory: Callable[..., Path]
) -> None:
    """If the manifest's sample_rate disagrees with the WAV, fail loudly."""
    wav = synthetic_wav_factory("clip.wav")  # 16 kHz
    manifest = tmp_path / "manifest.csv"
    download_ardad.write_manifest_csv(
        [_row(path=str(wav), surah=None, sample_rate=8_000)], manifest
    )
    with pytest.raises(AssertionError, match="sample-rate mismatch"):
        download_ardad.verify_random_clips(manifest, n=1, seed=0)


# --- iter_audio_files --------------------------------------------------------


def test_iter_audio_files_filters_by_extension(tmp_path: Path) -> None:
    """Only audio extensions are returned; output is sorted; case-insensitive suffix."""
    (tmp_path / "Reciter").mkdir()
    audio_paths = [
        tmp_path / "Reciter" / "001.mp3",
        tmp_path / "Reciter" / "002.WAV",  # uppercase suffix
        tmp_path / "Reciter" / "003.flac",
    ]
    for p in audio_paths:
        p.write_bytes(b"")
    (tmp_path / "Reciter" / "notes.txt").write_bytes(b"")
    (tmp_path / "Reciter" / "image.png").write_bytes(b"")

    found = download_ardad.iter_audio_files(tmp_path)
    assert {p.name for p in found} == {"001.mp3", "002.WAV", "003.flac"}
    assert found == sorted(found)


def test_iter_audio_files_skips_reserved_and_hidden(tmp_path: Path) -> None:
    """``manifest.csv`` / ``ATTRIBUTION.md`` and dotfiles are excluded."""
    (tmp_path / "Reciter").mkdir()
    (tmp_path / "Reciter" / "001.mp3").write_bytes(b"")
    (tmp_path / "manifest.csv").write_bytes(b"")
    (tmp_path / "ATTRIBUTION.md").write_bytes(b"")
    (tmp_path / "Reciter" / ".DS_Store").write_bytes(b"")
    (tmp_path / "Reciter" / ".hidden.mp3").write_bytes(b"")

    found = download_ardad.iter_audio_files(tmp_path)
    assert [p.name for p in found] == ["001.mp3"]


def test_iter_audio_files_returns_empty_for_missing_or_file(tmp_path: Path) -> None:
    """Missing path or path pointing to a file returns ``[]``."""
    assert download_ardad.iter_audio_files(tmp_path / "does_not_exist") == []
    a_file = tmp_path / "x.mp3"
    a_file.write_bytes(b"")
    assert download_ardad.iter_audio_files(a_file) == []


# --- build_manifest_rows -----------------------------------------------------


def _make_extract_tree(
    factory: Callable[..., Path], reciters: list[str], surahs: list[int]
) -> Path:
    """Create ``<tmp>/extracted/<reciter>/<NNN>.wav`` for each combo; return root."""
    roots: set[Path] = set()
    for reciter in reciters:
        for surah in surahs:
            wav_path = factory(f"extracted/{reciter}/{surah:03d}.wav")
            roots.add(wav_path.parents[1])
    assert len(roots) == 1, "all WAVs should share a single extracted root"
    return roots.pop()


def test_build_manifest_rows_happy_path(
    synthetic_wav_factory: Callable[..., Path],
) -> None:
    """A small synthetic tree produces well-formed rows with the right columns."""
    reciters = ["Alafasy", "Mishary Rashid", "Husary"]
    surahs = [1, 36, 114]
    root = _make_extract_tree(synthetic_wav_factory, reciters, surahs)

    rows = download_ardad.build_manifest_rows(root, min_distinct_reciters=2)

    assert len(rows) == len(reciters) * len(surahs)
    assert all(r["ayah"] is None for r in rows)
    assert all(r["sample_rate"] == 16_000 for r in rows)
    for row in rows:
        safe = download_ardad.safe_reciter_name(str(row["reciter"]))
        assert str(row["clip_id"]).startswith(f"ardad_{safe}_")
    assert {row["reciter"] for row in rows} == set(reciters)
    counts = pd.Series([row["surah"] for row in rows]).value_counts()
    assert (counts == len(reciters)).all()


def test_build_manifest_rows_skips_orphan_files_at_root(
    synthetic_wav_factory: Callable[..., Path],
) -> None:
    """Files at the extract root with no reciter dir are skipped."""
    root = _make_extract_tree(synthetic_wav_factory, ["A", "B"], [1, 2])
    orphan = synthetic_wav_factory("extracted/orphan.wav")
    assert orphan.parent == root

    rows = download_ardad.build_manifest_rows(root, min_distinct_reciters=2)
    paths = {Path(str(row["path"])).name for row in rows}
    assert "orphan.wav" not in paths
    assert len(rows) == 4


def test_build_manifest_rows_raises_when_too_few_reciters(
    synthetic_wav_factory: Callable[..., Path],
) -> None:
    """Fewer reciters than ``min_distinct_reciters`` fails loudly."""
    root = _make_extract_tree(synthetic_wav_factory, ["only_one"], [1, 2, 3])
    with pytest.raises(RuntimeError, match="distinct reciters"):
        download_ardad.build_manifest_rows(root, min_distinct_reciters=2)


def test_build_manifest_rows_raises_on_empty_tree(tmp_path: Path) -> None:
    """An extract tree with no audio files raises."""
    (tmp_path / "extracted").mkdir()
    with pytest.raises(RuntimeError, match="no decodable audio"):
        download_ardad.build_manifest_rows(tmp_path / "extracted", min_distinct_reciters=1)


# --- extract_archive ---------------------------------------------------------


def test_extract_archive_skips_when_audio_already_present(
    tmp_path: Path, synthetic_wav_factory: Callable[..., Path]
) -> None:
    """If dest already has audio, ``extract_archive`` is a no-op (bogus zip not opened)."""
    synthetic_wav_factory("extracted/Reciter/001.wav")
    dest = tmp_path / "extracted"
    bogus_zip = tmp_path / "fake.zip"
    bogus_zip.write_bytes(b"this is not a zip")
    result = download_ardad.extract_archive(bogus_zip, dest)
    assert result == dest
    assert (dest / "Reciter" / "001.wav").exists()


def test_extract_archive_unzips_into_dest(
    tmp_path: Path, synthetic_wav_factory: Callable[..., Path]
) -> None:
    """A real zip is unpacked into a previously empty dest."""
    src = synthetic_wav_factory("source/Reciter/001.wav")
    zip_path = tmp_path / "ardad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(src, arcname="Reciter/001.wav")

    dest = tmp_path / "extracted"
    download_ardad.extract_archive(zip_path, dest)
    assert (dest / "Reciter" / "001.wav").is_file()


# --- _validate_args ----------------------------------------------------------


def test_validate_args_requires_slug_or_archive() -> None:
    """``SystemExit`` when both ``--dataset-slug`` and ``--archive`` are missing."""
    args = argparse.Namespace(dataset_slug=None, archive=None)
    with pytest.raises(SystemExit, match="dataset-slug"):
        download_ardad._validate_args(args)


def test_validate_args_accepts_slug_only() -> None:
    """Slug present, archive missing → no error."""
    args = argparse.Namespace(dataset_slug="owner/ar-dad", archive=None)
    download_ardad._validate_args(args)  # must not raise


def test_validate_args_accepts_archive_only() -> None:
    """Archive present, slug missing → no error."""
    args = argparse.Namespace(dataset_slug=None, archive="/tmp/ardad.zip")
    download_ardad._validate_args(args)  # must not raise


# --- _resolve_output_dir -----------------------------------------------------


def test_resolve_output_dir_uses_cli_when_provided() -> None:
    """An explicit ``--output-dir`` overrides the YAML config path."""
    cfg = {"paths": {"raw_dir": "data/raw"}}
    assert download_ardad._resolve_output_dir("/custom/path", cfg) == Path("/custom/path")


def test_resolve_output_dir_falls_back_to_config() -> None:
    """With no CLI override, the path is ``<raw_dir>/ardad``."""
    cfg = {"paths": {"raw_dir": "data/raw"}}
    assert download_ardad._resolve_output_dir(None, cfg) == Path("data/raw/ardad")
