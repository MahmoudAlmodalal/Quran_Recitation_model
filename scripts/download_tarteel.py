"""Download a representative subset of the Tarteel/Everyayah dataset.

Streams ``tarteel-ai/everyayah`` from HuggingFace, saves a stratified
per-reciter sample to ``data/raw/tarteel/``, and writes an indexed
manifest CSV with columns
``clip_id, path, reciter, surah, ayah, duration_s, sample_rate``.

This script implements QRE-6. Crowdsourced/learner audio (gated
``tarteel-ai/tlog`` or the Tarteel public submission archive) is
**not** ingested here — that is deferred to a follow-up task.

Run from the project root::

    python scripts/download_tarteel.py --target-clips 3000 --seed 42

Streaming avoids materialising the full ~117 GB dataset, but parquet
shards are still cached under ``~/.cache/huggingface/`` as they are
read.

The script is idempotent: a re-run reuses any audio already present on
disk and recomputes the manifest from the current state. If the
manifest already has at least ``--target-clips`` rows, the script
short-circuits to the verification step.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402  (sys.path tweak above)

log = logging.getLogger("download_tarteel")

DATASET_ID = "tarteel-ai/everyayah"
DATASET_LICENSE = "CC BY 4.0"
DATASET_URL = "https://huggingface.co/datasets/tarteel-ai/everyayah"

MANIFEST_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "path",
    "reciter",
    "surah",
    "ayah",
    "duration_s",
    "sample_rate",
)

DEFAULT_TARGET_CLIPS = 3000
DEFAULT_PER_RECITER_CAP = 350
DEFAULT_SEED = 42
DEFAULT_VERIFY_N = 10
MIN_DISTINCT_RECITERS = 8

EXPECTED_SAMPLE_RATE = 16_000

_FILENAME_DIGITS = re.compile(r"^(\d{3})(\d{3})")


def parse_surah_ayah(audio_path: str) -> tuple[int | None, int | None]:
    """Extract (surah, ayah) from an Everyayah-style filename.

    The legacy Everyayah archive names files ``<reciter>/NNNNNN.mp3``
    where the first three digits encode the surah (1–114) and the next
    three encode the ayah (1–286). The HF repackage may preserve this
    layout in ``audio.path``; if not, the parser returns ``(None, None)``
    and the caller logs a warning.

    Args:
        audio_path: The ``audio.path`` field from a streamed example.

    Returns:
        Tuple of (surah, ayah) on a successful parse, otherwise
        ``(None, None)``.
    """
    if not audio_path:
        return (None, None)
    stem = Path(audio_path).stem
    match = _FILENAME_DIGITS.match(stem)
    if not match:
        return (None, None)
    surah = int(match.group(1))
    ayah = int(match.group(2))
    if not (1 <= surah <= 114) or not (1 <= ayah <= 286):
        return (None, None)
    return (surah, ayah)


def safe_reciter_name(name: str) -> str:
    """Convert a reciter name into a filesystem-safe identifier.

    Decomposes Unicode accents, strips non-ASCII (including Arabic
    diacritics), collapses any run of non-alphanumeric characters into
    a single underscore, and lowercases the result. The original name
    is preserved verbatim in the manifest's ``reciter`` column; this
    function is used only for directory paths and the synthetic
    ``clip_id``.

    Args:
        name: The original reciter name from the dataset.

    Returns:
        A lowercase ASCII identifier with underscores; ``"unknown"`` if
        the input has no representable ASCII characters.
    """
    if not name:
        return "unknown"
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", errors="ignore").decode("ascii")
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_only).strip("_").lower()
    return safe or "unknown"


def write_manifest_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a manifest CSV with the canonical column order.

    Surah and ayah are stored as pandas ``Int64`` so missing values are
    serialized as empty strings (not ``"nan"``). All other columns are
    written without coercion.

    Args:
        rows: Mapping per clip with the seven manifest keys.
        output_path: Destination CSV path; parent directory is created.

    Raises:
        ValueError: If any row is missing a required column.
    """
    if not rows:
        raise ValueError("cannot write an empty manifest")
    for row in rows:
        missing = [col for col in MANIFEST_COLUMNS if col not in row]
        if missing:
            raise ValueError(f"manifest row missing columns {missing}: {row!r}")

    df = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    df["surah"] = df["surah"].astype("Int64")
    df["ayah"] = df["ayah"].astype("Int64")
    df["sample_rate"] = df["sample_rate"].astype(int)
    df["duration_s"] = df["duration_s"].astype(float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("wrote manifest with %d rows to %s", len(df), output_path)


def verify_random_clips(manifest_path: Path, n: int = 10, seed: int = 42) -> int:
    """Decode a random sample of manifest entries to confirm integrity.

    Loads ``n`` rows sampled deterministically from the manifest and
    decodes each WAV with ``librosa.load(sr=None)``. Asserts the
    waveform is non-empty and the sample-rate matches the manifest
    column.

    Args:
        manifest_path: Path to the manifest CSV.
        n: Number of clips to verify (clipped to manifest size).
        seed: RNG seed for reproducible sampling.

    Returns:
        The count of clips that decoded cleanly.

    Raises:
        AssertionError: If any clip fails to decode or has a sample-rate
            mismatch.
    """
    df = pd.read_csv(manifest_path)
    if df.empty:
        raise AssertionError(f"manifest at {manifest_path} is empty")

    sample_size = min(n, len(df))
    rng = random.Random(seed)
    indices = rng.sample(range(len(df)), sample_size)

    n_ok = 0
    for idx in indices:
        row = df.iloc[idx]
        path = Path(row["path"])
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        try:
            wav, sr = librosa.load(str(path), sr=None)
        except Exception as exc:
            raise AssertionError(f"librosa failed to decode {path}: {exc}") from exc
        if wav.size == 0:
            raise AssertionError(f"empty waveform: {path}")
        expected_sr = int(row["sample_rate"])
        if int(sr) != expected_sr:
            raise AssertionError(
                f"sample-rate mismatch at {path}: got {sr}, expected {expected_sr}"
            )
        n_ok += 1

    log.info("verified %d/%d random clips decode cleanly with librosa", n_ok, sample_size)
    return n_ok


def _build_clip_filename(
    surah: int | None, ayah: int | None, audio_array: np.ndarray
) -> tuple[str, str]:
    """Pick a filesystem-safe filename and the matching id-suffix.

    For parseable rows we use the canonical ``NNNNNN.wav`` form. Rows
    whose path doesn't expose surah/ayah get a stable hash so reruns
    don't duplicate the same clip under a fresh random name.

    Returns:
        Tuple of (filename, id_suffix) where id_suffix is appended to
        ``tarteel_<safe_reciter>_`` to form the ``clip_id``.
    """
    if surah is not None and ayah is not None:
        stem = f"{surah:03d}{ayah:03d}"
        return (f"{stem}.wav", f"{surah:03d}_{ayah:03d}")
    sample_bytes = audio_array.astype(np.float32, copy=False).tobytes()[:4096]
    digest = hashlib.sha1(sample_bytes).hexdigest()[:12]
    return (f"unparsed_{digest}.wav", f"unparsed_{digest}")


def _to_repo_relative(path: Path) -> str:
    """Return ``path`` as a POSIX string relative to cwd when possible."""
    cwd = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        return resolved.as_posix()


def stream_and_save(
    target_clips: int,
    output_dir: Path,
    seed: int,
    per_reciter_cap: int = DEFAULT_PER_RECITER_CAP,
    dataset_iter: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Stream ``tarteel-ai/everyayah`` and save audio + manifest rows.

    The stream is shuffled (``buffer_size=10_000``) so reciter ordering
    in the underlying parquet shards does not gate the per-reciter cap.

    Args:
        target_clips: Stop once at least this many rows are collected.
        output_dir: Root directory for ``<reciter>/<filename>.wav``.
        seed: Shuffle seed.
        per_reciter_cap: Max clips kept per (safe) reciter name.
        dataset_iter: Optional pre-built iterator of examples (for
            tests). When ``None``, the script calls
            ``datasets.load_dataset(...)`` itself.

    Returns:
        List of manifest row dicts in collection order.

    Raises:
        RuntimeError: If the stream is exhausted before reaching the
            target, or fewer than ``MIN_DISTINCT_RECITERS`` reciters
            are represented.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_iter is None:
        from datasets import load_dataset  # Lazy import; tests don't need datasets.

        ds = load_dataset(DATASET_ID, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        dataset_iter = ds

    rows: list[dict[str, Any]] = []
    counter: Counter[str] = Counter()
    schema_logged = False
    started = time.monotonic()

    for example in dataset_iter:
        if not schema_logged:
            audio_keys = list(example.get("audio", {}).keys())
            log.info(
                "first example keys=%s audio_keys=%s audio.path=%r",
                list(example.keys()),
                audio_keys,
                example.get("audio", {}).get("path"),
            )
            schema_logged = True

        original_reciter = str(example.get("reciter") or "")
        safe = safe_reciter_name(original_reciter)
        if counter[safe] >= per_reciter_cap:
            continue

        audio = example.get("audio") or {}
        sample_rate = int(audio.get("sampling_rate") or 0)
        if sample_rate != EXPECTED_SAMPLE_RATE:
            log.warning(
                "skipping clip with unexpected sample_rate=%d (reciter=%r)",
                sample_rate,
                original_reciter,
            )
            continue

        wav_array = np.asarray(audio.get("array"))
        if wav_array.size == 0:
            log.warning("skipping clip with empty array (reciter=%r)", original_reciter)
            continue

        surah, ayah = parse_surah_ayah(str(audio.get("path") or ""))
        filename, id_suffix = _build_clip_filename(surah, ayah, wav_array)
        clip_path = output_dir / safe / filename
        clip_path.parent.mkdir(parents=True, exist_ok=True)

        if not clip_path.exists():
            sf.write(str(clip_path), wav_array, EXPECTED_SAMPLE_RATE, subtype="PCM_16")

        rows.append(
            {
                "clip_id": f"tarteel_{safe}_{id_suffix}",
                "path": _to_repo_relative(clip_path),
                "reciter": original_reciter,
                "surah": surah,
                "ayah": ayah,
                "duration_s": float(example.get("duration") or 0.0),
                "sample_rate": EXPECTED_SAMPLE_RATE,
            }
        )
        counter[safe] += 1

        if len(rows) % 200 == 0:
            elapsed = time.monotonic() - started
            log.info(
                "saved %d clips across %d reciters in %.1fs",
                len(rows),
                len(counter),
                elapsed,
            )

        if len(rows) >= target_clips:
            break

    if len(rows) < target_clips:
        raise RuntimeError(f"stream exhausted at {len(rows)} clips; target was {target_clips}")
    if len(counter) < MIN_DISTINCT_RECITERS:
        raise RuntimeError(f"only {len(counter)} distinct reciters; need ≥ {MIN_DISTINCT_RECITERS}")

    return rows


def write_attribution_file(output_dir: Path, target_clips: int, seed: int) -> None:
    """Write a CC BY 4.0 attribution stub if not already present."""
    path = output_dir / "ATTRIBUTION.md"
    if path.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(
        f"""# Tarteel/Everyayah subset attribution

Source dataset: `{DATASET_ID}` (HuggingFace)
URL: {DATASET_URL}
License: {DATASET_LICENSE}

Subset configuration:
- Generated by: scripts/download_tarteel.py
- Generated at: {timestamp}
- Target clips: {target_clips}
- Sampling seed: {seed}
- Audio format: 16 kHz mono WAV (PCM_16)

Use of this subset must preserve the upstream attribution and license terms.
""",
        encoding="utf-8",
    )
    log.info("wrote attribution file to %s", path)


def _resolve_output_dir(cli_value: str | None, cfg: dict[str, Any]) -> Path:
    """Return ``--output-dir`` if provided, else ``<raw_dir>/tarteel`` from cfg."""
    if cli_value:
        return Path(cli_value)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    return raw_dir / "tarteel"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. ``argv=None`` defaults to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-clips",
        type=int,
        default=DEFAULT_TARGET_CLIPS,
        help="Number of clips to save (default: %(default)s; AC requires ≥ 1500).",
    )
    parser.add_argument(
        "--per-reciter-cap",
        type=int,
        default=DEFAULT_PER_RECITER_CAP,
        help="Max clips kept per reciter (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override <raw_dir>/tarteel from configs/default.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Shuffle / verification seed (default: %(default)s).",
    )
    parser.add_argument(
        "--verify-n",
        type=int,
        default=DEFAULT_VERIFY_N,
        help="Random clips to decode-test post-download (default: %(default)s).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML config (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: load config, stream the dataset, write the manifest."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    output_dir = _resolve_output_dir(args.output_dir, cfg)
    manifest_path = output_dir / "manifest.csv"

    log.info(
        "starting tarteel/everyayah ingestion: target=%d cap=%d seed=%d output=%s",
        args.target_clips,
        args.per_reciter_cap,
        args.seed,
        output_dir,
    )

    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        if len(existing) >= args.target_clips:
            log.info(
                "manifest at %s already has %d clips (≥ target %d); skipping download",
                manifest_path,
                len(existing),
                args.target_clips,
            )
            verify_random_clips(manifest_path, n=args.verify_n, seed=args.seed)
            return

    rows = stream_and_save(
        target_clips=args.target_clips,
        output_dir=output_dir,
        seed=args.seed,
        per_reciter_cap=args.per_reciter_cap,
    )
    write_manifest_csv(rows, manifest_path)
    write_attribution_file(output_dir, args.target_clips, args.seed)
    verify_random_clips(manifest_path, n=args.verify_n, seed=args.seed)
    log.info("done: %d clips written to %s", len(rows), output_dir)


if __name__ == "__main__":
    main()
