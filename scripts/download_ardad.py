"""Download and verify the Ar-DAD dataset (37 chapters × 30 reciters).

Ar-DAD (Arabic Diversified Audio Dataset) is hosted on Kaggle. This
script ingests the archive into ``data/raw/ardad/``, leaves audio files
in their original format, and writes an indexed manifest CSV with
columns
``clip_id, path, reciter, surah, ayah, duration_s, sample_rate``.

This implements QRE-7 (Epic 2 — Data Pipeline). Ar-DAD serves as the
*advanced anchor* in the level classifier: 30 professional reciters,
clean studio quality. Reciter names are preserved verbatim in the
``reciter`` column because they are the anchor for the reciter-grouped
train/val/test splits used downstream.

Run from the project root, either with the Kaggle API::

    python scripts/download_ardad.py --dataset-slug <user/slug>

…or with a pre-downloaded zip (no Kaggle creds required)::

    python scripts/download_ardad.py --archive /path/to/ardad.zip

Audio files are kept in their original format (typically MP3); the
preprocessing pipeline resamples to 16 kHz mono at training and
inference time. The script is idempotent: a re-run reuses any audio
already extracted on disk and recomputes the manifest from the current
state. If the manifest already has at least
``EXPECTED_CLIPS_LOWER_BOUND`` rows, the script short-circuits to the
verification step.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import random
import re
import sys
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import librosa
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402  (sys.path tweak above)

log = logging.getLogger("download_ardad")

DATASET_LICENSE = "Research / non-commercial — see Ar-DAD paper"

MANIFEST_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "path",
    "reciter",
    "surah",
    "ayah",
    "duration_s",
    "sample_rate",
)

DEFAULT_SEED = 42
DEFAULT_VERIFY_N = 10
MIN_DISTINCT_RECITERS = 25
EXPECTED_CLIPS_LOWER_BOUND = 1000
AUDIO_EXTENSIONS: tuple[str, ...] = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
RESERVED_FILES: frozenset[str] = frozenset({"manifest.csv", "ATTRIBUTION.md"})

_SURAH_DIGITS = re.compile(r"\d+")


def safe_reciter_name(name: str) -> str:
    """Convert a reciter name into a filesystem-safe identifier.

    Decomposes Unicode accents, strips non-ASCII (including Arabic
    diacritics), collapses any run of non-alphanumeric characters into
    a single underscore, and lowercases the result. The original name
    is preserved verbatim in the manifest's ``reciter`` column; this
    function is used only for the synthetic ``clip_id``.

    Args:
        name: The original reciter name.

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


def parse_ardad_filename(path: Path) -> tuple[str, int | None]:
    """Extract ``(reciter, surah)`` from an Ar-DAD-style path.

    The Ar-DAD Kaggle archive nests audio under
    ``<reciter_name>/<surah>.<ext>``. The reciter is taken from the
    immediate parent directory name (verbatim, including spaces and
    punctuation — the manifest stores the original). The surah is
    parsed from the first run of digits in the file stem and validated
    to ``1 ≤ surah ≤ 114``.

    Args:
        path: Path to an Ar-DAD audio file.

    Returns:
        Tuple of ``(reciter, surah)`` where ``surah`` is ``None`` when
        the stem has no digits or the parsed value is out of range.
    """
    reciter = path.parent.name
    match = _SURAH_DIGITS.search(path.stem)
    if not match:
        return (reciter, None)
    surah = int(match.group(0))
    if not (1 <= surah <= 114):
        return (reciter, None)
    return (reciter, surah)


def iter_audio_files(root: Path) -> list[Path]:
    """Recursively collect audio files under ``root``.

    Filters to files whose suffix is in :data:`AUDIO_EXTENSIONS`, and
    excludes hidden files plus the reserved ``manifest.csv`` and
    ``ATTRIBUTION.md`` artefacts produced by this script. Returns a
    sorted list for deterministic iteration order.

    Args:
        root: Directory to walk. Returns an empty list if not a dir.

    Returns:
        Sorted list of audio-file paths under ``root``.
    """
    if not root.is_dir():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.name in RESERVED_FILES:
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        files.append(path)
    return sorted(files)


def write_manifest_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a manifest CSV with the canonical column order.

    Surah and ayah are stored as pandas ``Int64`` so missing values are
    serialized as empty strings (not ``"nan"``). All other columns are
    written without coercion.

    Args:
        rows: Mapping per clip with the seven manifest keys.
        output_path: Destination CSV path; parent directory is created.

    Raises:
        ValueError: If ``rows`` is empty or any row is missing a
            required column.
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
    decodes each file with ``librosa.load(sr=None)``. Asserts the
    waveform is non-empty and the sample-rate matches the manifest
    column.

    Args:
        manifest_path: Path to the manifest CSV.
        n: Number of clips to verify (clipped to manifest size).
        seed: RNG seed for reproducible sampling.

    Returns:
        The count of clips that decoded cleanly.

    Raises:
        AssertionError: If the manifest is empty, or any sampled clip
            fails to decode or has a sample-rate mismatch.
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


def _to_repo_relative(path: Path) -> str:
    """Return ``path`` as a POSIX string relative to cwd when possible."""
    cwd = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        return resolved.as_posix()


def probe_audio_file(path: Path) -> tuple[int, float] | None:
    """Read ``(sample_rate, duration_s)`` from an audio file's headers.

    Tries ``soundfile.info`` first (fast, header-only; works for WAV,
    FLAC, OGG, and MP3 with libsndfile ≥ 1.1). Falls back to a
    1-second ``librosa.load`` plus ``librosa.get_duration`` for codecs
    that soundfile can't open (typically older MP3 / M4A toolchains).

    Args:
        path: Path to an audio file.

    Returns:
        Tuple of ``(sample_rate_hz, duration_s)``, or ``None`` if the
        file cannot be probed.
    """
    try:
        info = sf.info(str(path))
        return int(info.samplerate), float(info.duration)
    except Exception:
        pass
    try:
        wav, sr = librosa.load(str(path), sr=None, mono=True, duration=1.0)
        if wav.size == 0:
            return None
        duration = float(librosa.get_duration(path=str(path)))
        return int(sr), duration
    except Exception:
        return None


def build_manifest_rows(
    extract_root: Path,
    min_distinct_reciters: int = MIN_DISTINCT_RECITERS,
) -> list[dict[str, Any]]:
    """Walk ``extract_root`` and build manifest rows for each audio file.

    Each row's ``reciter`` is the immediate parent directory's verbatim
    name. ``surah`` is parsed from the file stem and validated. The
    ``sample_rate`` and ``duration_s`` come from
    :func:`probe_audio_file` (header-based; falls back to a partial
    decode). ``ayah`` is always ``None`` because Ar-DAD entries are
    chapter-level recordings.

    Files at the top level of ``extract_root`` (no reciter directory)
    and files that fail to probe are logged and skipped.

    Args:
        extract_root: Directory containing the extracted Ar-DAD layout.
        min_distinct_reciters: Diversity guard. Mirrors the Tarteel
            ingestion contract; defaults to
            :data:`MIN_DISTINCT_RECITERS`.

    Returns:
        List of manifest row dicts.

    Raises:
        RuntimeError: If no audio files decode, or if fewer than
            ``min_distinct_reciters`` reciters are present.
    """
    rows: list[dict[str, Any]] = []
    skipped = 0
    for audio_path in iter_audio_files(extract_root):
        if audio_path.parent.resolve() == extract_root.resolve():
            log.warning("skipping orphan file at extract root: %s", audio_path)
            skipped += 1
            continue
        reciter, surah = parse_ardad_filename(audio_path)
        if not reciter:
            log.warning("skipping file with no reciter directory: %s", audio_path)
            skipped += 1
            continue
        probe = probe_audio_file(audio_path)
        if probe is None:
            log.warning("skipping non-decodable file: %s", audio_path)
            skipped += 1
            continue
        sample_rate, duration_s = probe
        safe = safe_reciter_name(reciter)
        id_suffix = f"{surah:03d}" if surah is not None else audio_path.stem
        rows.append(
            {
                "clip_id": f"ardad_{safe}_{id_suffix}",
                "path": _to_repo_relative(audio_path),
                "reciter": reciter,
                "surah": surah,
                "ayah": None,
                "duration_s": duration_s,
                "sample_rate": sample_rate,
            }
        )

    log.info("manifest build: %d clips ok, %d skipped", len(rows), skipped)
    if not rows:
        raise RuntimeError(f"no decodable audio files under {extract_root}")
    distinct = len({row["reciter"] for row in rows})
    if distinct < min_distinct_reciters:
        raise RuntimeError(f"only {distinct} distinct reciters; need ≥ {min_distinct_reciters}")
    return rows


def download_archive(dataset_slug: str, dest: Path) -> Path:
    """Download an Ar-DAD archive from Kaggle to ``dest``.

    Lazy-imports the ``kaggle`` SDK so callers that pass ``--archive``
    never need the package installed. Authenticates via
    ``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME``/``KAGGLE_KEY``
    env vars.

    Args:
        dataset_slug: Kaggle dataset slug (``"<owner>/<dataset>"``).
        dest: Directory to store the downloaded zip(s).

    Returns:
        Path to the (newest) downloaded ``.zip`` file.

    Raises:
        RuntimeError: If the SDK is missing, authentication fails, or
            no zip ends up in ``dest`` after the API call.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415  (lazy)
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle SDK not installed. Run `pip install kaggle` (already in "
            "requirements.txt) or supply --archive PATH to a pre-downloaded zip."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # KaggleApi raises broad subclasses on missing creds
        raise RuntimeError(
            "Kaggle API authentication failed. Place credentials at "
            "~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY env "
            "vars; or supply --archive PATH to a pre-downloaded zip."
        ) from exc

    log.info("downloading kaggle dataset %s to %s", dataset_slug, dest)
    api.dataset_download_files(dataset_slug, path=str(dest), unzip=False, quiet=False)
    zips = sorted(dest.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"Kaggle download produced no .zip in {dest}")
    if len(zips) > 1:
        log.warning("multiple zip files in %s; using newest by mtime", dest)
        zips.sort(key=lambda p: p.stat().st_mtime)
    return zips[-1]


def extract_archive(zip_path: Path, dest: Path) -> Path:
    """Unzip ``zip_path`` into ``dest`` if no audio files exist there yet.

    Args:
        zip_path: Path to the source zip archive.
        dest: Destination directory; created if missing.

    Returns:
        ``dest`` (the extraction root).
    """
    dest.mkdir(parents=True, exist_ok=True)
    has_audio = any(p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS for p in dest.rglob("*"))
    if has_audio:
        log.info("extract dest %s already has audio files; skipping unzip", dest)
        return dest
    log.info("extracting %s into %s", zip_path, dest)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def write_attribution_file(output_dir: Path, dataset_slug: str | None) -> None:
    """Write a CC-style attribution stub for the Ar-DAD subset."""
    path = output_dir / "ATTRIBUTION.md"
    if path.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    slug = dataset_slug or "(provided via --archive)"
    url = (
        f"https://www.kaggle.com/datasets/{dataset_slug}"
        if dataset_slug
        else "(N/A — local archive)"
    )
    path.write_text(
        f"""# Ar-DAD subset attribution

Source dataset: `{slug}` (Kaggle)
URL: {url}
License: {DATASET_LICENSE}

Subset configuration:
- Generated by: scripts/download_ardad.py
- Generated at: {timestamp}
- Audio format: original (kept as-shipped on Kaggle)

Use of this subset must preserve the upstream attribution and license terms.
""",
        encoding="utf-8",
    )
    log.info("wrote attribution file to %s", path)


def _resolve_output_dir(cli_value: str | None, cfg: dict[str, Any]) -> Path:
    """Return ``--output-dir`` if provided, else ``<raw_dir>/ardad`` from cfg."""
    if cli_value:
        return Path(cli_value)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    return raw_dir / "ardad"


def _validate_args(args: argparse.Namespace) -> None:
    """Ensure either ``--dataset-slug`` or ``--archive`` is supplied.

    Raises:
        SystemExit: With a usage hint when both are missing.
    """
    if not args.dataset_slug and not args.archive:
        raise SystemExit(
            "error: one of --dataset-slug or --archive must be provided. "
            "Use --dataset-slug <user/slug> for Kaggle API mode, or "
            "--archive PATH for a pre-downloaded zip."
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. ``argv=None`` defaults to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=None,
        help="Kaggle dataset slug (e.g. 'owner/ar-dad'); required unless --archive is given.",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=None,
        help="Path to a pre-downloaded Kaggle zip; bypasses the Kaggle API.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override <raw_dir>/ardad from configs/default.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Verification seed (default: %(default)s).",
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
    """CLI entrypoint: download or accept archive, extract, manifest, verify."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)
    _validate_args(args)

    random.seed(args.seed)

    cfg = load_config(args.config)
    output_dir = _resolve_output_dir(args.output_dir, cfg)
    manifest_path = output_dir / "manifest.csv"

    log.info(
        "starting ar-dad ingestion: slug=%s archive=%s seed=%d output=%s",
        args.dataset_slug,
        args.archive,
        args.seed,
        output_dir,
    )

    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        if len(existing) >= EXPECTED_CLIPS_LOWER_BOUND:
            log.info(
                "manifest at %s already has %d clips (≥ %d); skipping download",
                manifest_path,
                len(existing),
                EXPECTED_CLIPS_LOWER_BOUND,
            )
            verify_random_clips(manifest_path, n=args.verify_n, seed=args.seed)
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.archive:
        zip_path = Path(args.archive).resolve()
        if not zip_path.is_file():
            raise SystemExit(f"error: --archive path does not exist: {zip_path}")
    else:
        zip_path = download_archive(args.dataset_slug, output_dir / "_zip")
    extract_archive(zip_path, output_dir)

    rows = build_manifest_rows(output_dir)
    write_manifest_csv(rows, manifest_path)
    write_attribution_file(output_dir, args.dataset_slug)
    verify_random_clips(manifest_path, n=args.verify_n, seed=args.seed)
    log.info("done: %d clips written under %s", len(rows), output_dir)


if __name__ == "__main__":
    main()
