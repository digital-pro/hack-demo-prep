"""
Evaluate (source, back-translated) sentence pairs using the Hugging Face Inference API
sentence-similarity task. Pairs are read from a CSV file (default:
``es_ar_backtranslated_sample_20.csv``). ``HF_API_TOKEN`` may be set in a ``.env`` file
next to this script. Uses the hosted API only (no local model weights).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

# Load HF_API_TOKEN (and other vars) from a .env file next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CSV = Path("es_ar_backtranslated_sample_20.csv")
SNIPPET_WIDTH = 60
COL_IDX = 4
COL_SIM = 10
# API scores are in [0, 1] but may overshoot slightly in float32 (e.g. 1.0000001).
_SIMILARITY_TOLERANCE = 1e-3


def _require_hf_api_token() -> str:
    """Return the Hugging Face API token from the environment or raise."""
    token = os.environ.get("HF_API_TOKEN")
    if not token or not token.strip():
        raise RuntimeError(
            "HF_API_TOKEN is not set or is empty. Add it to a .env file next to this script "
            "(HF_API_TOKEN=hf_...), or export it in your shell, e.g.:\n"
            "  export HF_API_TOKEN=hf_...\n"
            "Create a token at https://huggingface.co/settings/tokens"
        )
    return token.strip()


def _normalize_csv_header(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def load_pairs_from_csv(path: Path) -> list[tuple[str, str, float | None]]:
    """
    Load (source, back-translated) pairs from a CSV file.

    Expects headers that normalize to ``source`` and ``backtranslated`` (e.g.
    ``Source`` and ``Back-translated``). An optional ``score`` column (e.g. ``Score``)
    is returned as the third tuple element for display; it is not used for API calls.

    Raises:
        ValueError: If required columns are missing or a row has empty source/back-translation.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")

        canon: dict[str, str] = {}
        for h in reader.fieldnames:
            if not h:
                continue
            key = _normalize_csv_header(h)
            if key == "source":
                canon["source"] = h
            elif key == "backtranslated":
                canon["backtranslated"] = h
            elif key == "score":
                canon["score"] = h

        if "source" not in canon or "backtranslated" not in canon:
            raise ValueError(
                f"CSV must include source and back-translated columns. Found: {reader.fieldnames!r}"
            )

        rows: list[tuple[str, str, float | None]] = []
        for line_no, row in enumerate(reader, start=2):
            src = (row.get(canon["source"]) or "").strip()
            bt = (row.get(canon["backtranslated"]) or "").strip()
            if not src and not bt:
                continue
            if not src or not bt:
                raise ValueError(
                    f"{path}:{line_no}: source and back-translated must both be non-empty"
                )
            gemini: float | None = None
            if "score" in canon:
                raw = (row.get(canon["score"]) or "").strip()
                if raw:
                    gemini = float(raw)
            rows.append((src, bt, gemini))

    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def compute_similarity_pair(source: str, backtranslated: str) -> float:
    """
    Call the Hugging Face Inference API sentence-similarity task for one pair.

    Uses ``huggingface_hub.InferenceClient``, which talks to the current inference
    router (``router.huggingface.co``). The legacy host ``api-inference.huggingface.co``
    is no longer supported.

    Compares ``source`` to a single candidate sentence ``backtranslated`` and returns
    the model's similarity score in the range [0.0, 1.0].

    Raises:
        RuntimeError: If HF_API_TOKEN is missing, or the inference request fails.
        ValueError: If the API returns an unexpected result shape, or a score far outside
            [0.0, 1.0] (small float overshoots are clamped).
    """
    token = _require_hf_api_token()
    model_id = os.environ.get("HF_SIMILARITY_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID

    client = InferenceClient(token=token)
    try:
        data = client.sentence_similarity(
            sentence=source,
            other_sentences=[backtranslated],
            model=model_id,
        )
    except (HfHubHTTPError, InferenceTimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Inference API request failed for model {model_id!r}: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list of similarity scores, got {type(data).__name__}: {data!r}"
        )
    if len(data) < 1:
        raise ValueError(f"Expected at least one similarity score, got an empty list: {data!r}")

    first = data[0]
    if isinstance(first, dict):
        raise ValueError(
            f"Unexpected list element type (dict). Raw response: {data!r}"
        )
    try:
        score = float(first)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not parse similarity score as float from {first!r}. Full response: {data!r}"
        ) from exc

    lo, hi = 0.0 - _SIMILARITY_TOLERANCE, 1.0 + _SIMILARITY_TOLERANCE
    if not (lo <= score <= hi) or not math.isfinite(score):
        raise ValueError(
            f"Similarity score out of expected range [0.0, 1.0] (±{_SIMILARITY_TOLERANCE}): "
            f"{score!r}. Full response: {data!r}"
        )

    return max(0.0, min(1.0, score))


def evaluate_pairs(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """
    Score each (source, back-translated) pair via :func:`compute_similarity_pair`.

    Returns:
        One dict per pair with keys: ``index`` (0-based), ``source``, ``backtranslation``,
        ``similarity`` (float in [0.0, 1.0]).
    """
    results: list[dict[str, Any]] = []
    for index, (source, backtranslated) in enumerate(pairs):
        similarity = compute_similarity_pair(source, backtranslated)
        results.append(
            {
                "index": index,
                "source": source,
                "backtranslation": backtranslated,
                "similarity": similarity,
            }
        )
    return results


def demo_sanity_checks() -> None:
    """
    Run two live Inference API calls to verify :func:`compute_similarity_pair` is not
    returning hard-coded values. Not mocked.
    """
    try:
        same = compute_similarity_pair("Hello world", "Hello world")
        assert same > 0.8, f"expected similarity > 0.8 for identical text, got {same!r}"
    except AssertionError as exc:
        print(
            "Sanity check failed: identical strings "
            '("Hello world", "Hello world") should score above 0.8. '
            f"Details: {exc}"
        )
        raise

    try:
        diff = compute_similarity_pair(
            "Hello world",
            "This sentence is about something else entirely",
        )
        assert diff < 0.8, f"expected similarity < 0.8 for unrelated text, got {diff!r}"
    except AssertionError as exc:
        print(
            'Sanity check failed: unrelated strings ("Hello world" vs long unrelated '
            "sentence) should score below 0.8. "
            f"Details: {exc}"
        )
        raise


def _snippet(text: str, max_len: int = SNIPPET_WIDTH) -> str:
    """Replace newlines and runs of whitespace with single spaces; truncate to ``max_len``."""
    single = " ".join(text.split())
    if len(single) <= max_len:
        return single
    return single[: max_len - 1] + "…"


def main() -> None:
    """Load pairs from CSV, score with the Inference API, print average and sorted table."""
    parser = argparse.ArgumentParser(
        description="Score back-translations with the Hugging Face Inference API."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV,
        help=f"CSV with Source, Back-translated, optional Score (default: {DEFAULT_CSV})",
    )
    args = parser.parse_args()

    loaded = load_pairs_from_csv(args.csv_path)
    pairs: list[tuple[str, str]] = [(s, b) for s, b, _ in loaded]
    gemini_by_index: dict[int, float | None] = {
        i: g for i, (_, _, g) in enumerate(loaded)
    }

    rows = evaluate_pairs(pairs)
    average = sum(row["similarity"] for row in rows) / len(rows)

    show_gemini = any(gemini_by_index.get(i) is not None for i in range(len(rows)))

    print(f"Average HF similarity: {average:.1f}\n")

    header = (
        f"{'idx':<{COL_IDX}} | {'similarity':>{COL_SIM}} | "
        f"{'source_snippet':<{SNIPPET_WIDTH}} | "
        f"{'backtranslation_snippet':<{SNIPPET_WIDTH}}"
    )
    rule = (
        f"{'-' * COL_IDX} | {'-' * COL_SIM} | "
        f"{'-' * SNIPPET_WIDTH} | {'-' * SNIPPET_WIDTH}"
    )
    if show_gemini:
        header += f" | {'gemini':>8}"
        rule += " | " + "-" * 8

    print(header)
    print(rule)

    for row in sorted(rows, key=lambda r: r["similarity"]):
        idx = row["index"]
        src = _snippet(row["source"])
        bt = _snippet(row["backtranslation"])
        sim = row["similarity"]
        sim_fmt = format(sim, f">{COL_SIM}.1f")
        line = (
            f"{idx:<{COL_IDX}} | {sim_fmt} | "
            f"{src:<{SNIPPET_WIDTH}} | {bt:<{SNIPPET_WIDTH}}"
        )
        if show_gemini:
            g = gemini_by_index.get(idx)
            g_str = f"{g:.2f}" if g is not None else "—"
            line += f" | {g_str:>8}"
        print(line)

    print()
    demo_sanity_checks()
    print("Sanity checks passed (live Inference API).")


if __name__ == "__main__":
    main()
