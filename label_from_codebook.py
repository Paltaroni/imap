#!/usr/bin/env python3
"""
label_from_codebook.py — Direct per-segment topic labeling against a fixed codebook.

Unlike direct_presegmented.py (which lets the LLM free-generate a topic label per
segment), this script constrains every classification to a closed set of categories
you supply. Each segment gets exactly one label, chosen from the codebook via a
JSON-schema enum -- the model cannot invent a new label string, so there's no
label drift to consolidate afterward.

Expects a CSV with at least these columns:
  video_url, segment_transcript, timecode_in, timecode_out, segment_duration_seconds

All other columns are passed through to the output.

Codebook:
  --codebook accepts either:
    - a CSV with a `label` column (any other columns are ignored)
    - a plain text file, one category per line (blank lines ignored)
  An "Other / Not Listed" category is added automatically if your codebook
  doesn't already have an equivalent, so segments that genuinely don't fit
  aren't force-matched to the closest wrong category.

Segments are classified in batches (one API call scores multiple segments at
once, each independently) for efficiency. Results are cached per-batch, so a
re-run only pays for batches that haven't completed yet.

Usage:
  python label_from_codebook.py \\
      --input sample.csv --codebook categories.csv --output-dir out/

Auth:
  export OPENAI_API_KEY="..."
  export OPENAI_BASE_URL="..."  # optional override
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
    _script_dir = Path(__file__).resolve().parent
    for _candidate in [_script_dir / ".env", _script_dir.parent / ".env"]:
        if _candidate.exists():
            load_dotenv(_candidate)
            break
    else:
        load_dotenv()
except ImportError:
    pass

csv.field_size_limit(10**7)

DEFAULT_CHAT_MODEL = "gpt-5.6-sol"
OTHER_LABEL = "Other / Not Listed"
MAX_TRANSCRIPT_CHARS = 700
DEFAULT_BATCH_SIZE = 20

_base_url = os.environ.get(
    "OPENAI_BASE_URL",
    "https://go.apis.huit.harvard.edu/ais-openai-direct/v1/",
)
_api_key = os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(
    api_key=_api_key,
    base_url=_base_url,
    default_headers={"api-key": _api_key},
    timeout=90.0,
)


# -------------------------
# Helpers
# -------------------------

def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_load(path: Path) -> Optional[Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def cache_save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def timecode_to_ms(tc: str) -> int:
    parts = tc.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000
    elif len(parts) == 2:
        m, s = parts
        return (int(m) * 60 + int(s)) * 1000
    return 0


# -------------------------
# Input loading
# -------------------------

def load_presegmented_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    segments: List[Dict[str, Any]] = []
    n = len(rows)
    for i, row in enumerate(rows):
        text = (row.get("segment_transcript") or "").strip()
        url = (row.get("video_url") or "").strip()
        if not text or not url:
            print(f"  [csv] row {i + 1}/{n}: skipping (missing video_url or segment_transcript)", file=sys.stderr)
            continue

        tc_in = (row.get("timecode_in") or "0:00").strip()
        tc_out = (row.get("timecode_out") or "0:00").strip()
        try:
            in_ms = timecode_to_ms(tc_in)
            out_ms = timecode_to_ms(tc_out)
        except Exception:
            in_ms, out_ms = 0, 0

        try:
            dur = float(row.get("segment_duration_seconds") or 0)
        except Exception:
            dur = round((out_ms - in_ms) / 1000, 3)

        passthrough = {k: str(v).strip() for k, v in row.items() if k != "segment_transcript"}

        segments.append({
            "video_url": url,
            **passthrough,
            "timecode_in_ms": in_ms,
            "timecode_out_ms": out_ms,
            "timecode_in": tc_in,
            "timecode_out": tc_out,
            "segment_duration_seconds": dur,
            "segment_transcript": text,
            "codebook_label": None,
        })

    print(f"Loaded {len(segments)} segments from {path}", flush=True)
    return segments


def load_codebook(path: str) -> List[str]:
    p = Path(path)
    labels: List[str] = []
    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "label" not in (reader.fieldnames or []):
                die(f"--codebook CSV must have a 'label' column (found: {reader.fieldnames})")
            for row in reader:
                val = (row.get("label") or "").strip()
                if val:
                    labels.append(val)
    else:
        with open(p, encoding="utf-8") as f:
            for line in f:
                val = line.strip()
                if val:
                    labels.append(val)

    # de-dupe, preserve order
    seen = set()
    deduped = []
    for l in labels:
        if l.lower() not in seen:
            seen.add(l.lower())
            deduped.append(l)

    if not any(l.lower() in (OTHER_LABEL.lower(), "other", "not listed", "none", "n/a") for l in deduped):
        deduped.append(OTHER_LABEL)

    print(f"Loaded codebook: {len(deduped)} categories from {path}", flush=True)
    return deduped


def load_definitions(path: str, column: str) -> Dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "label" not in (reader.fieldnames or []):
            die(f"--codebook-definitions CSV must have a 'label' column (found: {reader.fieldnames})")
        if column not in (reader.fieldnames or []):
            die(f"--definitions-column {column!r} not found in --codebook-definitions (found: {reader.fieldnames})")
        defs = {}
        for row in reader:
            label = (row.get("label") or "").strip()
            definition = (row.get(column) or "").strip()
            if label:
                defs[label.lower()] = definition
    print(f"Loaded {len(defs)} definitions from {path} (column: {column})", flush=True)
    return defs


# -------------------------
# LLM classification
# -------------------------

def build_system_prompt(labels: List[str], definitions: Optional[Dict[str, str]] = None) -> str:
    if definitions:
        lines = []
        for l in labels:
            defn = definitions.get(l.lower(), "")
            lines.append(f"- {l}: {defn}" if defn else f"- {l}")
        labels_text = "\n".join(lines)
    else:
        labels_text = "\n".join(f"- {l}" for l in labels)
    return f"""\
You classify podcast/YouTube transcript segments into exactly one category from a FIXED codebook.

CODEBOOK (choose only from these {len(labels)} categories -- the category string must match exactly):
{labels_text}

Rules:
- Every segment must get exactly one category from the list above -- do not invent a new category.
- Choose the category that best fits the segment's primary subject.
- If the segment doesn't clearly fit any category, or is filler/boilerplate/unclear content, use "{OTHER_LABEL}"."""


def build_schema(labels: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["classifications"],
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "label"],
                    "properties": {
                        "index": {"type": "integer"},
                        "label": {"type": "string", "enum": labels},
                    },
                },
            }
        },
    }


def llm_call(
    system: str,
    user: str,
    schema: Dict[str, Any],
    model: str,
    max_tokens: int = 3000,
    max_retries: int = 4,
) -> Dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "response", "strict": True, "schema": schema},
                },
            )
            if resp.choices[0].finish_reason == "length":
                raise RuntimeError(
                    f"truncated at max_completion_tokens={max_tokens} (reasoning tokens consumed the budget)"
                )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                max_tokens *= 2
                time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def cache_key_for_batch(model: str, system: str, batch: List[Dict[str, Any]]) -> str:
    ids = "|".join(f"{s['video_url']}::{s.get('segment_id','')}" for s in batch)
    return sha256(f"{model}|{system}|{ids}")


def classify_batch(
    batch: List[Dict[str, Any]],
    labels: List[str],
    system: str,
    schema: Dict[str, Any],
    model: str,
    cache_dir: Path,
) -> Dict[int, str]:
    cache_path = cache_dir / "codebook_labels" / f"{cache_key_for_batch(model, system, batch)}.json"
    cached = cache_load(cache_path)
    if cached is not None:
        return {int(k): v for k, v in cached.items()}

    items_text = "\n\n".join(
        f"[{i}] {s['segment_transcript'][:MAX_TRANSCRIPT_CHARS]}" for i, s in enumerate(batch)
    )
    user_msg = f"Classify these {len(batch)} segments:\n\n{items_text}"

    result = llm_call(system, user_msg, schema, model=model)
    label_by_idx = {c["index"]: c["label"] for c in result["classifications"]}
    # fill any missing indices with the fallback so every segment gets labeled
    for i in range(len(batch)):
        label_by_idx.setdefault(i, OTHER_LABEL)

    cache_save(cache_path, {str(k): v for k, v in label_by_idx.items()})
    return label_by_idx


# -------------------------
# Main
# -------------------------

def run(args: argparse.Namespace) -> None:
    if not _api_key:
        die("OPENAI_API_KEY not set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    print(f"Loading pre-segmented CSV: {args.input}", flush=True)
    all_segments = load_presegmented_csv(args.input)
    if not all_segments:
        die("No valid segments found in CSV")

    labels = load_codebook(args.codebook)
    definitions = load_definitions(args.codebook_definitions, args.definitions_column) if args.codebook_definitions else None
    system = build_system_prompt(labels, definitions)
    schema = build_schema(labels)

    batches = [all_segments[i:i + args.batch_size] for i in range(0, len(all_segments), args.batch_size)]
    print(f"\nClassifying {len(all_segments)} segments in {len(batches)} batches of {args.batch_size} ...", flush=True)

    for bi, batch in enumerate(batches):
        try:
            label_by_idx = classify_batch(batch, labels, system, schema, args.model, cache_dir)
        except Exception as e:
            print(f"  batch {bi + 1}/{len(batches)} FAILED, using fallback for this batch: {e}", flush=True)
            label_by_idx = {i: OTHER_LABEL for i in range(len(batch))}

        for i, seg in enumerate(batch):
            seg["codebook_label"] = label_by_idx.get(i, OTHER_LABEL)

        if (bi + 1) % 5 == 0 or bi == len(batches) - 1:
            print(f"  batch {bi + 1}/{len(batches)} done ({(bi + 1) * args.batch_size} segments)", flush=True)

    segments_path = output_dir / "segments.json"
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_segments)} segments -> {segments_path}", flush=True)

    # summary: one row per category, sorted by runtime desc
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "n_segments": 0, "total_runtime_seconds": 0.0, "samples": [],
    })
    for seg in all_segments:
        lbl = seg["codebook_label"] or OTHER_LABEL
        groups[lbl]["n_segments"] += 1
        groups[lbl]["total_runtime_seconds"] += float(seg.get("segment_duration_seconds") or 0)
        if len(groups[lbl]["samples"]) < 3:
            groups[lbl]["samples"].append(seg["segment_transcript"][:300])

    summary_path = output_dir / "codebook_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "n_segments", "total_runtime_seconds", "sample_1", "sample_2", "sample_3"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for label in labels:  # keep codebook order, including zero-count categories
            info = groups.get(label, {"n_segments": 0, "total_runtime_seconds": 0.0, "samples": []})
            samples = info["samples"] + ["", "", ""]
            writer.writerow({
                "label": label,
                "n_segments": info["n_segments"],
                "total_runtime_seconds": round(info["total_runtime_seconds"], 1),
                "sample_1": samples[0], "sample_2": samples[1], "sample_3": samples[2],
            })
    print(f"Wrote {summary_path}", flush=True)

    # full labeled CSV, all original columns + codebook_label
    labeled_csv_path = output_dir / "labeled_segments.csv"
    base_fields = [k for k in all_segments[0].keys() if k != "codebook_label"]
    fieldnames = base_fields + ["codebook_label"]
    with open(labeled_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for seg in all_segments:
            writer.writerow({k: seg.get(k, "") for k in fieldnames})
    print(f"Wrote {labeled_csv_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Directly label pre-segmented transcript rows against a fixed codebook of categories."
    )
    ap.add_argument("--input", required=True, help="Pre-segmented CSV with video_url and segment_transcript columns")
    ap.add_argument("--codebook", required=True, help="CSV with a 'label' column, or a plain text file (one category per line)")
    ap.add_argument("--codebook-definitions", default=None, help="Optional CSV with a 'label' column and a definitions column, used to give the model a one-sentence description per category")
    ap.add_argument("--definitions-column", default="definition", help="Column name in --codebook-definitions to use as the definition text (default: definition)")
    ap.add_argument("--output-dir", default="pipeline_output", help="Directory for segments.json, codebook_summary.csv, labeled_segments.csv")
    ap.add_argument("--model", default=DEFAULT_CHAT_MODEL, help=f"Chat model for classification (default: {DEFAULT_CHAT_MODEL})")
    ap.add_argument("--cache-dir", default=".pipeline_cache", help="Directory for caching LLM responses")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Segments classified per API call (default: {DEFAULT_BATCH_SIZE})")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
