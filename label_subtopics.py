#!/usr/bin/env python3
"""
label_subtopics.py — Subtopic presence + sentiment within an already-categorized topic.

Takes a labeled segment CSV (e.g. the output of label_from_codebook.py), filters it
down to segments already assigned a given topic label, and then asks the LLM, for each
of a fixed list of subtopics, whether it is substantively discussed in the segment and,
if so, the sentiment expressed specifically toward that subtopic (not the segment's
general tone). Subtopics are not mutually exclusive -- a segment can match zero, one,
or several.

This is a first-pass tool meant to be run and then manually reviewed.

Expects a CSV with at least these columns:
  video_url, segment_transcript, timecode_in, timecode_out, segment_duration_seconds,
  and a topic column (default: codebook_label) to filter on.

Subtopics:
  --subtopics accepts either:
    - a CSV with `label` and `definition` columns
    - a plain text file, one `label: definition` per line (definition optional)
  When omitted, a built-in default list of 12 subtopics is used (see DEFAULT_SUBTOPICS).

Segments are classified in batches (one API call scores multiple segments at once,
each independently) for efficiency. Results are cached per-batch, so a re-run only
pays for batches that haven't completed yet.

Usage:
  python label_subtopics.py \\
      --input labeled_segments.csv \\
      --topic-label "Donald Trump" \\
      --output-dir out/

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
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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

DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
MAX_TRANSCRIPT_CHARS = 700
DEFAULT_BATCH_SIZE = 20
NOT_DISCUSSED = "not_discussed"
SENTIMENT_VALUES = [NOT_DISCUSSED, "positive", "negative", "neutral"]

_base_url = os.environ.get("OPENAI_BASE_URL") or None  # None = the standard OpenAI endpoint
_api_key = os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(
    api_key=_api_key,
    base_url=_base_url,
    timeout=90.0,
)


# -------------------------
# Default subtopics
# -------------------------

DEFAULT_SUBTOPICS: Dict[str, str] = {
    "mental_health": (
        "Discussion of someone's mental health, cognitive fitness, memory, "
        "mental decline, or psychological/emotional state."
    ),
    "physical_health": (
        "Discussion of someone's physical health, medical condition, fitness, "
        "injuries, illness, or physical stamina."
    ),
    "christianity": (
        "Christianity, Christian faith, evangelical Christians, religious rhetoric, "
        "or appeals to Christian voters/communities."
    ),
    "corruption": (
        "Corruption, self-dealing, conflicts of interest, bribery, grift, abuse of "
        "power, or favor-trading by public officials or institutions."
    ),
    "iran_war": (
        "War or military conflict involving Iran, including strikes, escalation, "
        "retaliation, or broader Iran-related war discussion."
    ),
    "epstein": (
        "Jeffrey Epstein, including his actions, statements, victims, business "
        "interests, allies, or associates. References to only 'Jeff' should not "
        "count unless it's clear from context that it refers to Epstein."
    ),
    "debt": (
        "The national debt, federal deficit, government borrowing, or the debt ceiling."
    ),
    "vaccines": (
        "Vaccines (including mRNA vaccines), vaccination, immunization, or vaccine "
        "mandates/policy."
    ),
    "affordability": (
        "The affordability of everyday life -- cost of living, prices, wages, "
        "housing costs, or purchasing power."
    ),
    "transgender": (
        "Transgender people, transgender rights, or medical treatment related to "
        "transgender identity."
    ),
    "immigration_violence": (
        "Violence attributed to immigrants or crime committed by immigrants, or "
        "violence occurring during immigration enforcement actions."
    ),
    "border_chaos": (
        "Chaos, disorder, overcrowding, or crisis conditions at the U.S. border."
    ),
}


# -------------------------
# Helpers
# -------------------------

def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def slugify(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    return s.strip("_")


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
    with open(path, newline="", encoding="utf-8-sig") as f:
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
        })

    print(f"Loaded {len(segments)} segments from {path}", flush=True)
    return segments


def load_subtopics(path: str) -> Dict[str, str]:
    p = Path(path)
    subtopics: Dict[str, str] = {}
    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "label" not in (reader.fieldnames or []):
                die(f"--subtopics CSV must have a 'label' column (found: {reader.fieldnames})")
            for row in reader:
                label = (row.get("label") or "").strip()
                definition = (row.get("definition") or "").strip()
                if label:
                    subtopics[slugify(label)] = definition
    else:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    label, definition = line.split(":", 1)
                else:
                    label, definition = line, ""
                label = label.strip()
                if label:
                    subtopics[slugify(label)] = definition.strip()

    if not subtopics:
        die(f"No subtopics loaded from {path}")
    print(f"Loaded {len(subtopics)} subtopics from {path}", flush=True)
    return subtopics


# -------------------------
# LLM classification
# -------------------------

def build_system_prompt(subtopics: Dict[str, str]) -> str:
    lines = []
    for key, defn in subtopics.items():
        lines.append(f"- {key}: {defn}" if defn else f"- {key}")
    subtopics_text = "\n".join(lines)
    return f"""\
You are given a transcript segment. For each of the following {len(subtopics)} subtopics, \
determine whether it is substantively discussed in this specific segment, and if so, the \
sentiment expressed specifically toward that subtopic.

SUBTOPICS:
{subtopics_text}

Rules:
- Each subtopic is independent -- a segment can match zero, one, or several.
- Only consider a subtopic discussed if it is a meaningful focus of the segment, not an
  incidental or passing one-word mention. If not discussed, use "not_discussed".
- If discussed, assess the sentiment expressed specifically toward that subtopic --
  not the segment's overall tone. A segment can be generally negative in tone but
  neutral or positive toward a specific subtopic, or vice versa. Sentiment must be
  one of: "positive", "negative", "neutral".
- Base your answer only on the transcript text provided."""


def build_schema(subtopic_keys: List[str]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {"index": {"type": "integer"}}
    for key in subtopic_keys:
        properties[key] = {"type": "string", "enum": SENTIMENT_VALUES}
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
                    "required": ["index"] + subtopic_keys,
                    "properties": properties,
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
    subtopic_keys: List[str],
    system: str,
    schema: Dict[str, Any],
    model: str,
    cache_dir: Path,
) -> Dict[int, Dict[str, str]]:
    cache_path = cache_dir / "subtopic_labels" / f"{cache_key_for_batch(model, system, batch)}.json"
    cached = cache_load(cache_path)
    if cached is not None:
        return {int(k): v for k, v in cached.items()}

    items_text = "\n\n".join(
        f"[{i}] {s['segment_transcript'][:MAX_TRANSCRIPT_CHARS]}" for i, s in enumerate(batch)
    )
    user_msg = f"Classify these {len(batch)} segments:\n\n{items_text}"

    result = llm_call(system, user_msg, schema, model=model)
    fallback = {k: NOT_DISCUSSED for k in subtopic_keys}
    sentiment_by_idx: Dict[int, Dict[str, str]] = {}
    for c in result["classifications"]:
        row = {}
        for k in subtopic_keys:
            v = c.get(k, NOT_DISCUSSED)
            row[k] = v if v in SENTIMENT_VALUES else NOT_DISCUSSED
        sentiment_by_idx[c["index"]] = row
    # fill any missing indices with not_discussed so every segment gets a result
    for i in range(len(batch)):
        sentiment_by_idx.setdefault(i, dict(fallback))

    cache_save(cache_path, {str(k): v for k, v in sentiment_by_idx.items()})
    return sentiment_by_idx


# -------------------------
# Main
# -------------------------

def run(args: argparse.Namespace) -> None:
    if not _api_key:
        die("OPENAI_API_KEY not set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    print(f"Loading pre-labeled CSV: {args.input}", flush=True)
    all_segments = load_presegmented_csv(args.input)
    if not all_segments:
        die("No valid segments found in CSV")

    segments = [s for s in all_segments if s.get(args.topic_column) == args.topic_label]
    print(
        f"Filtered to {len(segments)} rows where {args.topic_column!r} == {args.topic_label!r} "
        f"(out of {len(all_segments)} total)",
        flush=True,
    )
    if not segments:
        die(f"No rows matched --topic-column {args.topic_column!r} == --topic-label {args.topic_label!r}")

    subtopics = load_subtopics(args.subtopics) if args.subtopics else dict(DEFAULT_SUBTOPICS)
    subtopic_keys = list(subtopics.keys())
    system = build_system_prompt(subtopics)
    schema = build_schema(subtopic_keys)

    for seg in segments:
        seg["topic_label"] = args.topic_label

    batches = [segments[i:i + args.batch_size] for i in range(0, len(segments), args.batch_size)]
    print(
        f"\nClassifying {len(segments)} segments in {len(batches)} batches of {args.batch_size} "
        f"({args.concurrency} concurrent) ...",
        flush=True,
    )

    def process_batch(bi: int) -> tuple[int, Dict[int, Dict[str, str]]]:
        batch = batches[bi]
        try:
            return bi, classify_batch(batch, subtopic_keys, system, schema, args.model, cache_dir)
        except Exception as e:
            print(f"  batch {bi + 1}/{len(batches)} FAILED, using not_discussed fallback for this batch: {e}", flush=True)
            return bi, {i: {k: NOT_DISCUSSED for k in subtopic_keys} for i in range(len(batch))}

    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(process_batch, bi) for bi in range(len(batches))]
        for future in as_completed(futures):
            bi, sentiment_by_idx = future.result()
            batch = batches[bi]
            for i, seg in enumerate(batch):
                seg.update(sentiment_by_idx.get(i, {k: NOT_DISCUSSED for k in subtopic_keys}))
            completed += 1
            if completed % 5 == 0 or completed == len(batches):
                print(f"  {completed}/{len(batches)} batches done (~{completed * args.batch_size} segments)", flush=True)

    segments_path = output_dir / "segments.json"
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(segments)} segments -> {segments_path}", flush=True)

    # summary: one row per subtopic (not mutually exclusive), sorted by discussed-count desc
    n_total = len(segments)
    summary_rows = []
    for key in subtopic_keys:
        matches = [s for s in segments if s.get(key) and s.get(key) != NOT_DISCUSSED]
        runtime = sum(float(s.get("segment_duration_seconds") or 0) for s in matches)
        samples = [s["segment_transcript"][:300] for s in matches[:3]] + ["", "", ""]
        summary_rows.append({
            "subtopic": key,
            "n_segments_discussed": len(matches),
            "pct_of_filtered": round(100 * len(matches) / n_total, 2) if n_total else 0.0,
            "n_positive": sum(1 for s in matches if s.get(key) == "positive"),
            "n_negative": sum(1 for s in matches if s.get(key) == "negative"),
            "n_neutral": sum(1 for s in matches if s.get(key) == "neutral"),
            "total_runtime_seconds_discussed": round(runtime, 1),
            "sample_1": samples[0], "sample_2": samples[1], "sample_3": samples[2],
        })
    summary_rows.sort(key=lambda r: r["n_segments_discussed"], reverse=True)

    summary_path = output_dir / "subtopic_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subtopic", "n_segments_discussed", "pct_of_filtered",
                        "n_positive", "n_negative", "n_neutral", "total_runtime_seconds_discussed",
                        "sample_1", "sample_2", "sample_3"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}", flush=True)

    # full labeled CSV, all original columns + one sentiment column per subtopic
    # (values: "not_discussed", "positive", "negative", "neutral")
    labeled_csv_path = output_dir / f"{slugify(args.topic_label)}_subtopics_labeled.csv"
    base_fields = [k for k in segments[0].keys() if k not in subtopic_keys]
    fieldnames = base_fields + subtopic_keys
    with open(labeled_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for seg in segments:
            writer.writerow({k: seg.get(k, "") for k in fieldnames})
    print(f"Wrote {labeled_csv_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Subtopic presence + per-subtopic sentiment within an already topic-categorized subset of segments."
    )
    ap.add_argument("--input", required=True, help="Labeled CSV with video_url, segment_transcript, and a topic column")
    ap.add_argument("--topic-label", default="Donald Trump", help="Value to filter --topic-column on (default: 'Donald Trump')")
    ap.add_argument("--topic-column", default="codebook_label", help="Column to filter on (default: codebook_label)")
    ap.add_argument("--subtopics", default=None, help="Optional CSV (label,definition columns) or text file to override the built-in 12 subtopics")
    ap.add_argument("--output-dir", default="pipeline_output", help="Directory for segments.json, subtopic_summary.csv, and the labeled CSV")
    ap.add_argument("--model", default=DEFAULT_CHAT_MODEL, help=f"Chat model for classification (default: {DEFAULT_CHAT_MODEL})")
    ap.add_argument("--cache-dir", default=".pipeline_cache", help="Directory for caching LLM responses")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Segments classified per API call (default: {DEFAULT_BATCH_SIZE})")
    ap.add_argument("--concurrency", type=int, default=8, help="Number of batches to classify concurrently (default: 8)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
