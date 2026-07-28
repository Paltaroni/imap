#!/usr/bin/env python3
"""
transcript_segmenter.py — Split transcripts into topic segments for human labeling.

Uses an LLM to detect topic-shift boundaries in transcripts, but does NOT label
the topics. Output is intended for human review and labeling before running
sentiment analysis.

Usage:
  python3 transcript_segmenter.py --input videos.csv --output-dir out/

  Then open out/segments_review.csv, fill in the `topic` column for each row,
  and use that as input for the sentiment phase of direct_topic_pipeline.py.

Output files (written to --output-dir):
  segments.json        — all segments with metadata + timecodes + transcript text
  segments_review.csv  — one row per segment, with empty `topic` column for human labeling

Auth:
  export OPENAI_API_KEY="..."
  export OPENAI_BASE_URL="..."  # optional override
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from openai import OpenAI

csv.field_size_limit(10**7)

DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_CHUNK_MINUTES = 10
DEFAULT_OVERLAP_SECONDS = 15

_base_url = os.environ.get(
    "OPENAI_BASE_URL",
    "https://go.apis.huit.harvard.edu/ais-openai-direct/v1/",
)
_api_key = os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(
    api_key=_api_key,
    base_url=_base_url,
    default_headers={"api-key": _api_key},
)


# -------------------------
# Helpers
# -------------------------

def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def ms_to_timecode(ms: int) -> str:
    total_seconds = int(ms) // 1000
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


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


# -------------------------
# Transcript parsing
# -------------------------

class _HtmlTextExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style", "head", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._depth = 0
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, _attrs: Any) -> None:
        if tag in self._SKIP_TAGS:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._depth == 0:
            stripped = data.strip()
            if stripped:
                self.parts.append(stripped)


def _html_to_text(raw_html: str) -> str:
    try:
        extractor = _HtmlTextExtractor()
        extractor.feed(raw_html)
        return re.sub(r"\s+", " ", " ".join(extractor.parts)).strip()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", raw_html)
        return re.sub(r"\s+", " ", text).strip()


def _extract_yt_initial_player_response(raw_html: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"ytInitialPlayerResponse\s*=\s*(\{)", raw_html)
    if not match:
        return None
    start = match.start(1)
    depth = 0
    in_string = False
    escape = False
    end = None
    for i, ch in enumerate(raw_html[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    try:
        return json.loads(raw_html[start:end])
    except Exception:
        return None


def _extract_timedtext_url(raw_html: str) -> Optional[str]:
    data = _extract_yt_initial_player_response(raw_html)
    if not data:
        return None
    try:
        tracks = (
            data.get("captions", {})
            .get("playerCaptionsTracklistRenderer", {})
            .get("captionTracks", [])
        )
    except Exception:
        return None
    if not isinstance(tracks, list) or not tracks:
        return None
    preferred = next((t for t in tracks if str(t.get("languageCode", "")).startswith("en")), tracks[0])
    base_url = preferred.get("baseUrl")
    if not base_url:
        return None
    parsed = urllib.parse.urlparse(base_url)
    query = urllib.parse.parse_qs(parsed.query)
    query["fmt"] = ["json3"]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))


def _fetch_timedtext(url: str) -> Optional[List[Dict[str, Any]]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        if data.lstrip().startswith("{"):
            parsed = json.loads(data)
            utterances = []
            for event in parsed.get("events", []):
                parts = event.get("segs") or []
                text = "".join(str(part.get("utf8", "")) for part in parts)
                text = html.unescape(re.sub(r"\s+", " ", text)).strip()
                if not text:
                    continue
                start = int(event.get("tStartMs", 0) or 0)
                dur = int(event.get("dDurationMs", 1000) or 1000)
                utterances.append({"start_ms": start, "end_ms": start + max(dur, 1), "text": text})
            return normalize_transcript_records(utterances) or None
        root = ET.fromstring(data)
        utterances = []
        for el in root.findall(".//text"):
            start = float(el.get("start", 0))
            dur = float(el.get("dur", 1))
            text = html.unescape(re.sub(r"\s+", " ", "".join(el.itertext()))).strip()
            if text:
                utterances.append({
                    "start_ms": int(start * 1000),
                    "end_ms": int((start + dur) * 1000),
                    "text": text,
                })
        return normalize_transcript_records(utterances) or None
    except Exception:
        return None


def _vtt_ts_to_ms(ts: str) -> int:
    parts = ts.strip().split(":")
    h, m, s = (parts if len(parts) == 3 else ["0"] + parts)
    s, ms = (s.split(".") + ["0"])[:2]
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms[:3].ljust(3, "0"))


def _time_value_to_ms(value: Any, unit: str = "seconds") -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return int(number if unit == "ms" else number * 1000)
    raw = str(value).strip().strip("<>")
    if not raw:
        return None
    if ":" in raw:
        try:
            return _vtt_ts_to_ms(raw.replace(",", "."))
        except Exception:
            return None
    lowered = raw.lower()
    inferred_unit = unit
    if lowered.endswith("ms"):
        inferred_unit = "ms"
        lowered = lowered[:-2]
    elif lowered.endswith("s"):
        inferred_unit = "seconds"
        lowered = lowered[:-1]
    try:
        number = float(lowered)
    except Exception:
        return None
    return int(number if inferred_unit == "ms" else number * 1000)


def _transcript_text_value(item: Dict[str, Any]) -> str:
    for key in ["text", "caption", "content", "transcript", "snippet", "utf8"]:
        value = item.get(key)
        if value is not None and str(value).strip():
            return re.sub(r"\s+", " ", html.unescape(str(value))).strip()
    return ""


def _field_ms(item: Dict[str, Any], keys: List[str], unit: str) -> Optional[int]:
    for key in keys:
        if key in item:
            value = _time_value_to_ms(item.get(key), unit=unit)
            if value is not None:
                return value
    return None


def normalize_transcript_records(
    records: List[Any],
    duration_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    start_ms_keys = ["start_ms", "startMs", "start_time_ms", "startTimeMs", "offset_ms", "offsetMs", "timecode_in_ms"]
    start_s_keys = ["start", "start_seconds", "startSecond", "start_time", "startTime", "offset", "startOffset"]
    end_ms_keys = ["end_ms", "endMs", "end_time_ms", "endTimeMs", "timecode_out_ms"]
    end_s_keys = ["end", "end_seconds", "endSecond", "end_time", "endTime"]
    duration_ms_keys = ["duration_ms", "durationMs", "dur_ms", "dDurationMs"]
    duration_s_keys = ["duration", "duration_seconds", "durationSeconds", "dur"]

    pending: List[Dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict):
            text = _transcript_text_value(record)
            start_ms = _field_ms(record, start_ms_keys, "ms") or _field_ms(record, start_s_keys, "seconds")
            end_ms = _field_ms(record, end_ms_keys, "ms") or _field_ms(record, end_s_keys, "seconds")
            dur_ms = _field_ms(record, duration_ms_keys, "ms") or _field_ms(record, duration_s_keys, "seconds")
            if end_ms is None and start_ms is not None and dur_ms is not None:
                end_ms = start_ms + max(dur_ms, 1)
        else:
            text = re.sub(r"\s+", " ", html.unescape(str(record))).strip()
            start_ms = end_ms = None
        if text:
            pending.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})

    if not pending:
        return []

    if all(item["start_ms"] is None for item in pending):
        total_ms = max(int(duration_seconds * 1000), len(pending) * 1000)
        step_ms = max(1, total_ms // len(pending))
        for idx, item in enumerate(pending):
            item["start_ms"] = idx * step_ms
            item["end_ms"] = (idx + 1) * step_ms if idx < len(pending) - 1 else total_ms
    else:
        prev_end = 0
        for idx, item in enumerate(pending):
            if item["start_ms"] is None:
                item["start_ms"] = prev_end
            if item["end_ms"] is None:
                next_start = next(
                    (pending[j]["start_ms"] for j in range(idx + 1, len(pending))
                     if pending[j]["start_ms"] is not None),
                    None,
                )
                item["end_ms"] = (int(next_start) if next_start and int(next_start) > int(item["start_ms"])
                                  else int(item["start_ms"]) + 1000)
            prev_end = max(prev_end, int(item["end_ms"]))

    normalized = [
        {"start_ms": int(i["start_ms"]), "end_ms": max(int(i["end_ms"]), int(i["start_ms"])), "text": str(i["text"]).strip()}
        for i in pending if str(i.get("text", "")).strip()
    ]
    return ensure_end_ms(normalized)


def _parse_vtt_inline(text: str, duration_seconds: float) -> Optional[List[Dict[str, Any]]]:
    ts_pattern = re.compile(r"<(\d{2}:\d{2}:\d{2}\.\d+)>(?:<c>)?([^<]*)(?:</c>)?")
    matches = ts_pattern.findall(text)
    if not matches:
        return None
    seen: set = set()
    words: List[Tuple[int, str]] = []
    for ts_str, word in matches:
        word = html.unescape(word).strip()
        if word and (key := (_vtt_ts_to_ms(ts_str), word)) not in seen:
            seen.add(key)
            words.append(key)
    if not words:
        return None
    total_ms = max(int(duration_seconds * 1000), words[-1][0] + 1000)
    utterances: List[Dict[str, Any]] = []
    window_words: List[str] = []
    window_start = words[0][0]
    for ms, word in words:
        if ms >= window_start + 5000 and window_words:
            utterances.append({"start_ms": window_start, "end_ms": ms, "text": " ".join(window_words)})
            window_start = ms
            window_words = []
        window_words.append(word)
    if window_words:
        utterances.append({"start_ms": window_start, "end_ms": total_ms, "text": " ".join(window_words)})
    return utterances or None


def _text_to_utterances(text: str, duration_seconds: float) -> List[Dict[str, Any]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return []
    total_ms = max(int(duration_seconds * 1000), len(sentences) * 1000)
    step_ms = max(1, total_ms // len(sentences))
    return [
        {"start_ms": i * step_ms, "end_ms": (i + 1) * step_ms if i < len(sentences) - 1 else total_ms, "text": sent}
        for i, sent in enumerate(sentences)
    ]


def ensure_end_ms(transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, utt in enumerate(transcript):
        start = int(utt.get("start_ms", 0))
        end = utt.get("end_ms")
        if end is None:
            end = int(transcript[i + 1].get("start_ms", start)) if i + 1 < len(transcript) else start + 1000
        end = max(int(end), start)
        out.append({"start_ms": start, "end_ms": end, "text": str(utt.get("text", "")).strip()})
    out.sort(key=lambda u: u["start_ms"])
    return out


def coalesce_transcript(transcript: List[Dict[str, Any]], window_ms: int = 8000) -> List[Dict[str, Any]]:
    if not transcript:
        return []
    out: List[Dict[str, Any]] = []
    ws = transcript[0]["start_ms"]
    texts: List[str] = []
    we = transcript[0]["end_ms"]
    for utt in transcript:
        if utt["start_ms"] >= ws + window_ms:
            if texts:
                out.append({"start_ms": ws, "end_ms": we, "text": " ".join(texts)})
            ws = utt["start_ms"]
            texts = []
        if t := str(utt.get("text", "")).strip():
            texts.append(t)
        we = utt["end_ms"]
    if texts:
        out.append({"start_ms": ws, "end_ms": we, "text": " ".join(texts)})
    return out


def slice_transcript(transcript: List[Dict[str, Any]], start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    return [u for u in transcript if u["end_ms"] > start_ms and u["start_ms"] < end_ms]


def transcript_text(transcript: List[Dict[str, Any]], start_ms: int, end_ms: int) -> str:
    return " ".join(
        t for u in slice_transcript(transcript, start_ms, end_ms)
        if (t := str(u.get("text", "")).strip())
    )


def _row_metadata(row: Dict[str, Any], row_index: int, duration_seconds: float) -> Dict[str, str]:
    metadata: Dict[str, str] = {k: str(v).strip() for k, v in row.items() if k != "transcript"}
    metadata["input_row_index"] = str(row_index)
    metadata["video_duration_seconds"] = str(round(duration_seconds, 3)) if duration_seconds else ""
    return metadata


def load_csv_as_videos(path: str, fetch_timedtext: bool = True) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    videos: List[Dict[str, Any]] = []
    n = len(rows)

    for i, row in enumerate(rows):
        url = (row.get("source_url") or "").strip()
        raw = (row.get("transcript") or "").strip()
        try:
            duration_seconds = float(row.get("duration_seconds") or 0)
        except Exception:
            duration_seconds = 0.0
        metadata = _row_metadata(row, i + 1, duration_seconds)

        if not url or not raw:
            print(f"  [csv] row {i + 1}/{n}: skipping (missing source_url or transcript)", file=sys.stderr)
            continue
        if "About Press Copyright Contact us Creators Advertise" in raw:
            print(f"  [csv] row {i + 1}/{n}: skipping (YouTube boilerplate)", file=sys.stderr)
            continue

        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list) and parsed:
                    normalized = normalize_transcript_records(parsed, duration_seconds)
                    if normalized:
                        print(f"  [csv] row {i + 1}/{n}: {url} — {len(normalized)} utterances (JSON)", flush=True)
                        videos.append({"url": url, "transcript": normalized, "metadata": metadata})
                        continue
            except Exception:
                pass

        vtt = _parse_vtt_inline(raw, duration_seconds)
        if vtt:
            normalized = normalize_transcript_records(vtt, duration_seconds)
            if normalized:
                print(f"  [csv] row {i + 1}/{n}: {url} — {len(normalized)} utterances (inline VTT)", flush=True)
                videos.append({"url": url, "transcript": normalized, "metadata": metadata})
                continue

        if raw.startswith("<") and fetch_timedtext:
            tt_url = _extract_timedtext_url(raw)
            if tt_url:
                utterances = _fetch_timedtext(tt_url)
                if utterances:
                    print(f"  [csv] row {i + 1}/{n}: {url} — {len(utterances)} utterances (timedtext)", flush=True)
                    videos.append({"url": url, "transcript": utterances, "metadata": metadata})
                    continue

        text = _html_to_text(raw) if raw.startswith("<") else raw
        text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d+>|</?c>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        utterances = _text_to_utterances(text, duration_seconds)
        if utterances:
            normalized = normalize_transcript_records(utterances, duration_seconds)
            print(f"  [csv] row {i + 1}/{n}: {url} — {len(normalized)} utterances (text)", flush=True)
            videos.append({"url": url, "transcript": normalized, "metadata": metadata})
        else:
            print(f"  [csv] row {i + 1}/{n}: skipping (empty transcript)", file=sys.stderr)

    return videos


# -------------------------
# LLM helper
# -------------------------

def llm_call(
    system: str,
    user: str,
    schema: Dict[str, Any],
    model: str,
    max_tokens: int = 512,
    max_retries: int = 3,
) -> Dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "response", "strict": True, "schema": schema},
                },
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


# -------------------------
# Segmentation
# -------------------------

_SEG_SYSTEM = """\
You segment transcripts by shifts in subject matter.

Rules:
- Divide the transcript into contiguous segments with no gaps or overlaps.
- Minimum 60 seconds per segment; merge adjacent spans with weak shifts.
- The first segment must start exactly at chunk_start_ms.
- The last segment must end exactly at chunk_end_ms.
- Return only timecode boundaries — do not label topics.
"""

_SEG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["segments"],
    "properties": {
        "segments": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["timecode_in_ms", "timecode_out_ms"],
                "properties": {
                    "timecode_in_ms": {"type": "integer"},
                    "timecode_out_ms": {"type": "integer"},
                },
            },
        }
    },
}


def _reconcile_chunk_bounds(segs: List[Dict[str, Any]], chunk_start: int, chunk_end: int) -> List[Tuple[int, int]]:
    bounds: List[Tuple[int, int]] = []
    for s in segs:
        a = max(chunk_start, int(s.get("timecode_in_ms", chunk_start)))
        b = min(chunk_end, int(s.get("timecode_out_ms", chunk_end)))
        if b > a:
            bounds.append((a, b))
    bounds.sort()

    out: List[Tuple[int, int]] = []
    cur = chunk_start
    for a, b in bounds:
        a = max(a, cur)
        if b > a:
            if a > cur:
                out.append((cur, a))
            out.append((a, b))
            cur = b
    if cur < chunk_end:
        out.append((cur, chunk_end))
    return out


def segment_video(
    transcript: List[Dict[str, Any]],
    model: str,
    cache_dir: Path,
    chunk_ms: int,
    overlap_ms: int,
) -> List[Tuple[int, int]]:
    transcript = ensure_end_ms(transcript)
    if not transcript:
        return []

    t0 = transcript[0]["start_ms"]
    t1 = transcript[-1]["end_ms"]

    spans: List[Tuple[int, int]] = []
    t = t0
    while t < t1:
        a, b = t, min(t1, t + chunk_ms)
        spans.append((a, b))
        if b == t1:
            break
        t = b - overlap_ms

    accepted_bounds: List[Dict[str, Any]] = []
    last_chunk_idx = len(spans) - 1
    for chunk_idx, (chunk_a, chunk_b) in enumerate(spans):
        chunk = coalesce_transcript(slice_transcript(transcript, chunk_a, chunk_b))
        if not chunk:
            continue
        user_msg = (
            f"chunk_start_ms: {chunk_a}\nchunk_end_ms: {chunk_b}\n\n"
            f"Transcript:\n{json.dumps(chunk, ensure_ascii=False)}"
        )
        cache_key = sha256(f"{model}|{_SEG_SYSTEM}|{user_msg}")
        cache_path = cache_dir / "segmentation" / f"{cache_key}.json"
        cached = cache_load(cache_path)
        if cached is not None:
            raw_segs = cached.get("segments", [])
        else:
            result = llm_call(_SEG_SYSTEM, user_msg, _SEG_SCHEMA, model=model, max_tokens=1024)
            cache_save(cache_path, result)
            raw_segs = result.get("segments", [])
        raw_bounds = _reconcile_chunk_bounds(raw_segs, chunk_a, chunk_b)
        emit_start = chunk_a
        emit_end = chunk_b if chunk_idx == last_chunk_idx else max(chunk_a, chunk_b - overlap_ms)
        for raw_a, raw_b in raw_bounds:
            a = max(raw_a, emit_start)
            b = min(raw_b, emit_end)
            if b > a:
                accepted_bounds.append({
                    "start": a, "end": b,
                    "chunk_idx": chunk_idx,
                    "handoff_start": chunk_idx > 0 and a == emit_start,
                    "handoff_end": chunk_idx < last_chunk_idx and b == emit_end,
                })

    accepted_bounds.sort(key=lambda item: (item["start"], item["end"]))

    merged: List[Dict[str, Any]] = []
    for item in accepted_bounds:
        if not merged:
            merged.append(dict(item))
            continue
        prev = merged[-1]
        if item["start"] < prev["end"]:
            if item["end"] > prev["end"]:
                prev["end"] = item["end"]
                prev["chunk_idx"] = item["chunk_idx"]
                prev["handoff_end"] = item["handoff_end"]
            continue
        if (
            item["start"] == prev["end"]
            and item["chunk_idx"] != prev["chunk_idx"]
            and (item["handoff_start"] or prev["handoff_end"])
        ):
            prev["end"] = item["end"]
            prev["chunk_idx"] = item["chunk_idx"]
            prev["handoff_end"] = item["handoff_end"]
            continue
        merged.append(dict(item))

    final: List[Tuple[int, int]] = []
    cur = t0
    for item in merged:
        a, b = int(item["start"]), int(item["end"])
        if a > cur:
            final.append((cur, a))
        final.append((a, b))
        cur = b
    if cur < t1:
        final.append((cur, t1))

    return [(a, b) for a, b in final if b > a]


# -------------------------
# Main
# -------------------------

def run(args: argparse.Namespace) -> None:
    if not _api_key:
        die("OPENAI_API_KEY not set")

    # Each run gets its own timestamped subfolder so previous runs are never overwritten.
    # e.g. Top_100_segments/2026-06-22_143052_videos/
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    input_stem = Path(args.input).stem
    run_dir = Path(args.output_dir) / f"{timestamp}_{input_stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir
    print(f"Output directory: {output_dir}", flush=True)

    cache_dir = Path(args.cache_dir)

    print(f"Loading CSV: {args.input}", flush=True)
    videos = load_csv_as_videos(args.input, fetch_timedtext=not args.no_fetch_timedtext)
    if not videos:
        die("No valid videos found in CSV")

    if args.skip_segments:
        skip_urls: set = set()
        for sf in args.skip_segments:
            sf_path = Path(sf)
            if sf_path.exists():
                with open(sf_path, encoding="utf-8") as f:
                    for seg in json.load(f):
                        skip_urls.add(seg.get("video_url", ""))
                print(f"Loaded {len(skip_urls)} already-segmented URLs from {sf_path}", flush=True)
            else:
                print(f"[warn] --skip-segments file not found: {sf_path}", flush=True)
        if skip_urls:
            before = len(videos)
            videos = [v for v in videos if v["url"] not in skip_urls]
            print(f"Skipping {before - len(videos)} already-segmented videos ({len(videos)} new to process)", flush=True)
        if not videos:
            die("No new videos to segment after skipping already-segmented ones")

    chunk_ms = args.chunk_minutes * 60_000
    overlap_ms = args.overlap_seconds * 1000

    all_segments: List[Dict[str, Any]] = []
    for idx, vid in enumerate(videos):
        url = vid["url"]
        transcript = vid["transcript"]
        metadata = {k: str(v) for k, v in vid.get("metadata", {}).items()}
        print(f"\n[{idx + 1}/{len(videos)}] {url}", flush=True)
        try:
            bounds = segment_video(transcript, args.model, cache_dir, chunk_ms, overlap_ms)
        except Exception as e:
            print(f"  SKIPPED (segmentation error: {e})", flush=True)
            continue
        print(f"  {len(bounds)} segments", flush=True)
        normed = ensure_end_ms(transcript)
        for a, b in bounds:
            text = transcript_text(normed, a, b).strip()
            if text:
                all_segments.append({
                    "video_url": url,
                    **metadata,
                    "timecode_in_ms": a,
                    "timecode_out_ms": b,
                    "timecode_in": ms_to_timecode(a),
                    "timecode_out": ms_to_timecode(b),
                    "segment_duration_seconds": round((b - a) / 1000, 3),
                    "segment_transcript": text,
                    "topic": "",
                })

    print(f"\nTotal segments: {len(all_segments)}", flush=True)
    if not all_segments:
        die("No non-empty segments produced")

    segments_path = output_dir / "segments.json"
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_segments)} segments → {segments_path}", flush=True)

    # Write review CSV for human labeling — one row per segment
    review_path = output_dir / "segments_review.csv"
    review_fields = [
        "video_url", "title", "creator_label", "upload_date",
        "timecode_in", "timecode_out", "segment_duration_seconds",
        "topic",  # empty — for human to fill in
        "segment_transcript_preview",
        "segment_transcript",
    ]
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=review_fields, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        for seg in all_segments:
            writer.writerow({
                **seg,
                "segment_transcript_preview": seg["segment_transcript"][:300],
            })
    print(f"Saved segments_review.csv ({len(all_segments)} rows) → {review_path}", flush=True)
    print("\nNext step: open segments_review.csv and fill in the `topic` column for each row.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Segment podcast transcripts by topic shift for human labeling."
    )
    ap.add_argument("--input", required=True,
                    help="Input CSV with source_url, transcript, and duration_seconds columns")
    ap.add_argument("--output-dir", default="Top_100_segments",
                    help="Parent directory for run outputs (default: Top_100_segments). "
                         "Each run creates a timestamped subfolder inside.")
    ap.add_argument("--model", default=DEFAULT_CHAT_MODEL,
                    help=f"LLM model for segmentation boundary detection (default: {DEFAULT_CHAT_MODEL})")
    ap.add_argument("--chunk-minutes", type=int, default=DEFAULT_CHUNK_MINUTES,
                    help=f"Transcript chunk size in minutes (default: {DEFAULT_CHUNK_MINUTES})")
    ap.add_argument("--overlap-seconds", type=int, default=DEFAULT_OVERLAP_SECONDS,
                    help=f"Overlap between chunks in seconds (default: {DEFAULT_OVERLAP_SECONDS})")
    ap.add_argument("--cache-dir", default=".pipeline_cache",
                    help="Directory for caching LLM responses (default: .pipeline_cache)")
    ap.add_argument("--no-fetch-timedtext", action="store_true",
                    help="Skip fetching YouTube timedtext captions from HTML transcripts")
    ap.add_argument("--skip-segments", nargs="*", metavar="SEGMENTS_JSON",
                    help="Path(s) to existing segments.json files; videos already segmented there will be skipped")
    args = ap.parse_args()

    if args.chunk_minutes <= 0:
        die("--chunk-minutes must be greater than 0")
    if args.overlap_seconds < 0:
        die("--overlap-seconds must be 0 or greater")
    if args.overlap_seconds >= args.chunk_minutes * 60:
        die("--overlap-seconds must be smaller than --chunk-minutes")

    run(args)


if __name__ == "__main__":
    main()
