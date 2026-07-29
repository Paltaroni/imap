#!/usr/bin/env python3
"""
cluster_presegmented.py — Embed, cluster, label, and score sentiment on pre-segmented transcript data.

Expects a CSV with at least these columns:
  video_url, segment_transcript, timecode_in, timecode_out, segment_duration_seconds

All other columns are passed through to the output.

Phase 1 (--phase cluster):
  Read pre-segmented rows → embed segments → cluster → LLM label each cluster →
  write segments.json + clusters_review.csv

  clusters_review.csv has one row per cluster. Edit proposed_label before running phase 2.
  To consolidate clusters, give them the same proposed_label.

Phase 2 (--phase sentiment):
  Load approved labels from clusters_review.csv → LLM sentiment per segment →
  write results.csv

Usage:
  python cluster_presegmented.py --phase cluster \\
      --input sample.csv --output-dir out/ [--n-clusters 50]
  # edit out/clusters_review.csv
  python cluster_presegmented.py --phase sentiment --output-dir out/

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering, KMeans

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

try:
    import pandas as pd
except ImportError:
    pd = None

csv.field_size_limit(10**7)

DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_N_CLUSTERS = 300
DEFAULT_CLUSTER_ALGORITHM = "hdbscan"
DEFAULT_MIN_CLUSTER_SIZE = 6
EMBED_BATCH_SIZE = 100
MAX_EMBED_CHARS = 2000
CLUSTER_ALGORITHMS = ["kmeans", "spherical-kmeans", "agglomerative", "hdbscan"]

_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
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


def timecode_to_ms(tc: str) -> int:
    parts = tc.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000
    elif len(parts) == 2:
        m, s = parts
        return (int(m) * 60 + int(s)) * 1000
    return 0


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
            "cluster_id": None,
        })

    print(f"Loaded {len(segments)} segments from {path}", flush=True)
    return segments


# -------------------------
# LLM helper
# -------------------------

def llm_call(
    system: str,
    user: str,
    schema: Dict[str, Any],
    model: str,
    max_tokens: int = 2048,
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
# Embedding
# -------------------------

def embed_texts(texts: List[str], model: str, cache_dir: Path) -> np.ndarray:
    cache_subdir = cache_dir / "embeddings"
    embeddings: List[Optional[List[float]]] = [None] * len(texts)
    to_embed: List[Tuple[int, str]] = []

    for i, text in enumerate(texts):
        path = cache_subdir / f"{sha256(model + '|' + text)}.json"
        cached = cache_load(path)
        if cached is not None:
            embeddings[i] = cached
        else:
            to_embed.append((i, text))

    if to_embed:
        print(f"  embedding {len(to_embed)} segments ...", flush=True)
        for start in range(0, len(to_embed), EMBED_BATCH_SIZE):
            batch = to_embed[start : start + EMBED_BATCH_SIZE]
            try:
                resp = client.embeddings.create(model=model, input=[t for _, t in batch])
                for (idx, text), emb_obj in zip(batch, resp.data):
                    vec = emb_obj.embedding
                    embeddings[idx] = vec
                    cache_save(cache_subdir / f"{sha256(model + '|' + text)}.json", vec)
            except Exception as e:
                die(f"Embedding API error: {e}")

    return np.array(embeddings, dtype=np.float32)


# -------------------------
# Clustering
# -------------------------

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms == 0, 1.0, norms)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def _compute_cluster_centers(normalized: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    centers: Dict[int, np.ndarray] = {}
    for cid in sorted(int(label) for label in set(labels.tolist()) if int(label) >= 0):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        center = normalized[idxs].mean(axis=0, dtype=np.float64)
        norm = np.linalg.norm(center)
        centers[cid] = (center / norm if norm else center).astype(np.float32)
    return centers


def _spherical_kmeans(
    normalized: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    k = min(n_clusters, len(normalized))
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(normalized)
    centers = _normalize_rows(km.cluster_centers_.astype(np.float32))

    for _ in range(max_iter):
        old_labels = labels.copy()
        labels = np.argmax(normalized @ centers.T, axis=1)
        new_centers = centers.copy()
        for cid in range(k):
            idxs = np.where(labels == cid)[0]
            if len(idxs) > 0:
                new_centers[cid] = normalized[idxs].mean(axis=0)
        centers = _normalize_rows(new_centers)
        if np.array_equal(labels, old_labels):
            break

    return labels.astype(int), {cid: centers[cid] for cid in range(k)}


def _agglomerative_cosine(normalized: np.ndarray, n_clusters: int) -> np.ndarray:
    if len(normalized) == 1:
        return np.array([0], dtype=int)
    k = min(n_clusters, len(normalized))
    try:
        clusterer = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    except TypeError:
        clusterer = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
    return clusterer.fit_predict(normalized).astype(int)


def _umap_reduce(
    normalized: np.ndarray,
    n_components: int,
    n_neighbors: int,
) -> np.ndarray:
    try:
        import umap as umap_module  # type: ignore
    except ImportError:
        die("umap-learn is not installed. Install it with: pip install umap-learn")
    n = len(normalized)
    reducer = umap_module.UMAP(
        n_components=min(n_components, n - 2),
        n_neighbors=min(n_neighbors, n - 1),
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(normalized)


def _hdbscan_labels(
    normalized: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int],
    umap_n_components: int,
    umap_n_neighbors: int,
) -> np.ndarray:
    if len(normalized) < max(2, min_cluster_size):
        return np.full(len(normalized), -1, dtype=int)
    try:
        import hdbscan  # type: ignore
    except ImportError:
        die("hdbscan is not installed. Install it with: pip install hdbscan")

    print(f"  reducing to {umap_n_components} dimensions with UMAP ...", flush=True)
    reduced = _umap_reduce(normalized, umap_n_components, umap_n_neighbors)

    kwargs: Dict[str, Any] = {
        "min_cluster_size": max(2, min_cluster_size),
        "metric": "euclidean",
        "cluster_selection_method": "leaf",
    }
    if min_samples is not None:
        kwargs["min_samples"] = max(1, min_samples)
    clusterer = hdbscan.HDBSCAN(**kwargs)
    return clusterer.fit_predict(reduced).astype(int)


def cluster_embeddings(
    embeddings: np.ndarray,
    algorithm: str,
    n_clusters: int,
    min_cluster_size: int,
    min_samples: Optional[int],
    umap_n_components: int = 15,
    umap_n_neighbors: int = 15,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    if len(embeddings) == 0:
        die("No embeddings to cluster")
    if n_clusters <= 0:
        die("--n-clusters must be greater than 0")
    normalized = normalize_embeddings(embeddings)

    if algorithm == "kmeans":
        k = min(n_clusters, len(embeddings))
        print(f"  clustering {len(embeddings)} segments into {k} clusters with KMeans ...", flush=True)
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(normalized).astype(int)
        centers = {cid: center for cid, center in enumerate(_normalize_rows(km.cluster_centers_.astype(np.float32)))}
    elif algorithm == "spherical-kmeans":
        k = min(n_clusters, len(embeddings))
        print(f"  clustering {len(embeddings)} segments into {k} clusters with spherical KMeans ...", flush=True)
        labels, centers = _spherical_kmeans(normalized, n_clusters)
    elif algorithm == "agglomerative":
        k = min(n_clusters, len(embeddings))
        print(f"  clustering {len(embeddings)} segments into {k} clusters with cosine agglomerative clustering ...", flush=True)
        labels = _agglomerative_cosine(normalized, n_clusters)
        centers = _compute_cluster_centers(normalized, labels)
    elif algorithm == "hdbscan":
        print(
            f"  clustering {len(embeddings)} segments with HDBSCAN "
            f"(min_cluster_size={min_cluster_size}, min_samples={min_samples or 'auto'}) ...",
            flush=True,
        )
        labels = _hdbscan_labels(normalized, min_cluster_size, min_samples,
                                  umap_n_components, umap_n_neighbors)
        centers = _compute_cluster_centers(normalized, labels)
        n_noise = int(np.sum(labels == -1))
        n_clusters_found = len({int(label) for label in labels.tolist() if int(label) >= 0})
        print(f"  HDBSCAN found {n_clusters_found} clusters and {n_noise} noise/outlier segments", flush=True)
    else:
        die(f"Unsupported --cluster-algorithm: {algorithm}")

    return labels, centers, normalized


def ordered_cluster_indices(
    labels: np.ndarray,
    normalized: np.ndarray,
    centers: Dict[int, np.ndarray],
    cid: int,
) -> np.ndarray:
    idxs = np.where(labels == cid)[0]
    center = centers.get(cid)
    if center is None or len(idxs) == 0:
        return idxs
    similarities = normalized[idxs] @ center
    return idxs[np.argsort(-similarities)]


def write_cluster_quality(
    path: Path,
    labels: np.ndarray,
    normalized: np.ndarray,
    centers: Dict[int, np.ndarray],
    all_segments: List[Dict[str, Any]],
    cluster_info: Dict[int, Dict[str, Any]],
    algorithm: str,
) -> None:
    cluster_ids = sorted(int(label) for label in set(labels.tolist()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_algorithm",
                "cluster_id",
                "is_noise",
                "n_segments",
                "total_runtime_seconds",
                "avg_cosine_to_centroid",
                "std_cosine_to_centroid",
                "min_cosine_to_centroid",
                "label_confidence",
                "proposed_label",
                "definition",
            ],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for cid in cluster_ids:
            idxs = np.where(labels == cid)[0]
            center = centers.get(cid)
            similarities = normalized[idxs] @ center if center is not None and len(idxs) else np.array([])
            info = cluster_info.get(cid, {})
            writer.writerow({
                "cluster_algorithm": algorithm,
                "cluster_id": cid,
                "is_noise": cid == -1,
                "n_segments": len(idxs),
                "total_runtime_seconds": round(sum(float(all_segments[i].get("segment_duration_seconds") or 0) for i in idxs), 3),
                "avg_cosine_to_centroid": round(float(np.mean(similarities)), 4) if len(similarities) else "",
                "std_cosine_to_centroid": round(float(np.std(similarities)), 4) if len(similarities) else "",
                "min_cosine_to_centroid": round(float(np.min(similarities)), 4) if len(similarities) else "",
                "label_confidence": info.get("confidence", ""),
                "proposed_label": info.get("topic_label", ""),
                "definition": info.get("definition", ""),
            })


# -------------------------
# Cluster labeling
# -------------------------

_LABEL_SYSTEM = """\
You label groups of semantically similar podcast or YouTube transcript excerpts with a topic.

Content may cover anything: politics, true crime, entertainment, health and wellness,
internet culture, sports, spirituality, Reddit content, or random YouTube phenomena.
The same labeling rules apply regardless of domain.

Label the PRIMARY SUBJECT being discussed — the person, work, phenomenon, event, or entity
that the excerpts are fundamentally about. Not the specific angle, incident, or sub-topic.

The right level is a Wikipedia article: specific enough to have its own page, broad enough
to cover the full subject rather than one incident within it.

Too narrow → correct level (across domains):
  "Trump Phone Call Controversy"      →  "Trump"
  "Iran Missile Strike Coverage"      →  "Iran War"
  "BPC-157 Healing Protocol"          →  "Peptides"
  "Mewing Jawline Exercise Guide"     →  "Looksmaxxing"
  "Ted Bundy Trial Testimony"         →  "Ted Bundy"
  "Walter White's Moral Descent"      →  "Breaking Bad"
  "AITA Family Holiday Dispute"       →  "Reddit AITA Stories"
  "Andrew Tate Kickboxing Career"     →  "Andrew Tate"
  "Rogan Spotify Contract Drama"      →  "Joe Rogan"

Too broad (no specific subject to hold an opinion about):
  "Politics", "Health", "True Crime", "Entertainment", "Internet Culture"
  Prefer the specific entity: "Ted Bundy" over "True Crime", "Trump" over "Politics"

Reaction and watchalong content:
If the excerpts contain scripted narrative dialogue (fictional character names, plot events)
mixed with commentary from hosts watching it, the subject is the film or show —
not the events depicted in it.
  Wrong: "Illegal Gun Trafficking in Central America"  (plot point in Beverly Hills Cop)
  Right:  "Beverly Hills Cop"
Name the work, not what happens in it.

Return:
  topic_label: 1-4 words, title case — the primary subject
  definition: one sentence defining what this topic covers and what sentiment toward it means
  confidence: float 0.0–1.0
    0.9–1.0  All excerpts clearly concern one identifiable subject; label names it precisely
    0.7–0.8  Most excerpts fit but there are some outliers or mild ambiguity
    0.5–0.6  Cluster seems coherent but label feels uncertain or approximate
    0.3–0.4  Mixed cluster or label is a guess
    0.0–0.2  Incoherent or unidentifiable — noise, boilerplate, or mixed content
"""

_LABEL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["topic_label", "definition", "confidence"],
    "properties": {
        "topic_label": {"type": "string"},
        "definition": {"type": "string"},
        "confidence": {"type": "number"},
    },
}

_CONSOLIDATE_SYSTEM = """\
You consolidate podcast/YouTube topic labels into a lean entity-level taxonomy.

Goal: one label per primary subject. Multiple cluster labels about the same core subject
— from any angle, incident, or subtopic — should collapse into one.

The right level is a Wikipedia article: broad enough to cover the full subject, not just
one incident within it. This applies across all content domains:

  "Trump" not "Trump Phone Call" or "Trump Immigration Policy"
  "Iran War" not "Iran Missile Strike" or "Iran Nuclear Talks"
  "Peptides" not "BPC-157 Benefits" or "Peptide Dosing Protocol"
  "Looksmaxxing" not "Mewing Technique" or "Jawline Exercises"
  "Ted Bundy" not "Bundy Trial" or "Bundy Escape Attempts"
  "Breaking Bad" not "Walter White Arc" or "Heisenberg Identity"
  "Andrew Tate" not "Tate Kickboxing" or "Tate Arrest Coverage"
  "Reddit AITA Stories" not "AITA Holiday Dispute" or "AITA Roommate Story"

Rules:
- Merge all clusters about the same primary person, entity, phenomenon, or work
- Collapse incident/angle/subtopic labels up to the primary subject
- Keep distinctions only when topics are genuinely different subjects:
    "Trump" and "Biden" stay separate
    "Iran War" and "Israel-Gaza War" stay separate
    "Ted Bundy" and "Jeffrey Dahmer" stay separate
- Canonical label: 1-4 words, title case
- Each merge group must contain at least 2 cluster ids

Return only the groups that need merging. Topics not listed are kept as-is.
"""

_CONSOLIDATE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["merges"],
    "properties": {
        "merges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["canonical_label", "canonical_definition", "cluster_ids"],
                "properties": {
                    "canonical_label": {"type": "string"},
                    "canonical_definition": {"type": "string"},
                    "cluster_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                    },
                },
            },
        }
    },
}


def consolidate_labels(
    cluster_info: Dict[int, Dict[str, Any]],
    model: str,
    cache_dir: Path,
) -> Dict[int, Dict[str, Any]]:
    topics_input = [
        {"id": cid, "label": info.get("topic_label", ""), "definition": info.get("definition", "")}
        for cid, info in sorted(cluster_info.items())
        if info.get("topic_label")
    ]
    if len(topics_input) < 2:
        return cluster_info

    user_msg = json.dumps({"topics": topics_input}, ensure_ascii=False)
    cache_key = sha256(f"{model}|{_CONSOLIDATE_SYSTEM}|{user_msg}")
    cache_path = cache_dir / "consolidation" / f"{cache_key}.json"
    cached = cache_load(cache_path)
    if cached is not None:
        result = cached
    else:
        result = llm_call(_CONSOLIDATE_SYSTEM, user_msg, _CONSOLIDATE_SCHEMA,
                          model=model, max_tokens=2048)
        cache_save(cache_path, result)

    updated = {cid: dict(info) for cid, info in cluster_info.items()}
    n_merged = 0
    n_groups = 0
    for group_idx, merge in enumerate(result.get("merges", []), start=1):
        canonical_label = merge.get("canonical_label", "")
        canonical_definition = merge.get("canonical_definition", "")
        affected = [cid for cid in merge.get("cluster_ids", []) if cid in updated]
        if len(affected) < 2:
            continue
        n_groups += 1
        print(f"  group {group_idx}: {canonical_label!r}", flush=True)
        for cid in affected:
            original = updated[cid].get("topic_label", "")
            updated[cid]["topic_label"] = canonical_label
            updated[cid]["definition"] = canonical_definition
            updated[cid]["consolidated_from"] = original
            updated[cid]["consolidated_group"] = group_idx
            n_merged += 1
            if original != canonical_label:
                print(f"    {cid}: {original!r} → {canonical_label!r}", flush=True)
            else:
                print(f"    {cid}: same label, marked as duplicate", flush=True)
    print(f"  {n_groups} merge groups, {n_merged} cluster slots consolidated", flush=True)
    return updated


def label_cluster(sample_texts: List[str], model: str, cache_dir: Path) -> Dict[str, str]:
    user_msg = "\n\n---\n\n".join(
        f"Excerpt {i + 1}:\n{t[:600]}" for i, t in enumerate(sample_texts)
    )
    cache_key = sha256(f"{model}|{_LABEL_SYSTEM}|{user_msg}")
    cache_path = cache_dir / "labels" / f"{cache_key}.json"
    cached = cache_load(cache_path)
    if cached is not None:
        return cached
    result = llm_call(
        system=_LABEL_SYSTEM,
        user=user_msg,
        schema=_LABEL_SCHEMA,
        model=model,
        max_tokens=256,
    )
    cache_save(cache_path, result)
    return result


_VERIFY_SYSTEM = """\
You verify whether a proposed topic label accurately represents the peripheral segments of a cluster.

You are given:
- A proposed topic label and its definition (derived from the cluster's core/closest segments)
- Several excerpts from the periphery of that cluster (the segments least similar to the cluster center)

Assess whether these peripheral segments genuinely belong to the same topic, or whether they suggest
the cluster is a catch-all / mixed bag.

Return:
  coherent: true if peripheral segments reasonably fit the label, false if they reveal a mixed cluster
  coherence_note: one sentence — either confirming the fit, or briefly describing what the peripheral
                  content actually is
"""

_VERIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["coherent", "coherence_note"],
    "properties": {
        "coherent": {"type": "boolean"},
        "coherence_note": {"type": "string"},
    },
}


def verify_label(
    topic_label: str,
    definition: str,
    peripheral_texts: List[str],
    model: str,
    cache_dir: Path,
) -> Dict[str, Any]:
    excerpts = "\n\n---\n\n".join(
        f"Excerpt {i + 1}:\n{t[:600]}" for i, t in enumerate(peripheral_texts)
    )
    user_msg = (
        f"Proposed label: {topic_label}\n"
        f"Definition: {definition}\n\n"
        f"Peripheral segments:\n\n{excerpts}"
    )
    cache_key = sha256(f"{model}|{_VERIFY_SYSTEM}|{user_msg}")
    cache_path = cache_dir / "verify" / f"{cache_key}.json"
    cached = cache_load(cache_path)
    if cached is not None:
        return cached
    result = llm_call(
        system=_VERIFY_SYSTEM,
        user=user_msg,
        schema=_VERIFY_SCHEMA,
        model=model,
        max_tokens=128,
    )
    cache_save(cache_path, result)
    return result


# -------------------------
# Sentiment
# -------------------------

_SENT_SYSTEM = """\
You assess sentiment in podcast transcript segments toward a specific named topic.

Assess how the speaker(s) feel specifically about the given topic — not the overall tone.
A segment may be broadly negative but neutral or positive toward a specific subtopic.
Use the supplied topic definition to disambiguate scope. If the segment mentions the
label but does not address the defined topic, classify it as neutral with low confidence.

Return:
  sentiment: "positive", "negative", or "neutral"
  confidence: float 0.0–1.0
  rationale: one sentence explaining your assessment
"""

_SENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sentiment", "confidence", "rationale"],
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
}


def analyze_sentiment(
    segment_text: str, topic_label: str, topic_definition: str, model: str, cache_dir: Path
) -> Dict[str, Any]:
    definition = topic_definition.strip() or "(no reviewed definition provided)"
    user_msg = (
        f"topic: {topic_label}\n"
        f"topic_definition: {definition}\n\n"
        f"transcript:\n{segment_text[:3000]}"
    )
    cache_key = sha256(f"{model}|{_SENT_SYSTEM}|{user_msg}")
    cache_path = cache_dir / "sentiment" / f"{cache_key}.json"
    cached = cache_load(cache_path)
    if cached is not None:
        return cached
    result = llm_call(
        system=_SENT_SYSTEM,
        user=user_msg,
        schema=_SENT_SCHEMA,
        model=model,
        max_tokens=256,
    )
    cache_save(cache_path, result)
    return result


# -------------------------
# Confirmation
# -------------------------

_CONFIRM_SYSTEM = """\
You check whether a podcast or YouTube transcript segment is primarily about a specific topic.

Answer yes if the segment's main subject is the given topic.
Answer no if the topic is only mentioned in passing, or if the segment is primarily about something else.
"""

_CONFIRM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["confirmed"],
    "properties": {
        "confirmed": {"type": "boolean"},
    },
}


def confirm_segment(
    segment_text: str, topic_label: str, topic_definition: str, model: str, cache_dir: Path
) -> bool:
    definition = topic_definition.strip() or "(no definition provided)"
    user_msg = (
        f"topic: {topic_label}\n"
        f"topic_definition: {definition}\n\n"
        f"transcript:\n{segment_text[:3000]}"
    )
    cache_key = sha256(f"{model}|{_CONFIRM_SYSTEM}|{user_msg}")
    cache_path = cache_dir / "confirmation" / f"{cache_key}.json"
    cached = cache_load(cache_path)
    if cached is not None:
        return bool(cached.get("confirmed", False))
    result = llm_call(
        system=_CONFIRM_SYSTEM,
        user=user_msg,
        schema=_CONFIRM_SCHEMA,
        model=model,
        max_tokens=64,
    )
    cache_save(cache_path, result)
    return bool(result.get("confirmed", False))


# -------------------------
# Topics review output
# -------------------------

def write_topics_review(
    path: Path,
    labels: np.ndarray,
    all_segments: List[Dict[str, Any]],
    cluster_info: Dict[int, Dict[str, Any]],
    normalized: np.ndarray,
    centers: Dict[int, np.ndarray],
) -> None:
    topic_clusters: Dict[str, List[int]] = {}
    topic_definition: Dict[str, str] = {}

    for cid, info in cluster_info.items():
        label = info.get("topic_label", "")
        if not label or cid < 0:
            continue
        topic_clusters.setdefault(label, []).append(cid)
        topic_definition[label] = info.get("definition", "")

    rows = []
    for label, cids in topic_clusters.items():
        all_idxs: List[int] = []
        for cid in cids:
            all_idxs.extend(int(i) for i in np.where(labels == cid)[0])

        total_runtime = sum(
            float(all_segments[i].get("segment_duration_seconds") or 0) for i in all_idxs
        )

        largest_cid = max(cids, key=lambda c: int(np.sum(labels == c)))
        order = ordered_cluster_indices(labels, normalized, centers, largest_cid)
        samples = [all_segments[i]["segment_transcript"][:300] for i in order[:3]]
        while len(samples) < 3:
            samples.append("")

        rows.append({
            "topic_label": label,
            "n_segments": len(all_idxs),
            "total_runtime_minutes": round(total_runtime / 60, 1),
            "keep": "",
            "definition": topic_definition.get(label, ""),
            "sample_1": samples[0],
            "sample_2": samples[1],
            "sample_3": samples[2],
        })

    rows.sort(key=lambda r: r["n_segments"], reverse=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["topic_label", "n_segments", "total_runtime_minutes", "keep",
                        "definition", "sample_1", "sample_2", "sample_3"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)


# -------------------------
# Phase 1: run_cluster
# -------------------------

def run_cluster(args: argparse.Namespace) -> None:
    if not _api_key:
        die("OPENAI_API_KEY not set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    print(f"Loading pre-segmented CSV: {args.input}", flush=True)
    all_segments = load_presegmented_csv(args.input)
    if not all_segments:
        die("No valid segments found in CSV")

    # Embed
    texts = [s["segment_transcript"][:MAX_EMBED_CHARS] for s in all_segments]
    embeddings = embed_texts(texts, args.embed_model, cache_dir)

    # Cluster
    labels, centers, normalized = cluster_embeddings(
        embeddings=embeddings,
        algorithm=args.cluster_algorithm,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        umap_n_components=args.umap_n_components,
        umap_n_neighbors=args.umap_n_neighbors,
    )

    # Handle noise: reassign segments above floor to nearest cluster
    n_noise_raw = int(np.sum(labels == -1))
    if n_noise_raw > 0 and centers:
        cluster_id_list = sorted(centers.keys())
        centroid_matrix = np.stack([centers[cid] for cid in cluster_id_list])
        noise_idxs = np.where(labels == -1)[0]
        sims = normalized[noise_idxs] @ centroid_matrix.T
        best_sim = sims.max(axis=1)
        best_col = sims.argmax(axis=1)
        n_reassigned = 0
        for i, idx in enumerate(noise_idxs):
            if best_sim[i] >= args.noise_floor:
                labels[idx] = cluster_id_list[best_col[i]]
                n_reassigned += 1
        n_solo = int(np.sum(labels == -1))
        print(
            f"  Noise: {n_noise_raw} total → {n_reassigned} reassigned to nearest cluster "
            f"(≥{args.noise_floor}), {n_solo} below floor for individual labeling",
            flush=True,
        )

    for i, seg in enumerate(all_segments):
        seg["cluster_id"] = int(labels[i])

    segments_path = output_dir / "segments.json"
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_segments)} segments → {segments_path}", flush=True)

    # Label clusters
    cluster_ids = sorted(int(label) for label in set(labels.tolist()))
    labelable_cluster_ids = [cid for cid in cluster_ids if cid >= 0]
    print(f"\nLabeling {len(labelable_cluster_ids)} clusters ...", flush=True)

    cluster_info: Dict[int, Dict[str, Any]] = {}
    for cid in labelable_cluster_ids:
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        order = ordered_cluster_indices(labels, normalized, centers, cid)
        core_samples = [all_segments[i]["segment_transcript"] for i in order[:5]]
        info = label_cluster(core_samples, args.model, cache_dir)
        # Verify label holds against peripheral (furthest-from-centroid) segments
        n_peripheral = max(3, min(5, len(order) // 10))
        peripheral_samples = [all_segments[i]["segment_transcript"] for i in order[-n_peripheral:]]
        verification = verify_label(
            info["topic_label"], info.get("definition", ""),
            peripheral_samples, args.model, cache_dir,
        )
        info["coherent"] = verification.get("coherent", True)
        info["coherence_note"] = verification.get("coherence_note", "")
        cluster_info[cid] = info
        coherence_flag = "" if info["coherent"] else " [MIXED]"
        print(f"  {cid:3d} ({len(idxs):4d} segs): {info['topic_label']}{coherence_flag}", flush=True)

    # print(f"\nConsolidating {len(cluster_info)} topic labels ...", flush=True)
    # cluster_info = consolidate_labels(cluster_info, args.model, cache_dir)

    quality_path = output_dir / "cluster_quality.csv"
    write_cluster_quality(
        path=quality_path,
        labels=labels,
        normalized=normalized,
        centers=centers,
        all_segments=all_segments,
        cluster_info=cluster_info,
        algorithm=args.cluster_algorithm,
    )
    print(f"Wrote {quality_path}", flush=True)

    # Write clusters_review.csv
    _REVIEW_FIELDS = [
        "cluster_id", "is_noise", "consolidated_group", "proposed_label",
        "label_confidence", "coherent", "coherence_note",
        "avg_cosine", "std_cosine", "definition",
        "n_segments", "sample_1", "sample_2", "sample_3",
    ]

    review_path = output_dir / "clusters_review.csv"
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_REVIEW_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for cid in cluster_ids:
            idxs = np.where(labels == cid)[0]
            if len(idxs) == 0:
                continue
            center = centers.get(cid)
            similarities = normalized[idxs] @ center if center is not None and len(idxs) else np.array([])
            order = ordered_cluster_indices(labels, normalized, centers, cid)
            samples = [all_segments[i]["segment_transcript"][:300] for i in order[:3]]
            while len(samples) < 3:
                samples.append("")
            if cid == -1:
                info = {
                    "topic_label": "",
                    "definition": "Noise/outlier segments not assigned to a dense topic cluster.",
                    "confidence": "",
                    "coherent": "",
                    "coherence_note": "",
                }
            else:
                info = cluster_info.get(cid, {"topic_label": "unknown", "definition": "", "confidence": ""})
            writer.writerow({
                "cluster_id": cid,
                "is_noise": cid == -1,
                "consolidated_group": info.get("consolidated_group", ""),
                "proposed_label": info["topic_label"],
                "label_confidence": info.get("confidence", ""),
                "coherent": info.get("coherent", ""),
                "coherence_note": info.get("coherence_note", ""),
                "avg_cosine": round(float(np.mean(similarities)), 4) if len(similarities) else "",
                "std_cosine": round(float(np.std(similarities)), 4) if len(similarities) else "",
                "definition": info["definition"],
                "n_segments": len(idxs),
                "sample_1": samples[0],
                "sample_2": samples[1],
                "sample_3": samples[2],
            })

    print(f"\nWrote {review_path}", flush=True)

    # Label remaining noise segments individually
    solo_noise_idxs = np.where(labels == -1)[0]
    if len(solo_noise_idxs) > 0:
        next_cid = max(cluster_ids) + 1 if cluster_ids else 0
        print(f"\nLabeling {len(solo_noise_idxs)} individual noise segments (ids {next_cid}+) ...", flush=True)
        for pos, idx in enumerate(solo_noise_idxs):
            new_cid = next_cid + pos
            labels[idx] = new_cid
            all_segments[idx]["cluster_id"] = new_cid
            centers[new_cid] = normalized[idx].copy()
            text = all_segments[idx]["segment_transcript"]
            info = label_cluster([text], args.model, cache_dir)
            cluster_info[new_cid] = info
        print(f"  Done. {len(solo_noise_idxs)} segments labeled individually.", flush=True)
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(all_segments, f, ensure_ascii=False, indent=2)
        noise_cluster_ids = sorted(cluster_info.keys() - set(labelable_cluster_ids))
        with open(review_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_REVIEW_FIELDS, quoting=csv.QUOTE_ALL)
            for cid in noise_cluster_ids:
                idxs = np.where(labels == cid)[0]
                if len(idxs) == 0:
                    continue
                center = centers.get(cid)
                similarities = normalized[idxs] @ center if center is not None and len(idxs) else np.array([])
                order = ordered_cluster_indices(labels, normalized, centers, cid)
                samples = [all_segments[i]["segment_transcript"][:300] for i in order[:3]]
                while len(samples) < 3:
                    samples.append("")
                info = cluster_info.get(cid, {"topic_label": "unknown", "definition": "", "confidence": ""})
                writer.writerow({
                    "cluster_id": cid,
                    "is_noise": True,
                    "consolidated_group": info.get("consolidated_group", ""),
                    "proposed_label": info["topic_label"],
                    "label_confidence": info.get("confidence", ""),
                    "coherent": info.get("coherent", ""),
                    "coherence_note": info.get("coherence_note", ""),
                    "avg_cosine": round(float(np.mean(similarities)), 4) if len(similarities) else "",
                    "std_cosine": round(float(np.std(similarities)), 4) if len(similarities) else "",
                    "definition": info["definition"],
                    "n_segments": len(idxs),
                    "sample_1": samples[0],
                    "sample_2": samples[1],
                    "sample_3": samples[2],
                })
        print(f"Appended {len(noise_cluster_ids)} noise rows → {review_path}", flush=True)

    topics_review_path = output_dir / "topics_review.csv"
    write_topics_review(
        path=topics_review_path,
        labels=labels,
        all_segments=all_segments,
        cluster_info=cluster_info,
        normalized=normalized,
        centers=centers,
    )
    print(f"Wrote {topics_review_path}", flush=True)
    print(f"\nNext: open topics_review.csv, mark 'Y' in the keep column for topics to analyze,")
    print(f"then run: --phase sentiment --output-dir {args.output_dir}")


# -------------------------
# Phase 2: run_sentiment
# -------------------------

def run_sentiment(args: argparse.Namespace) -> None:
    if not _api_key:
        die("OPENAI_API_KEY not set")

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    segments_path = output_dir / "segments.json"
    if not segments_path.exists():
        die(f"segments.json not found in {output_dir} — run --phase cluster first")
    with open(segments_path, encoding="utf-8") as f:
        all_segments: List[Dict[str, Any]] = json.load(f)

    review_path = Path(args.clusters) if args.clusters else output_dir / "clusters_review.csv"
    if not review_path.exists():
        die(f"Clusters review file not found: {review_path}")

    cluster_to_topic: Dict[int, Dict[str, str]] = {}
    with open(review_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["cluster_id"])
                label = (row.get("proposed_label") or "").strip()
                definition = (row.get("definition") or "").strip()
                if label:
                    cluster_to_topic[cid] = {"label": label, "definition": definition}
            except Exception:
                continue

    print(f"Loaded {len(cluster_to_topic)} cluster labels from {review_path}", flush=True)

    # Load topics_review.csv for keep filter (optional — if absent, process all topics)
    kept_topics: Optional[set] = None
    topics_path = Path(args.topics) if args.topics else output_dir / "topics_review.csv"
    if topics_path.exists():
        with open(topics_path, newline="", encoding="utf-8") as f:
            kept_topics = {
                row["topic_label"].strip()
                for row in csv.DictReader(f)
                if row.get("keep", "").strip().upper() in {"Y", "YES", "1", "TRUE", "X"}
            }
        if kept_topics:
            print(f"Filtering to {len(kept_topics)} kept topics from {topics_path}", flush=True)
        else:
            print(f"[warn] topics_review.csv found but no topics marked keep — processing all", flush=True)
            kept_topics = None
    else:
        print("No topics_review.csv found — processing all topics", flush=True)

    n = len(all_segments)
    results: List[Dict[str, Any]] = []

    for i, seg in enumerate(all_segments):
        cid = seg.get("cluster_id")
        try:
            cid_int = int(cid) if cid is not None else None
        except Exception:
            cid_int = None
        topic_info = cluster_to_topic.get(cid_int, {}) if cid_int is not None else {}
        topic_label = topic_info.get("label", "")
        topic_definition = topic_info.get("definition", "")
        text = seg.get("segment_transcript", "").strip()

        topic_confirmed: Optional[bool] = None
        sentiment_result: Dict[str, Any] = {"sentiment": None, "confidence": None, "rationale": ""}

        if text and topic_label:
            if kept_topics is not None and topic_label not in kept_topics:
                pass  # topic not selected — leave sentiment null
            else:
                print(
                    f"  [{i + 1}/{n}] {seg['timecode_in']}–{seg['timecode_out']} "
                    f"cluster={cid} \"{topic_label}\" ...",
                    end=" ", flush=True,
                )
                topic_confirmed = confirm_segment(text, topic_label, topic_definition, args.model, cache_dir)
                if topic_confirmed:
                    sentiment_result = analyze_sentiment(text, topic_label, topic_definition, args.model, cache_dir)
                    print(f"confirmed → {sentiment_result['sentiment']}", flush=True)
                else:
                    print("not confirmed", flush=True)

        _skip = {"timecode_in_ms", "timecode_out_ms", "timecode_in", "timecode_out",
                 "segment_duration_seconds", "segment_transcript", "cluster_id"}
        results.append({
            **{k: v for k, v in seg.items() if k not in _skip},
            "timecode_in_ms": seg["timecode_in_ms"],
            "timecode_out_ms": seg["timecode_out_ms"],
            "timecode_in": seg["timecode_in"],
            "timecode_out": seg["timecode_out"],
            "segment_duration_seconds": seg.get("segment_duration_seconds", round((seg["timecode_out_ms"] - seg["timecode_in_ms"]) / 1000, 3)),
            "cluster_id": cid,
            "topic_label": topic_label,
            "topic_definition": topic_definition,
            "topic_confirmed": topic_confirmed,
            "sentiment": sentiment_result.get("sentiment"),
            "confidence": sentiment_result.get("confidence"),
            "rationale": sentiment_result.get("rationale"),
            "segment_transcript": text,
        })

    results_path = output_dir / "results.csv"
    if pd is not None:
        pd.DataFrame(results).to_csv(results_path, index=False)
    else:
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)

    n_scored = sum(1 for r in results if r["sentiment"] is not None)
    print(f"\nWrote {len(results)} rows ({n_scored} scored) → {results_path}", flush=True)


# -------------------------
# Phase: rebuild topics_review from edited clusters_review
# -------------------------

def run_topics(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)

    segments_path = output_dir / "segments.json"
    if not segments_path.exists():
        die(f"segments.json not found in {output_dir}")
    with open(segments_path, encoding="utf-8") as f:
        all_segments: List[Dict[str, Any]] = json.load(f)

    seg_by_cluster: Dict[int, List[Dict[str, Any]]] = {}
    for seg in all_segments:
        cid = seg.get("cluster_id")
        try:
            cid_int = int(cid) if cid is not None else None
        except Exception:
            cid_int = None
        if cid_int is not None:
            seg_by_cluster.setdefault(cid_int, []).append(seg)

    review_path = Path(args.clusters) if args.clusters else output_dir / "clusters_review.csv"
    if not review_path.exists():
        die(f"Clusters review file not found: {review_path}")

    topic_info: Dict[str, Dict[str, Any]] = {}
    with open(review_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["cluster_id"])
            except Exception:
                continue
            label = (row.get("proposed_label") or "").strip()
            definition = (row.get("definition") or "").strip()
            if not label:
                continue
            if label not in topic_info:
                topic_info[label] = {"definition": definition, "cluster_ids": [], "samples": []}
            topic_info[label]["cluster_ids"].append(cid)
            for k in ["sample_1", "sample_2", "sample_3"]:
                s = (row.get(k) or "").strip()
                if s and len(topic_info[label]["samples"]) < 3:
                    topic_info[label]["samples"].append(s)

    rows = []
    for label, info in topic_info.items():
        segs = []
        for cid in info["cluster_ids"]:
            segs.extend(seg_by_cluster.get(cid, []))
        total_runtime = sum(float(s.get("segment_duration_seconds") or 0) for s in segs)
        samples = info["samples"][:3]
        while len(samples) < 3:
            samples.append("")
        rows.append({
            "topic_label": label,
            "n_segments": len(segs),
            "total_runtime_minutes": round(total_runtime / 60, 1),
            "keep": "",
            "definition": info["definition"],
            "sample_1": samples[0],
            "sample_2": samples[1],
            "sample_3": samples[2],
        })

    rows.sort(key=lambda r: r["n_segments"], reverse=True)

    topics_path = Path(args.topics) if args.topics else output_dir / "topics_review.csv"
    with open(topics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["topic_label", "n_segments", "total_runtime_minutes", "keep",
                        "definition", "sample_1", "sample_2", "sample_3"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} topics → {topics_path}", flush=True)


# -------------------------
# Phase: reclassify individual segments
# -------------------------

def run_reclassify(args: argparse.Namespace) -> None:
    reclassify_path = Path(args.reclassify)
    if not reclassify_path.exists():
        die(f"Reclassify file not found: {reclassify_path}")

    output_dir = Path(args.output_dir)

    segments_path = output_dir / "segments.json"
    if not segments_path.exists():
        die(f"segments.json not found in {output_dir}")
    with open(segments_path, encoding="utf-8") as f:
        all_segments: List[Dict[str, Any]] = json.load(f)

    review_path = Path(args.clusters) if args.clusters else output_dir / "clusters_review.csv"
    if not review_path.exists():
        die(f"Clusters review file not found: {review_path}")

    cluster_rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    with open(review_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        cluster_rows = list(reader)

    # label → first cluster_id for that label
    label_to_cid: Dict[str, int] = {}
    max_cid = -1
    for row in cluster_rows:
        try:
            cid = int(row["cluster_id"])
            max_cid = max(max_cid, cid)
        except Exception:
            continue
        label = (row.get("proposed_label") or "").strip()
        if label and label not in label_to_cid:
            label_to_cid[label] = cid

    # (video_url, timecode_in) → segment index
    seg_lookup: Dict[Tuple[str, str], int] = {}
    for idx, seg in enumerate(all_segments):
        key = (seg.get("video_url", "").strip(), seg.get("timecode_in", "").strip())
        seg_lookup[key] = idx

    with open(reclassify_path, newline="", encoding="utf-8") as f:
        reclassify_rows = list(csv.DictReader(f))

    n_updated = 0
    new_cluster_rows: List[Dict[str, str]] = []

    for i, row in enumerate(reclassify_rows):
        video_url = (row.get("video_url") or "").strip()
        timecode_in = (row.get("timecode_in") or "").strip()
        topic_label = (row.get("topic_label") or "").strip()

        if not video_url or not timecode_in or not topic_label:
            print(f"  [row {i + 1}] skipping — missing fields", flush=True)
            continue

        seg_idx = seg_lookup.get((video_url, timecode_in))
        if seg_idx is None:
            print(f"  [row {i + 1}] no segment found for {video_url} @ {timecode_in}", flush=True)
            continue

        if topic_label in label_to_cid:
            target_cid = label_to_cid[topic_label]
        else:
            max_cid += 1
            target_cid = max_cid
            label_to_cid[topic_label] = target_cid
            new_row = {fn: "" for fn in fieldnames}
            new_row["cluster_id"] = str(target_cid)
            new_row["is_noise"] = "False"
            new_row["proposed_label"] = topic_label
            new_row["n_segments"] = "1"
            new_cluster_rows.append(new_row)
            print(f"  New cluster {target_cid}: '{topic_label}'", flush=True)

        old_cid = all_segments[seg_idx].get("cluster_id")
        all_segments[seg_idx]["cluster_id"] = target_cid
        n_updated += 1
        print(
            f"  {video_url} @ {timecode_in}: cluster {old_cid} → {target_cid} '{topic_label}'",
            flush=True,
        )

    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"Updated {n_updated} segments → {segments_path}", flush=True)

    if new_cluster_rows:
        with open(review_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writerows(new_cluster_rows)
        print(f"Appended {len(new_cluster_rows)} new cluster rows → {review_path}", flush=True)

    run_topics(args)


# -------------------------
# Main
# -------------------------

def validate_args(args: argparse.Namespace) -> None:
    if args.n_clusters <= 0:
        die("--n-clusters must be greater than 0")
    if args.cluster_algorithm not in CLUSTER_ALGORITHMS:
        die(f"--cluster-algorithm must be one of: {', '.join(CLUSTER_ALGORITHMS)}")
    if args.min_cluster_size <= 1:
        die("--min-cluster-size must be greater than 1")
    if args.min_samples is not None and args.min_samples <= 0:
        die("--min-samples must be greater than 0 when provided")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Embed, cluster, label, and score sentiment on pre-segmented transcript data."
    )
    ap.add_argument("--phase", choices=["cluster", "topics", "reclassify", "sentiment"], default="cluster",
                    help="cluster: embed/cluster/label. topics: rebuild topics_review from edited clusters_review. reclassify: reassign specific segments to a topic. sentiment: confirm + score sentiment.")
    ap.add_argument("--input", help="Pre-segmented CSV with video_url and segment_transcript columns (required for cluster phase)")
    ap.add_argument("--output-dir", default="pipeline_output",
                    help="Directory for segments.json, clusters_review.csv, results.csv")
    ap.add_argument("--clusters", default=None,
                    help="Path to reviewed clusters_review.csv (defaults to output-dir/clusters_review.csv)")
    ap.add_argument("--topics", default=None,
                    help="Path to topics_review.csv with keep column filled in (defaults to output-dir/topics_review.csv)")
    ap.add_argument("--reclassify", default=None,
                    help="Path to reclassify.csv (columns: video_url, timecode_in, topic_label) for --phase reclassify")
    ap.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS,
                    help=f"Cluster count for kmeans, spherical-kmeans, and agglomerative (default: {DEFAULT_N_CLUSTERS})")
    ap.add_argument("--cluster-algorithm", choices=CLUSTER_ALGORITHMS, default=DEFAULT_CLUSTER_ALGORITHM,
                    help=f"Clustering algorithm to use (default: {DEFAULT_CLUSTER_ALGORITHM})")
    ap.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE,
                    help=f"HDBSCAN minimum cluster size (default: {DEFAULT_MIN_CLUSTER_SIZE})")
    ap.add_argument("--min-samples", type=int, default=None,
                    help="HDBSCAN min_samples; defaults to HDBSCAN's automatic setting")
    ap.add_argument("--umap-n-components", type=int, default=50,
                    help="UMAP target dimensions before HDBSCAN (default: 50)")
    ap.add_argument("--umap-n-neighbors", type=int, default=15,
                    help="UMAP neighborhood size (default: 15)")
    ap.add_argument("--noise-floor", type=float, default=0.7,
                    help="Cosine similarity floor for assigning noise segments to nearest cluster (default: 0.7)")
    ap.add_argument("--model", default=DEFAULT_CHAT_MODEL,
                    help=f"Chat model for labeling and sentiment (default: {DEFAULT_CHAT_MODEL})")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help=f"Embedding model (default: {DEFAULT_EMBED_MODEL})")
    ap.add_argument("--cache-dir", default=".pipeline_cache",
                    help="Directory for caching LLM and embedding responses")
    ap.add_argument("--label-sample-pct", type=float, default=0.10,
                    help="Fraction of each cluster to sample for LLM labeling (default: 0.10)")
    ap.add_argument("--label-sample-min", type=int, default=5,
                    help="Minimum samples sent to LLM per cluster (default: 5)")
    ap.add_argument("--label-sample-max", type=int, default=30,
                    help="Maximum samples sent to LLM per cluster (default: 30)")
    args = ap.parse_args()
    validate_args(args)

    if args.phase == "cluster":
        if not args.input:
            die("--input is required for --phase cluster")
        run_cluster(args)
    elif args.phase == "topics":
        run_topics(args)
    elif args.phase == "reclassify":
        if not args.reclassify:
            die("--reclassify is required for --phase reclassify")
        run_reclassify(args)
    else:
        run_sentiment(args)


if __name__ == "__main__":
    main()
