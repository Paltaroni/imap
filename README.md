# cluster_presegmented.py

Embeds, clusters, topic-labels, and scores sentiment on pre-segmented podcast/YouTube
transcript data. Designed to pick up where [transcript_segmenter.py](transcript_segmenter.py)
leaves off: give it a CSV of transcript segments and it groups them into topics, has an LLM
name each topic, and (once you've reviewed the topics) scores per-segment sentiment toward
each one.

The pipeline is split into phases so you can review and correct the LLM's work between
steps, rather than trusting one end-to-end pass.

## Input format

A CSV with at least these columns:

| column | description |
|---|---|
| `video_url` | source video URL |
| `segment_transcript` | transcript text for the segment |
| `timecode_in` | segment start, `m:ss` or `h:mm:ss` |
| `timecode_out` | segment end, `m:ss` or `h:mm:ss` |
| `segment_duration_seconds` | segment length in seconds |

Any other columns in the CSV are passed through untouched to the output.

## Setup

```bash
pip install openai numpy scikit-learn pandas python-dotenv hdbscan umap-learn
```

`hdbscan` and `umap-learn` are only required for the default `hdbscan` clustering algorithm
(see [Clustering algorithms](#clustering-algorithms) below) — skip them if you always pass
`--cluster-algorithm kmeans/spherical-kmeans/agglomerative`.

Set credentials via environment variables or a `.env` file (checked in the script's
directory, its parent, then the working directory):

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."   # optional, defaults to https://api.openai.com/v1
```

## Pipeline

### Phase 1 — cluster

```bash
python cluster_presegmented.py --phase cluster \
    --input sample.csv --output-dir out/
```

1. Loads and validates the CSV (rows missing `video_url` or `segment_transcript` are skipped).
2. Embeds each segment (`text-embedding-3-small` by default), caching embeddings on disk.
3. Clusters the embeddings (HDBSCAN by default).
4. For each cluster, samples core segments (closest to centroid) and has an LLM assign a
   `topic_label` + `definition`, then checks that label against peripheral segments
   (furthest from centroid) to flag clusters that look like a mixed bag (`coherent: false`).
5. Any segments HDBSCAN left as noise are either folded into their nearest cluster (if
   cosine similarity ≥ `--noise-floor`) or labeled individually as singleton clusters.

Writes to `--output-dir`:

- `segments.json` — every segment with its assigned `cluster_id`
- `clusters_review.csv` — one row per cluster: `proposed_label`, `definition`, confidence,
  coherence flag/note, cosine-similarity stats, and 3 sample excerpts
- `cluster_quality.csv` — cluster-level diagnostics (size, runtime, cosine stats) without
  the review/editing columns
- `topics_review.csv` — clusters rolled up by label into topics, with an empty `keep` column

**Edit `clusters_review.csv` before continuing.** Fix any `proposed_label` that's wrong, and
give two clusters the same label to merge them into one topic.

### Phase 2 — sentiment

```bash
python cluster_presegmented.py --phase sentiment --output-dir out/
```

1. Loads `segments.json` and the (edited) `clusters_review.csv` to map each segment's
   `cluster_id` to a topic label/definition.
2. If `topics_review.csv` has any rows marked `keep` (`Y`/`Yes`/`1`/`True`/`X`), only those
   topics are processed; otherwise all topics are processed.
3. For each in-scope segment, an LLM first confirms the segment is actually *about* its
   topic (`topic_confirmed`), then — only if confirmed — scores sentiment toward that topic
   (`positive`/`negative`/`neutral` with confidence and rationale).

Writes `results.csv` to `--output-dir` with one row per input segment, passthrough columns
plus `topic_label`, `topic_definition`, `topic_confirmed`, `sentiment`, `confidence`,
`rationale`.

## Supporting phases

Use these between phase 1 and phase 2 as you clean up the topic list.

### `--phase topics`

Rebuilds `topics_review.csv` from an edited `clusters_review.csv` without re-running
clustering or labeling. Use after manually editing `proposed_label` values.

```bash
python cluster_presegmented.py --phase topics --output-dir out/
```

### `--phase reclassify`

Moves specific segments to a (possibly new) topic label by hand, keyed on
`(video_url, timecode_in)`. Give it a CSV with columns `video_url`, `timecode_in`,
`topic_label`; it updates `segments.json`, appends new cluster rows to `clusters_review.csv`
as needed, and rebuilds `topics_review.csv`.

```bash
python cluster_presegmented.py --phase reclassify \
    --output-dir out/ --reclassify fixes.csv
```

## Clustering algorithms

Set via `--cluster-algorithm`:

- `hdbscan` (default) — density-based, finds its own cluster count; reduces embeddings with
  UMAP first. Leaves sparse points as noise, which phase 1 then reassigns or labels solo.
  Tune with `--min-cluster-size`, `--min-samples`, `--umap-n-components`, `--umap-n-neighbors`.
- `kmeans` / `spherical-kmeans` / `agglomerative` — fixed cluster count via `--n-clusters`.
  All operate on the embeddings after L2 (cosine) normalization.

## Caching

All embedding and LLM calls (labeling, verification, sentiment, confirmation) are cached to
disk under `--cache-dir` (default `.pipeline_cache`), keyed by a hash of the model + prompt.
Re-running a phase with unchanged inputs reuses cached results instead of re-calling the API.

## Key flags

| flag | default | purpose |
|---|---|---|
| `--phase` | `cluster` | `cluster`, `topics`, `reclassify`, or `sentiment` |
| `--input` | — | pre-segmented CSV (required for `--phase cluster`) |
| `--output-dir` | `pipeline_output` | where all phase outputs are read/written |
| `--clusters` | `<output-dir>/clusters_review.csv` | override reviewed clusters file |
| `--topics` | `<output-dir>/topics_review.csv` | override topics keep-filter file |
| `--reclassify` | — | CSV of manual segment→topic overrides (`--phase reclassify`) |
| `--cluster-algorithm` | `hdbscan` | `hdbscan`, `kmeans`, `spherical-kmeans`, `agglomerative` |
| `--n-clusters` | `300` | target cluster count for non-HDBSCAN algorithms |
| `--min-cluster-size` | `6` | HDBSCAN minimum cluster size |
| `--noise-floor` | `0.7` | cosine similarity above which noise segments join their nearest cluster |
| `--model` | `gpt-4.1-mini` | chat model for labeling/verification/sentiment/confirmation |
| `--embed-model` | `text-embedding-3-small` | embedding model |
| `--cache-dir` | `.pipeline_cache` | disk cache location |

Run `python cluster_presegmented.py --help` for the full list.
