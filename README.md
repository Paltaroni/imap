# Podcast transcript segmentation, topic labeling, and subtopic sentiment

Three command-line scripts that turn long-form video transcripts (built for YouTube podcasts) into topic-level data you can analyze:

1. **`transcript_segmenter.py`** splits each transcript into segments wherever the topic changes.
2. **`label_from_codebook.py`** assigns every segment exactly one topic from a codebook you provide.
3. **`label_subtopics.py`** takes the segments for one topic and scores, for each subtopic you define, whether it is discussed and with what sentiment.

All three use an OpenAI-compatible chat model with structured (JSON-schema) output, cache every response on disk, and write plain CSV/JSON you can open in a spreadsheet or load with pandas.

```
videos.csv ──► transcript_segmenter.py ──► segments_review.csv
                                                 │
                                                 ▼
                      codebook.csv ──► label_from_codebook.py ──► labeled_segments.csv
                                                                        │
                                                                        ▼
                                 subtopics.csv ──► label_subtopics.py ──► <topic>_subtopics_labeled.csv
```

---

## Setup

Requires Python 3.9+.

```bash
pip install openai python-dotenv
```

`python-dotenv` is optional; it lets the scripts read credentials from a `.env` file.

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
# optional: any OpenAI-compatible endpoint (a proxy, Azure, a local server)
export OPENAI_BASE_URL="https://..."
```

The scripts also look for a `.env` file in their own folder or the folder above it, with the same two variables. **Do not commit your `.env` file.**

Every script has a `--model` option. The default is `gpt-4.1-mini`. The model must support JSON-schema structured output.

---

## Step 1: Segment transcripts by topic (`transcript_segmenter.py`)

The script finds topic-shift boundaries in each transcript. It does **not** name the topics; that is step 2.

### Input

A CSV with one row per video and these columns:

| Column | Required | Notes |
|---|---|---|
| `source_url` | yes | Video URL; becomes `video_url` in the output |
| `transcript` | yes | Transcript in any of the formats below |
| `duration_seconds` | recommended | Video length; used to place timestamps when the transcript has none |

Any other columns (title, channel, upload date, view count, …) are copied onto every segment from that video.

Accepted transcript formats:

- **JSON list** of timed records, e.g. `[{"start_ms": 0, "end_ms": 4200, "text": "..."}]`. Common field names for start, end and duration in seconds or milliseconds are recognized.
- **WebVTT** with word-level timestamps (YouTube auto-captions). Words are grouped into utterances of about 5 seconds.
- **YouTube page HTML.** The script fetches the timed captions from it. Turn this off with `--no-fetch-timedtext`.
- **Plain text.** Split into sentences, with timestamps spread evenly across `duration_seconds`. Timecodes for these videos are estimates.

### How it works

Each transcript becomes a time-ordered list of utterances (`start_ms`, `end_ms`, `text`). The video is processed in chunks (10 minutes by default, overlapping by 15 seconds). Within a chunk, utterances are merged into blocks of about 8 seconds and sent to the model, which returns segment boundaries in milliseconds. Boundaries from neighboring chunks are reconciled. Each segment's text is every utterance that overlaps its time window.

### Run

```bash
python transcript_segmenter.py --input videos.csv --output-dir segments_out
```

| Option | Default | Description |
|---|---|---|
| `--output-dir` | `segments_out` | Parent folder; each run writes to a new timestamped subfolder, so earlier runs are never overwritten |
| `--model` | `gpt-4.1-mini` | Model used to find boundaries |
| `--chunk-minutes` | `10` | Transcript chunk size sent to the model |
| `--overlap-seconds` | `15` | Overlap between chunks |
| `--cache-dir` | `.pipeline_cache` | Where model responses are cached |
| `--no-fetch-timedtext` | off | Don't fetch captions for HTML transcripts |
| `--skip-segments` | — | One or more earlier `segments.json` files; videos already in them are skipped |

### Output (in `segments_out/<timestamp>_<input name>/`)

- **`segments.json`**: every segment with all of its video's metadata, `segment_id`, `timecode_in_ms`/`timecode_out_ms`, `timecode_in`/`timecode_out`, `segment_duration_seconds`, `segment_transcript`, and an empty `topic` field.
- **`segments_review.csv`**: one row per segment with `video_url`, `segment_id`, `title`, `creator_label`, `upload_date`, timecodes, `segment_duration_seconds`, an empty `topic` column, a 300-character preview, and the full `segment_transcript`. Use this file as the input to step 2, or label the `topic` column by hand.

`segment_id` is the segment's position within its video (0, 1, 2, …), so `video_url` + `segment_id` uniquely identifies a segment.

---

## Step 2: Label each segment with a topic (`label_from_codebook.py`)

Every segment gets **exactly one** label from your codebook. The label set is enforced through the JSON schema, so the model cannot invent new categories.

### Input

- **Segments CSV** (`--input`): needs `video_url` and `segment_transcript`. `segment_id`, `timecode_in`, `timecode_out` and `segment_duration_seconds` are used when present. `segments_review.csv` from step 1 works as-is. All columns are passed through to the output.
- **Codebook** (`--codebook`), in either format:
  - a **CSV** with a `label` column and an optional `definition` column:
    ```csv
    label,definition
    Sports,"Professional or amateur sports, athletes, games, or leagues."
    Economy,"Inflation, jobs, interest rates, markets, or the economy in general."
    ```
  - a **text file**, one category per line, optionally `label: definition`:
    ```
    Sports: Professional or amateur sports, athletes, games, or leagues.
    Economy
    ```

  Definitions are optional but noticeably improve consistency. If your definitions column has another name, pass `--definitions-column <name>`. An **"Other / Not Listed"** category is added automatically unless the codebook already has an equivalent, so segments that fit nothing aren't forced into the closest wrong category.

### Run

```bash
python label_from_codebook.py \
    --input segments_out/<run>/segments_review.csv \
    --codebook codebook.csv \
    --output-dir labeled
```

| Option | Default | Description |
|---|---|---|
| `--definitions-column` | `definition` | Name of the definitions column in the codebook CSV |
| `--codebook-definitions` | — | Optional separate CSV of definitions; overrides definitions in `--codebook` |
| `--output-dir` | `pipeline_output` | Output folder |
| `--model` | `gpt-4.1-mini` | Classification model |
| `--batch-size` | `20` | Segments classified per API call |
| `--cache-dir` | `.pipeline_cache` | Where model responses are cached |

### Output

- **`labeled_segments.csv`**: every input column plus `codebook_label`.
- **`codebook_summary.csv`**: one row per category with `n_segments`, `total_runtime_seconds` and three sample transcripts. Categories that were never used are kept, with zero counts.
- **`segments.json`**: the same data as JSON.

---

## Step 3: Subtopic presence and sentiment (`label_subtopics.py`)

For segments with one chosen topic, this script checks a list of **subtopics** one by one. For each segment and each subtopic it records `not_discussed`, `positive`, `negative` or `neutral`. Sentiment is **toward that subtopic**, not the segment's overall tone. Subtopics are not mutually exclusive, so one segment can mention several.

### Input

- **Labeled CSV** (`--input`): usually `labeled_segments.csv` from step 2.
- **Topic to analyze**: rows where `--topic-column` (default `codebook_label`) equals `--topic-label`.
- **Subtopics** (`--subtopics`): the same formats as the codebook, either a CSV with `label` and `definition` columns or a `label: definition` text file. Labels are converted to snake_case column names (`Gas prices` → `gas_prices`). If omitted, a built-in example list of 12 subtopics is used (see `DEFAULT_SUBTOPICS` in the script).

```csv
label,definition
Gas prices,"Gasoline prices, oil prices, fuel costs, or price increases at the pump."
Troops and casualties,"Troops or service members, including deployments, deaths, or injuries."
```

### Run

```bash
python label_subtopics.py \
    --input labeled/labeled_segments.csv \
    --topic-label "Economy" \
    --subtopics economy_subtopics.csv \
    --output-dir economy_subtopics
```

| Option | Default | Description |
|---|---|---|
| `--topic-label` | `Donald Trump` | Topic value to filter on; set this to one of your codebook labels |
| `--topic-column` | `codebook_label` | Column to filter on |
| `--subtopics` | built-in list | Subtopic definitions |
| `--output-dir` | `pipeline_output` | Output folder |
| `--model` | `gpt-4.1-mini` | Classification model |
| `--batch-size` | `20` | Segments per API call |
| `--concurrency` | `8` | Batches sent in parallel |
| `--cache-dir` | `.pipeline_cache` | Where model responses are cached |

### Output

- **`<topic>_subtopics_labeled.csv`**: the filtered segments with every original column plus one column per subtopic.
- **`subtopic_summary.csv`**: one row per subtopic with how many segments discuss it, the share of the topic's segments, counts of positive, negative and neutral, runtime discussed, and three sample transcripts.
- **`segments.json`**: the same data as JSON.

---

## Caching, re-runs and cost

- Every model response is cached in `--cache-dir` (default `.pipeline_cache/`). Cache keys include the model, the prompt and the exact segments in each batch (including a hash of their text), so re-running a command only pays for work not done yet, and a cached result is never reused for different segments. Interrupted runs can be restarted with the same command.
- Changing the codebook, definitions, subtopics or model changes the prompt, so everything is reclassified.
- Steps 2 and 3 send **20 segments per call**; step 1 sends one 10-minute chunk per call. Try a small sample first to check labels and cost before running a full corpus.
- In steps 2 and 3, a batch that fails after retries falls back to `Other / Not Listed` or `not_discussed` and logs `FAILED`. Check the console output on large runs.

## Files not to commit

The scripts create files that contain transcripts, model output or credentials. Keep these out of version control: `.env`, `.pipeline_cache/`, and your output folders (`segments_out/`, `pipeline_output/`, …).
