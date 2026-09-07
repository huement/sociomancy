# Sociomancy

Currently this library is only able to analyze YouTube channels, but the pipeline is designed to be extensible to other social media platforms.

> Catoptromancy is divination using a mirror. Sociomancy is a Portmanteau / Neologism for divination using YouTube.

## YouTube Creator Analytics Pipeline

An end-to-end data processing and machine learning pipeline that fetches YouTube channel and comment data via the YouTube Data API v3, processes text through local ONNX Transformer models, and outputs statistical analytics on community health, audience retention depth, and engagement authenticity.

---

## QUICK START

All pipeline commands are managed via [`uv`](https://github.com/astral-sh/uv). Run the steps below in sequence to ingest channel data, run local ONNX model inferences, establish control benchmarks, and render animated ECharts visualizers.

### Prerequisites & Setup

Ensure `uv` is installed, then synchronize project dependencies and local environment:

```bash
uv sync
```

#### Step-by-Step Execution Sequence

1. Target Channel Ingestion & 5-Pillar Analysis
   Collects raw channel metadata/comments and runs local ONNX Transformer inference (RoBERTa, GoEmotions, BART-MNLI, HDBSCAN) to build the primary scorecard payload.

```
uv run sociomancy @donflurgundy
```

Outputs: `data/raw/donflurgundy.json` and `data/processed/donflurgundy_metrics.csv`

2. Peer Group Discovery
   Generates a candidate pool of peer channels in the same niche using semantic topic profiling (BAAI/bge-small-en-v1.5) and Jaccard audience overlap scoring.

```
uv run python src/sociomancy/discover.py donflurgundy
```

Outputs: `output_cards/donflurgundy/related.json`

3. Control Group BenchmarkingInfers 5-pillar metrics across discovered peer channels, calculates niche arithmetic means ($\mu$) and standard deviations ($\sigma$), and normalizes target channel performance into relative $Z$-scores.

```
uv run python src/sociomancy/benchmark.py donflurgundy --delay 3.0
```

Outputs: data/processed/donflurgundy_scorecard.json and data/processed/donflurgundy_benchmark.json

4. Generate Animated ECharts Visualizers
   Renders dark-themed Apache ECharts HTML/Jinja templates and exports them into high-bitrate animated .mp4 video files using Pyppeteer and MoviePy.

```
# Render animated MP4 chart videos
uv run python src/charts/build_charts.py donflurgundy

# (Optional) Add --png to render static high-res images instead
uv run python src/charts/build_charts.py donflurgundy --png
```

Outputs: `output_charts/donflurgundy/*.mp4` (or .png)

5. (Optional) Generate Channel Profile Cards
   Creates transparent PNG profile cards displaying fetched avatar imagery, channel banners, handles, and subscriber counts.

```
uv run python src/cards/build_cards.py path/to/channels.json -o ./output_cards
```

---

## Data Ingestion (Data Fetched)

The pipeline interacts with the **YouTube Data API v3** to collect raw datasets for a target creator as well as a pre-defined set of 10–15 niche benchmark channels (control group).

### 1. Channel Metadata

- Channel ID, title, creation date, total view count, subscriber count, total video count.

### 2. Video Metadata (Last 15–30 Uploads)

- Video ID, title, publish timestamp, view count, like count, total comment count.

### 3. Comment Data (Up to 500 Top Comments Per Video)

- Comment ID, video ID, publish timestamp, comment text, like count.
- Anonymized author metadata (`authorChannelId` / `authorDisplayName`).

---

## Computed Statistics & Analytical Methodology

The pipeline processes raw JSON payloads and converts unstructured comment text into quantitative metrics using local ONNX Runtime models (`optimum.onnxruntime`).

### 1. Empathy vs. Toxicity Index

- **Data Sources:** Comment text and `like_count` across last 10–15 videos[cite: 1, 3].
- **Models Used:**
  - `unitary/unbiased-toxic-roberta` (ONNX): Evaluates probability scores ($0.0 \text{ to } 1.0$) for toxicity, severe toxicity, insults, and identity attacks[cite: 1].
  - `SamLowe/roberta-base-go_emotions` (ONNX): Maps text across 28 discrete emotional categories (_admiration_, _gratitude_, _disapproval_, _annoyance_)[cite: 1].
- **Computed Output:**
  - **Baseline Sentiment:** Overall toxicity percentage (% of comments scoring $>0.5$ toxicity) and dominant emotional distribution[cite: 1].
  - **Top Comment Resonance Delta ($\Delta R_{\text{tox}}$):** Isolates the top $N$ most-liked comments per video upload to evaluate community endorsement dynamics:
    $$\Delta R_{\text{tox}} = \text{Toxicity}_{\text{Top Comments}} - \text{Toxicity}_{\text{Baseline}}$$
    _A negative $\Delta R_{\text{tox}}$ indicates the broader community actively upvotes empathetic/supportive comments and suppresses baseline noise.\_
  - **Relative $Z$-score:**
    $$Z_{\text{toxicity}} = \frac{X_{\text{channel}} - \mu_{\text{control}}}{\sigma_{\text{control}}}$$
    _Normalizes $\Delta R_{\text{tox}}$ relative to the control group’s average toxicity distribution.\_
  - **Interpretation:** A higher $Z_{\text{toxicity}}$ indicates the channel’s toxicity profile is more extreme relative to the control group.

### 2. Core Tribe Ratio

- **Data Sources:** `authorChannelId` array collected across all scraped video uploads.
- **Calculation:** Identifies repeat commenters to measure audience loyalty vs. one-time visitors.
  $$\text{Core Tribe Ratio (\%)} = \left( \frac{\text{Unique Authors with } \ge 3 \text{ Video Comments}}{\text{Total Unique Authors in Dataset}} \right) \times 100$$
- **Computed Output:** Percentage of active community members participating regularly across multiple uploads.

### 3. Parasocial Impact Density Score

- **Data Sources:** Comment text, character lengths, token variance, and `like_count`[cite: 1, 3].
- **Models Used:**
  - `facebook/bart-large-mnli` (ONNX): Zero-shot classification tagging text for personal disclosure intent (labels: `"Life Impact"`, `"Applied Skill"`, `"Emotional Support"`, `"Generic Praise"`)[cite: 1].
- **Computed Output:**
  - Average word count and token variance per comment[cite: 1].
  - Baseline disclosure rate expressed as `Impact Disclosures per 1,000 Comments`[cite: 1].
  - **Top Comment Parasocial Resonance Delta ($\Delta R_{\text{parasocial}}$):** Isolates top-voted comments to test if audience upvoting amplifies parasocial disclosures vs. surface-level noise:
    $$\Delta R_{\text{parasocial}} = \text{Parasocial Score}_{\text{Top Comments}} - \text{Parasocial Score}_{\text{Baseline}}$$
    _A positive $\Delta R_{\text{parasocial}}$ indicates the community actively rewards deep emotional connection, life impact disclosures, and vulnerability with high upvote counts.\_

```text
                         ┌──────────────────┐
                         │ YouTube Comment  │
                         └────────┬─────────┘
                                  │
                 ┌────────────────▼─────────────────┐
                 │ Text Normalize & language Detect │
                 └────────────────┬─────────────────┘
                                  │
            ┌─────────────────────┼────────────────────┐
            │                     │                    │
            ▼                     ▼                    ▼
       Regex Rules          Semantic Model       Emotion Model
       High Precision       Meaning/Intent       Affect/Similarity
            │                     │                    │
            └─────────────────────┼────────────────────┘
                                  ▼
                  Weighted Scoring via Emotion ONNX
                                  │
                                  ▼
                       Parasocial Signal Score
                      ┌───────────────────────┐
                      │ Parasocial Assessment │
                      ├───────────────────────┤
                      │ Score: 0.81           │
                      │ Level: strong         │
                      │                       │
                      │ perceived intimacy    │
                      │ emotional reliance    │
                      │ behavioral influence  │
                      └───────────────────────┘
```

### 4. Cold-Audience Stress Test ($\Delta T$)

- **Data Sources:** View-to-subscriber ratios, video median view counts, comment toxicity scores.
- **Detection Trigger:** Flags a video as an "Algorithmic Push" if it meets both conditions:
  1. $\frac{\text{Video Views}}{\text{Channel Subscribers}} \ge 3.0$
  2. $\text{Video Views} > \text{Median Channel Views} + (2 \times \text{Interquartile Range})$
- **Calculation:** Computes toxicity variance between algorithmic breakout videos and baseline uploads:
  $$\Delta T = \text{Toxicity}_{\text{Pushed Videos}} - \text{Toxicity}_{\text{Baseline Videos}}$$
- **Computed Output:** A delta score ($\Delta T$) indicating whether cold non-subscriber traffic degrades comment section sentiment.

### 5. Bot & Authenticity Estimation

- **Data Sources:** Comment text, timestamps, author metadata.
- **Models & Methods Used:**
  - `gpt2` / `distilgpt2`: Computes token perplexity and burstiness. Scripted/bot text displays unnaturally low perplexity and low perplexity variance across sentences.
  - `sentence-transformers/all-MiniLM-L6-v2` (ONNX) + HDBSCAN: Converts comments into 384-dimensional embeddings and clusters them to find tight semantic copy-paste groups.
- **Computed Output:** Composite 0–100% **Bot Confidence Rating** per comment, aggregated into an overall channel **Comment Authenticity Score (%)**.

The critical thing is that no single signal should be enough to call something a bot.

```
                   COMMENTS
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   Text Analysis   Behavior       Duplication
        │              │              │
        │              │              │
   Perplexity      Timing         Embeddings
   Burstiness      Frequency      HDBSCAN
   Length          Author reuse   Copy groups
        │              │              │
        └──────────────┼──────────────┘
                       ↓
              COMMENT BOT SCORE
                    0–100
                       ↓
             Aggregate by channel
                       ↓
         COMMENT AUTHENTICITY SCORE
                    0–100
```

---

## Peer Channel Discovery (Refactored)

The `discover.py` script has been refactored into a multi-stage system to identify peer channels with greater accuracy. The new system leverages a combination of semantic similarity and audience overlap to provide a more holistic and explainable ranking.

### 1. Candidate Generation

A large pool of 50-100 candidate channels is generated using multiple discovery signals:

- **Channel Search**: Uses the target channel's keywords to find similar channels.
- **Video Search**: Searches for relevant topics using `type="video"` and extracts the resulting channel IDs.

### 2. Semantic Ranking

The candidate pool is then ranked based on semantic similarity:

- **Channel Profiling**: A textual profile is created for the target and each candidate channel, including titles, descriptions, keywords, and recent video titles.
- **Embeddings**: The `BAAI/bge-small-en-v1.5` model is used to generate embeddings for each channel profile.
- **Cosine Similarity**: The cosine similarity between the target and candidate embeddings is calculated to produce a `semantic_similarity` score.

The top 20 candidates with the highest semantic similarity are selected for the next stage.

### 3. Audience Similarity

For the top 20 candidates, a more in-depth audience analysis is performed:

- **Jaccard Similarity**: The intersection over union of the target and candidate commenter sets.
- **Target Audience Overlap**: `shared / target_commenters`
- **Candidate Audience Overlap**: `shared / candidate_commenters`
- **Audience Affinity**: The harmonic mean of the two overlap ratios.

### 4. Peer Score

A composite `peer_score` is calculated as a weighted combination of the semantic similarity and audience affinity.

## Output Files Generated

- `data/raw/{channel_id}_data.json` — Raw API responses (videos and comment threads).
- `data/processed/{channel_id}_metrics.csv` — Feature vectors per comment (toxicity, emotion, perplexity).
- `reports/{channel_id}_scorecard.md` — Final markdown summary containing computed metrics, control group comparisons, and overall scorecard scores.

### Channel Cards

The cards are move of a 'nice to have' than a need, they simply take the given channels images and handle and create a helpful png graphic.

## Commands

### Generate transparent PNG cards from custom JSON data:

```bash
uv run python src/cards/build_cards.py path/to/my_channels.json
```

### Save PNGs to a specific output folder:

```bash
uv run python src/cards/build_cards.py path/to/my_channels.json -o ./output_cards
```

### Preview custom JSON data in the browser (dev mode):

```bash
uv run python src/cards/build_cards.py src/cards/sample.json --dev
```

### Channel Charts

This is the most important asset the library generates, after gather up all the data for a channel and then the same data for related channels, you can finally construct helpful graphics that explain the results visually.

By default this will output animated MP4 versions of each chart.

```
uv run python src/charts/build_charts.py donflurgundy
```

Optionally add the `--png` parameter to output static images instead of animations

```
uv run python src/charts/build_charts.py donflurgundy --png
```

The charts are powered by `Jinja` templates and `Apache ECharts` libraries. You can easily add different charts, simply drop in new templates and update the build_chart.py command.
