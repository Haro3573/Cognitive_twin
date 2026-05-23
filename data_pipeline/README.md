# LLM-Based Data Pipeline (Cognitive Refiner Subagent)

This project is an automated pipeline designed to preprocess large dialog datasets and extract structured insights about user preferences, values, and decision-making styles using a local/CLI LLM runner ("Cognitive Refiner" Subagent).

## Pipeline Flow

1. **Preprocessing (`extractor_service/`)**:
   - Written in Go.
   - Extracts dialog triads of `User(t-1) -> AI(t) -> User(t)` from raw conversation files (e.g. `final_data.json`).
   - Uses `github.com/buger/jsonparser` for extremely fast parsing of large JSON payloads.
2. **Data Extraction (`batch.py`)**:
   - Sends the raw data to the Go service.
   - Saves the extracted conversational triads into a local `extracted_pairs.json`.
3. **Cognitive Refiner Subagent (`refine_batch.py`)**:
   - Reads `extracted_pairs.json`.
   - Iterates through the data in batches of 20.
   - Leverages a local `gemini` CLI tool to apply the **Cognitive Refiner** logic.
   - Extracts actionable insights in English and saves them incrementally to `final_seeds.json` with the schema:
     - `content` (string): The extracted preference or value.
     - `type` (string): `'decision'`, `'preference'`, `'value'`, or `'anchor_candidate'`.
     - `context` (object): Contain `domain_tags` (slugified array) and `action` (string).
     - `source_uuid` (string): Unique identifier mapping back to the conversation.

## Getting Started

### Prerequisites

- **Go** (v1.18+)
- **Python 3.x**
- **pip requirements**:
  ```bash
  pip install requests
  ```
- **Gemini CLI tool (`gemini`)**: Make sure you have the `gemini` CLI tool installed, authenticated, and globally available via your path.

### Execution

To run the entire pipeline from end-to-end automatically, use the provided orchestrator script:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This shell script will:
1. Spin up the Go extractor service in the background.
2. Run `batch.py` to call the service and dump `extracted_pairs.json`.
3. Stop the Go extractor service safely.
4. Execute `refine_batch.py` to prompt the LLM and generate the final `final_seeds.json`.

## Samples

To help users understand the input and output formats without processing the entire dataset, a few sample items have been provided:
- **Raw Data Sample (`samples/raw_sample.json`)**: Contains 3 conversational objects extracted from `final_data.json`.
- **Processed Insights Sample (`samples/seeds_sample.json`)**: Contains 3 corresponding post-processed insight objects (SeedItems) extracted from `final_seeds.json`.

