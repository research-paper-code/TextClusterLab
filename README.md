# TextClusterLab

TextClusterLab is a research-oriented toolkit for text embedding analysis, clustering evaluation, and synthetic data augmentation for intent-style datasets.

It combines:

- Multi-model text embeddings (`instructor`, `e5`, `gemma`, `qwen`, `sbert`)
- Clusterability/suitability diagnostics (8 criteria)
- Clustering protocol comparison (KMeans, HDBSCAN, Agglomerative)
- LLM-based synthetic utterance generation through an OpenAI-compatible endpoint (for example vLLM)

## Project Structure

```text
TextClusterLab/
├── datasets/                      # Example JSONL datasets
├── local_models/                  # Place local embedding models here (optional)
├── scripts/
│   ├── data_clusterable_benchmark.py
│   ├── cluster_evaluation_protocol.py
│   ├── cluster_evaluation.py
│   ├── main.py
│   ├── extract_sample.py
│   ├── extract_label.py
│   └── pseudo_label_assignment.py
└── src/
    ├── embedding/                 # Data loading + embedding pipelines
    ├── metrics/                   # Clustering metrics + utility functions
    └── synthetic/                 # Synthetic data generation pipeline
```

## Requirements

- Python 3.10+
- `pip` (or any equivalent package manager)

Install baseline dependencies:

```bash
pip install -U pip
pip install numpy scikit-learn scipy networkx pandas openpyxl matplotlib openai sentence-transformers InstructorEmbedding
```

Optional:

- `hdbscan` (needed for HDBSCAN in `cluster_evaluation_protocol.py`)

```bash
pip install hdbscan
```

## Data Format

The toolkit expects JSONL files with at least:

```json
{"input": "book a table for dinner", "label": "restaurant_reservation"}
```

Extra keys (for example `task`) are allowed.

## Quick Start

### 1) Clusterability benchmark (C1-C8)

```bash
python scripts/data_clusterable_benchmark.py \
  --data-path datasets/clinc/small.jsonl \
  --model-name instructor \
  --second-model-name e5
```

This prints eight clusterability criteria (neighborhood signal, stability, separation ratio, Hopkins statistic, and more).

### 2) Compare clustering methods and export Excel

```bash
python scripts/cluster_evaluation_protocol.py \
  --data-dir datasets/clinc \
  --datasets small \
  --model instructor \
  --output outputs/result.xlsx
```

This evaluates KMeans, HDBSCAN, and Agglomerative clustering and saves a result table.

### 3) End-to-end synthesis + evaluation

Before running, start an OpenAI-compatible LLM endpoint (for example vLLM) and set:

```bash
export VLLM_BASE_URL="http://127.0.0.1:6006/v1"
export VLLM_API_KEY="EMPTY"
export VLLM_MODEL="your-model-id"   # optional; auto-selects first served model if omitted
```

Then run:

```bash
python scripts/main.py
```

`scripts/main.py` generates synthetic data for selected labels, merges with original data, and reports clustering metrics.

## Script Notes

- `scripts/cluster_evaluation.py` is a fixed-config script (edit constants in file, then run).
- `scripts/extract_sample.py`, `scripts/extract_label.py`, and `scripts/pseudo_label_assignment.py` are dataset utility scripts with in-file configuration constants.
- `scripts/main.py` is an experiment driver and also uses in-file configuration constants.

## Important Configuration Notes

1. Some defaults still point to paths used in an earlier project layout (for example `./code/DataGene/...` or `/home/wdm/code/DataGene/...`).
2. Default embedding model paths in `src/embedding/embeddings.py` (`get_default_model_paths`) are placeholders and should be updated for your machine.
3. The `local_models/` folder is included for convenience, but model files are not bundled in this repository.

## Typical Outputs

- Console metrics (ARI, NMI, Silhouette, Davies-Bouldin, Calinski-Harabasz)
- t-SNE figures (when using `cluster_evaluation.py`)
- Excel result table (from `cluster_evaluation_protocol.py`)
- Synthetic JSONL outputs and merged dataset files (from `main.py`)

## Troubleshooting

- `ImportError: InstructorEmbedding`:
  Install `InstructorEmbedding` and its backend dependencies.
- `No models returned from /v1/models`:
  Check `VLLM_BASE_URL` and confirm your model server is running.
- HDBSCAN failure:
  Install `hdbscan` or skip that method.
- File-not-found on dataset/model paths:
  Update script constants or pass explicit CLI paths where supported.

## License

No license file is currently included. Add a `LICENSE` before open-source distribution.

