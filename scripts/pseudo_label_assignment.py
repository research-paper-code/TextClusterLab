"""
Embed ext_clinc_10s.jsonl using Instructor and assign KMeans cluster labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


def _add_project_root_to_syspath() -> None:
    here = Path(__file__).resolve()
    cur = here.parent
    for _ in range(10):
        if (cur / "src").is_dir():
            sys.path.insert(0, str(cur))
            return
        cur = cur.parent


_add_project_root_to_syspath()

from src.embedding.embeddings import EmbeddingConfig, encode  # noqa: E402
from src.metrics.cluster_metrics import kmeans_cluster  # noqa: E402


DATA_PATH = Path("./code/DataGene/datasets/test/ext_small_labels10_n0.jsonl")
OUT_PATH = Path("./code/DataGene/datasets/test/ext_small_labels10_n0_plabel.jsonl")
MODEL_NAME = "instructor"
N_CLUSTERS = 10
RANDOM_STATE = 12
N_INIT = 5


def _load_inputs(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input" not in obj:
                continue
            records.append(obj)
    return records


def run() -> int:
    records = _load_inputs(DATA_PATH)
    texts = [str(r["input"]) for r in records]
    if not texts:
        raise ValueError(f"No valid 'input' records found in {DATA_PATH}")

    cfg = EmbeddingConfig()
    emb = encode(texts, model_name=MODEL_NAME, config=cfg)
    y_pred = kmeans_cluster(
        emb,
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    labels = (y_pred + 1).astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for obj, lab in zip(records, labels):
            out_obj = {"input": obj.get("input"), "label": str(lab)}
            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"records: {len(records)}")
    print(f"clusters: {N_CLUSTERS}")
    print(f"output: {OUT_PATH}")
    return len(records)


if __name__ == "__main__":
    run()
