"""
Main test script: load dataset and select labels.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import List, Any
from datetime import datetime
from time import perf_counter


def _add_project_root_to_syspath() -> None:
    here = Path(__file__).resolve()
    cur = here.parent
    for _ in range(10):
        if (cur / "src").is_dir():
            sys.path.insert(0, str(cur))
            return
        cur = cur.parent


_add_project_root_to_syspath()

from src.synthetic.dataprocesser import load_jsonl, infer_text_and_label_keys  # noqa: E402
from src.synthetic.pipeline import generate_and_save  # noqa: E402
from src.embedding.pipeline import run_and_save_embeddings, get_all_embeddings  # noqa: E402
from src.metrics.cluster_metrics import compute_clustering_metrics  # noqa: E402
from src.metrics.utils import (
    compute_weights_for_chosen_labels,
    example_pick,
    merge_synthetic_with_original,
)  # noqa: E402
from src.synthetic.dataprocesser import write_jsonl  # noqa: E402
from sklearn.metrics import calinski_harabasz_score  # noqa: E402
import numpy as np


# === Config ===

# DA = "clinc"
# DC = "small"
# DATA_NAME = "extracted_data"
# DATA_CHOICE = f"extract_10_{DA}_{DC}"

# DA = "test"
# DC = "ext_small_banking77_labels10_n0_prelabel"
# DATA_NAME = "test"
# DATA_CHOICE = "ext_small_banking77_labels10_n0_prelabel"
# EMBED_MODEL_NAME = "e5"


DATA_NAME = "massive_intent"
DATA_CHOICE = "small"
DA = DATA_NAME
DC = DATA_CHOICE
EMBED_MODEL_NAME = "instructor"  # "instructor" | "e5" | "gemma" | "qwen" | "sbert"

DATA_PATH = Path(f"./code/DataGene/datasets/{DATA_NAME}/{DATA_CHOICE}.jsonl")
OUT_DIR = Path("./code/DataGene/synthetic_data")



MODEL_SELECT = os.getenv("VLLM_MODEL")  # if None -> auto-select
N_NEW = int(os.getenv("N_NEW", "50"))
SEED = int(os.getenv("SEED", "42"))
SYNC = os.getenv("SYNC", "false").strip().lower() in ("1", "true", "yes", "y", "t")
SYNC = False            # True: n_new equal the sample size of each label

NUM_LABELS = 59
LABEL_LIST: List[Any] | None = None  # e.g. ["change volume", "timer"]
# LABEL_LIST = ["translate", "transfer"]
NUM_EXAMPLES = 5
TYPE = "imbalance"  # "imbalance" | "compact" | "diversity"
IMBALANCE_RANDOM = False  # only applies when TYPE == "imbalance"
DUPLICATE = False
LLM_NAME = "Qwen"
merge_name = f"{LLM_NAME}_{TYPE}_{NUM_LABELS}_{DA}_{DC}_{NUM_EXAMPLES}_{EMBED_MODEL_NAME}"

# === Config end ===

def _select_labels(all_labels: List[Any], num_labels: int, label_list: List[Any] | None, seed: int) -> List[Any]:
    if num_labels <= 0:
        raise ValueError("NUM_LABELS must be >= 1")
    chosen = []
    if label_list:
        chosen = [lab for lab in label_list if lab in all_labels]
    if len(chosen) < num_labels:
        rng = random.Random(seed)
        remaining = [lab for lab in all_labels if lab not in chosen]
        need = min(num_labels - len(chosen), len(remaining))
        if need > 0:
            chosen.extend(rng.sample(remaining, k=need))
    return chosen[:num_labels]


def run():
    start_time = perf_counter()
    start_dt = datetime.now()
    rows = load_jsonl(DATA_PATH)
    rows_arr = list(rows)
    _, label_key = infer_text_and_label_keys(rows)
    all_labels = sorted({str(r.get(label_key, "")).strip() for r in rows if str(r.get(label_key, "")).strip()})

    selected = _select_labels(all_labels, NUM_LABELS, LABEL_LIST, SEED)
    run_dir = OUT_DIR / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("data_path:", DATA_PATH)
    print("embed_model:", EMBED_MODEL_NAME)
    print("num_labels:", NUM_LABELS)
    print("selected_labels:", selected)
    print("run_dir:", run_dir)
    print("merge_dir:", f"{run_dir}/{merge_name}.jsonl")

    emb, labels = run_and_save_embeddings(
        data_name=DATA_NAME,
        data_choice=DATA_CHOICE,
        data_path=str(DATA_PATH),
        model_name=EMBED_MODEL_NAME,
    )
    print("embeddings shape:", emb.shape)
    print("labels count:", len(labels))

    labels_arr = [str(lab) for lab in labels]
    imbalance_rng = random.Random(SEED)
    for lab in selected:
        idx = [i for i, y in enumerate(labels_arr) if y == str(lab)]
        emb_subset = emb[idx]
        print(f"label: {lab}")
        # print(f"  indices: {idx}")
        print(f"  subset shape: {emb_subset.shape}")

        weights, chosen_idx, _ = compute_weights_for_chosen_labels(
            emb,
            labels,
            [lab],
            pick_type=TYPE,
            use_inverse=True,
        )
        # Map weights to global indices for sampling
        sample_indices = list(chosen_idx)
        k = min(NUM_EXAMPLES, len(sample_indices))
        if IMBALANCE_RANDOM and TYPE.strip().lower() == "imbalance":
            label_size = len(sample_indices)
            n_new = imbalance_rng.randint(1, max(1, 3 * label_size))
        else:
            n_new = len(sample_indices) if SYNC else N_NEW

        picked_by_type = example_pick(
            samples=sample_indices,
            weights=weights,
            num_examples=k,
            seed=SEED,
        )

        # N = len(weights)  
        # wmax, wmin = weights.max(), weights.min()
        # gamma = np.log(N) / np.log(wmax / wmin)
        # p = weights**gamma
        # p = p / p.sum()  
        # print("chosen w:", weights, "chosen p:", p)
        # print(f"  weights first10: {weights[:10]}")

        print(f"  weights size: {weights.shape}")
        # print(f"  picked_by_{TYPE} (k={k}): {picked_by_type}")

        picked_rows = [rows_arr[i] for i in picked_by_type]
        synthetic_records = generate_and_save(
            rows=picked_rows,
            out_dir=run_dir,
            n_new=n_new,
            model_select=MODEL_SELECT,
        )
        print(f"  synthetic records: {len(synthetic_records)}")

    merged = merge_synthetic_with_original(
        data_name=DATA_NAME,
        data_choice=DATA_CHOICE,
        synthetic_dir=run_dir,
        duplicate=DUPLICATE,
    )
    result_path = "./code/DataGene/result"
    synthetic_dir = Path(result_path)
    merged_path = synthetic_dir / f"{merge_name}.jsonl"
    merged_path = Path(merged_path)
    write_jsonl(merged_path, merged)
    print(f"merged file: {merged_path}")

    merged_emb, merged_labels = get_all_embeddings(
        data_name=DATA_NAME,
        data_choice=DATA_CHOICE,
        data_path=str(merged_path),
        model_name=EMBED_MODEL_NAME,
    )
    y_pred, ari, nmi, sil, db = compute_clustering_metrics(merged_emb, merged_labels)
    ch = None
    if y_pred is not None and len(set(y_pred)) > 1 and merged_emb.shape[0] > 1:
        ch = calinski_harabasz_score(merged_emb, y_pred)
    results = {
        "config": {
            "data_name": DATA_NAME,
            "data_choice": DATA_CHOICE,
            "embed_model_name": EMBED_MODEL_NAME,
            "num_labels": NUM_LABELS,
            "label_list": LABEL_LIST,
            "num_examples": NUM_EXAMPLES,
            "type": TYPE,
            "n_new": N_NEW,
            "imbalance_random": IMBALANCE_RANDOM,
            "model_select": MODEL_SELECT,
            "duplicate": DUPLICATE,
        },
        "metrics": {
            "n_clusters": len(set(merged_labels)),
            "ari": round(float(ari), 4),
            "nmi": round(float(nmi), 4),
            "silhouette": None if sil is None else round(float(sil), 4),
            "davies_bouldin": None if db is None else round(float(db), 4),
            "calinski_harabasz": None if ch is None else round(float(ch), 4),
        },
    }
    write_jsonl(synthetic_dir / f"results_{TYPE}_{NUM_LABELS}_{DA}_{DC}_{NUM_EXAMPLES}_{EMBED_MODEL_NAME}.jsonl", [results])

    end_dt = datetime.now()
    elapsed = perf_counter() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"end time: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"total running time: {h:02d}:{m:02d}:{s:02d}")
    return emb, labels


if __name__ == "__main__":
    run()
