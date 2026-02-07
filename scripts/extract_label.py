"""
Randomly sample a fixed number of labels and examples per label.

Outputs two JSONL files:
1) input-only: {"input": ...}
2) input+label: {"input": ..., "label": ...}
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List


def _add_project_root_to_syspath() -> None:
    here = Path(__file__).resolve()
    cur = here.parent
    for _ in range(10):
        if (cur / "src").is_dir():
            sys.path.insert(0, str(cur))
            return
        cur = cur.parent


_add_project_root_to_syspath()

from src.synthetic.dataprocesser import (  # noqa: E402
    group_texts_by_label,
    infer_text_and_label_keys,
    load_jsonl,
)

NAME = "mtop_intent"
DATA_PATH = Path(f"./code/DataGene/datasets/{NAME}/small.jsonl")
OUT_PATH = Path("./code/DataGene/datasets/test/")
N_LABELS = 10
N_PER_LABEL = 0  # set <= 0 to keep all samples per label
SEED = 12
LABEL_LIST: List[str] | None = None
OUT_NAME = f"ext_{DATA_PATH.stem}_{NAME}_labels{N_LABELS}_n{N_PER_LABEL}"


def _select_labels(all_labels: List[str], num_labels: int, label_list: List[str] | None, rng: random.Random) -> List[str]:
    if num_labels <= 0:
        raise ValueError("N_LABELS must be >= 1")
    chosen: List[str] = []
    if label_list:
        chosen = [lab for lab in label_list if lab in all_labels]
    if len(chosen) < num_labels:
        remaining = [lab for lab in all_labels if lab not in chosen]
        need = min(num_labels - len(chosen), len(remaining))
        if need > 0:
            chosen.extend(rng.sample(remaining, k=need))
    return chosen[:num_labels]


def run() -> int:
    rng = random.Random(SEED)
    rows = load_jsonl(DATA_PATH)
    if not rows:
        raise ValueError(f"No rows found in {DATA_PATH}")

    text_key, label_key = infer_text_and_label_keys(rows)
    by_label: Dict[str, List[str]] = group_texts_by_label(rows, text_key, label_key)
    all_labels = sorted(by_label.keys())
    if not all_labels:
        raise ValueError("No valid labels found in the dataset.")

    selected_labels = _select_labels(all_labels, N_LABELS, LABEL_LIST, rng)

    out_input = OUT_PATH / f"{OUT_NAME}.jsonl"
    out_label = OUT_PATH / f"{OUT_NAME}_prelabel.jsonl"
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with out_input.open("w", encoding="utf-8") as f_input, out_label.open("w", encoding="utf-8") as f_label:
        for label in selected_labels:
            samples = by_label.get(label, [])
            if not samples:
                continue
            if N_PER_LABEL <= 0 or len(samples) <= N_PER_LABEL:
                chosen = list(samples)
            else:
                chosen = rng.sample(samples, k=N_PER_LABEL)
            for text in chosen:
                f_input.write(json.dumps({"input": text}, ensure_ascii=False) + "\n")
                f_label.write(json.dumps({"input": text, "label": label}, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"data: {DATA_PATH}")
    print(f"labels_all: {len(all_labels)}")
    print(f"labels_selected: {len(selected_labels)}")
    print(f"n_per_label: {N_PER_LABEL}")
    print(f"written: {total_written}")
    print(f"output_input: {out_input}")
    print(f"output_label: {out_label}")
    return total_written


if __name__ == "__main__":
    run()
