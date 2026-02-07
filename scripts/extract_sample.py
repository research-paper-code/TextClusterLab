"""
Extract a fixed number of samples per label and keep only the input text.
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


DATA_PATH = Path("./code/DataGene/datasets/clinc/small.jsonl")
OUT_PATH = Path("./code/DataGene/datasets/test/ext_clinc_small_10s.jsonl")
N_PER_LABEL = 10
SEED = 13


def run() -> int:
    rng = random.Random(SEED)
    by_label: Dict[str, List[str]] = {}

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = str(obj.get("label", "")).strip()
            text = obj.get("input")
            if not label or text is None:
                continue
            by_label.setdefault(label, []).append(str(text))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for label in sorted(by_label):
            samples = by_label[label]
            if len(samples) <= N_PER_LABEL:
                chosen = list(samples)
            else:
                chosen = rng.sample(samples, k=N_PER_LABEL)
            for text in chosen:
                out.write(json.dumps({"input": text}, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"labels: {len(by_label)}")
    print(f"written: {total_written}")
    print(f"output: {OUT_PATH}")
    return total_written


if __name__ == "__main__":
    run()
