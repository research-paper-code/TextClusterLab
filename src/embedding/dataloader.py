"""
Data loading & class selection utilities.

This module is derived from the original methodCompare.ipynb notebook, but simplified:
- Loads CLINC-style JSONL where each line is a JSON dict with keys:
  - "input": text
  - "label": class/intent label
- Supports selecting:
  - a random subset of classes (default 1 class),
  - a user-provided list of class names,
  - or all data.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


DEFAULT_DATA_BASE_DIR = "./code/DataGene/datasets"


@dataclass(frozen=True)
class DataSpec:
    """Dataset specification."""
    name: str = "clinc"
    choice: str = "large"
    path: Optional[str] = None  # if None, uses DEFAULT_DATA_BASE_DIR/{name}/{choice}.jsonl

    def resolved_path(self, base_dir: str = DEFAULT_DATA_BASE_DIR) -> str:
        """Return an absolute/explicit JSONL path for this dataset."""
        if self.path:
            return self.path
        return os.path.join(base_dir, self.name, f"{self.choice}.jsonl")


def load_clinc_jsonl(path: str) -> Tuple[List[str], List[Any]]:
    """
    Load a CLINC-style JSONL file.

    Each line should be a JSON object with:
      - "input": the utterance text
      - "label": the class/intent label

    Returns:
      sentences: list[str]
      labels: list[Any]
    """
    sentences: List[str] = []
    labels: List[Any] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("input", None)
            label = obj.get("label", None)
            if text is None or label is None:
                continue
            sentences.append(str(text))
            labels.append(label)

    return sentences, labels


def select_classes(
    sentences: Sequence[str],
    labels: Sequence[Any],
    *,
    class_num: int = 1,
    class_names: Optional[Sequence[Any]] = None,
    select_all: bool = False,
    seed: int = 42,
) -> Tuple[List[str], List[Any], List[Any]]:
    """
    Select data by class/label.

    Args:
      class_num: number of classes to randomly sample when class_names is None and select_all is False.
      class_names: explicit list of class labels to keep. If provided, overrides class_num.
      select_all: if True, keeps all data and ignores class_num/class_names.
      seed: RNG seed used only for random class selection.

    Returns:
      sel_sentences, sel_labels, chosen_class_names
    """
    if len(sentences) != len(labels):
        raise ValueError("sentences and labels must have the same length")

    unique_labels = sorted(set(labels), key=lambda x: str(x))

    if select_all:
        return list(sentences), list(labels), unique_labels

    if class_names is None:
        if class_num <= 0:
            raise ValueError("class_num must be >= 1 when select_all is False and class_names is None")
        rng = random.Random(seed)
        if class_num >= len(unique_labels):
            chosen = unique_labels
        else:
            chosen = rng.sample(unique_labels, k=class_num)
    else:
        chosen = list(class_names)

    chosen_set = set(chosen)
    sel_sentences: List[str] = []
    sel_labels: List[Any] = []
    for s, y in zip(sentences, labels):
        if y in chosen_set:
            sel_sentences.append(s)
            sel_labels.append(y)

    return sel_sentences, sel_labels, chosen


def load_and_select(
    data: DataSpec,
    *,
    base_dir: str = DEFAULT_DATA_BASE_DIR,
    class_num: int = 1,
    class_names: Optional[Sequence[Any]] = None,
    select_all: bool = False,
    seed: int = 42,
) -> Tuple[List[str], List[Any], List[Any]]:
    """
    Convenience wrapper: resolve path, load CLINC JSONL, then select classes.

    Returns:
      sentences, labels, chosen_class_names
    """
    path = data.resolved_path(base_dir=base_dir)
    sentences, labels = load_clinc_jsonl(path)
    return select_classes(
        sentences, labels,
        class_num=class_num,
        class_names=class_names,
        select_all=select_all,
        seed=seed,
    )
