"""
Data loading / grouping / writing utilities for CLINC-style JSONL datasets.

This module is intentionally focused on IO + dataset shaping:
- Load JSONL rows
- Infer text/label keys
- Group utterances by label
- Choose which labels to augment
- Write JSONL outputs
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into memory.

    Parameters
    ----------
    path:
        Path to a *.jsonl file where each line is a JSON object.

    Returns
    -------
    rows:
        A list of parsed JSON objects (dicts).

    Notes
    -----
    - Blank lines are skipped.
    - This does not validate schema beyond JSON parsing.
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_text_and_label_keys(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Infer which keys correspond to "text" and "label" from a CLINC-like JSON row.

    Parameters
    ----------
    rows:
        Loaded dataset rows. The first row is used for key inference.

    Returns
    -------
    (text_key, label_key):
        The key names to read utterance text and intent label from each row.

    Raises
    ------
    ValueError:
        If a reasonable text key / label key cannot be inferred.

    Heuristic
    ---------
    1) Try common key names.
    2) Otherwise, pick the first non-empty string field as text, and the next
       non-empty string field as label.
    """
    common_text_keys = ["text", "utterance", "sentence", "query", "input"]
    common_label_keys = ["intent", "label", "class", "domain"]

    if not rows:
        raise ValueError("Empty dataset: no rows in JSONL.")

    sample = rows[0]
    text_key = next((k for k in common_text_keys if k in sample), None)
    label_key = next((k for k in common_label_keys if k in sample), None)

    if text_key is None:
        for k, v in sample.items():
            if isinstance(v, str) and v.strip():
                text_key = k
                break

    if label_key is None:
        for k, v in sample.items():
            if k == text_key:
                continue
            if isinstance(v, str) and v.strip():
                label_key = k
                break

    if text_key is None or label_key is None:
        raise ValueError(f"Could not infer keys from row keys={list(sample.keys())}")
    return text_key, label_key


def group_texts_by_label(
    rows: List[Dict[str, Any]],
    text_key: str,
    label_key: str,
) -> Dict[str, List[str]]:
    """
    Build a mapping: label -> list of utterances for that label.

    Parameters
    ----------
    rows:
        Parsed dataset rows.
    text_key:
        Key for utterance text in each row.
    label_key:
        Key for intent label in each row.

    Returns
    -------
    label2texts:
        Dict mapping each label to its non-empty utterances (strings).

    Notes
    -----
    - Returned texts are not normalized; normalization is handled downstream.
    """
    label2texts: Dict[str, List[str]] = {}
    for r in rows:
        label = str(r.get(label_key, "")).strip()
        if not label:
            continue
        text = str(r.get(text_key, "")).strip()
        if not text:
            continue
        label2texts.setdefault(label, []).append(text)
    return label2texts


def choose_labels(
    available_labels: Iterable[str],
    seed: int,
    requested: Optional[List[str]],
) -> List[str]:
    """
    Decide which labels (classes) to augment.

    Parameters
    ----------
    available_labels:
        All labels that exist in the dataset.
    seed:
        Random seed used when no labels are explicitly requested.
    requested:
        Labels passed by the user via --classes (may be None).

    Returns
    -------
    labels_to_augment:
        The final list of labels to generate for.

    Behavior
    --------
    - If requested is provided and non-empty: validate and return it.
    - Else: randomly pick ONE label uniformly (matching the original behavior).

    Raises
    ------
    ValueError:
        If any requested label does not exist in the dataset.
    """
    avail = sorted(set(map(str, available_labels)))
    if not avail:
        raise ValueError("No valid labels found in the dataset.")

    if requested:
        missing = [c for c in requested if c not in avail]
        if missing:
            preview = ", ".join(avail[:30]) + (" ..." if len(avail) > 30 else "")
            raise ValueError(
                f"Unknown --classes: {missing}. "
                f"Available labels include: {preview}"
            )
        return requested

    rng = random.Random(seed)
    return [rng.choice(avail)]


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """
    Write a list of dict records to JSONL.

    Parameters
    ----------
    path:
        Output path to write.
    records:
        Each record becomes one JSON line.

    Notes
    -----
    - The parent folder is created automatically.
    - Output is UTF-8 and preserves non-ASCII characters (ensure_ascii=False).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
