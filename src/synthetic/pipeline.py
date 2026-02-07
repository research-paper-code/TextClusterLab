"""
High-level helpers for synthetic data generation (single-label input).

Input data is already in list[dict] format and contains exactly one label.
This module does not perform label selection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataprocesser import group_texts_by_label, infer_text_and_label_keys, write_jsonl
from .model import generate_synthetic


def _get_single_label(rows: List[Dict[str, Any]]) -> str:
    """
    Validate that rows contain exactly one label and return it.
    """
    text_key, label_key = infer_text_and_label_keys(rows)
    label2texts = group_texts_by_label(rows, text_key, label_key)

    labels = sorted(label2texts.keys())
    if not labels:
        raise ValueError("No labels found in dataset.")
    if len(labels) != 1:
        raise ValueError(f"Input data must contain exactly one label, found {len(labels)}: {labels}")
    return labels[0]


def generate_from_rows(
    *,
    rows: List[Dict[str, Any]],
    n_new: int,
    model_select: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic examples from in-memory rows (single label).
    """
    return generate_synthetic(
        data={"rows": rows},
        n_new=n_new,
        model_select=model_select,
    )


def generate_and_save(
    *,
    rows: List[Dict[str, Any]],
    out_dir: Path,
    n_new: int,
    model_select: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic examples and write them to {label}.jsonl in out_dir.
    """
    label = _get_single_label(rows)
    safe_label = str(label).strip().replace(os.sep, "_")
    out_path = Path(out_dir) / f"{safe_label}.jsonl"
    records = generate_from_rows(
        rows=rows,
        n_new=n_new,
        model_select=model_select,
    )
    write_jsonl(out_path, records)
    print(f"[DONE] Wrote {len(records)} synthetic examples to: {out_path}")
    return records
