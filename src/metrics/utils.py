"""
Utilities for centroid-based weighting over embedding clusters.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple
from pathlib import Path

import numpy as np

from src.synthetic.dataprocesser import infer_text_and_label_keys, load_jsonl, write_jsonl


def _as_label_list(chosen_labels: Any) -> List[Any]:
    if chosen_labels is None:
        return []
    if isinstance(chosen_labels, (list, tuple, set)):
        return list(chosen_labels)
    return [chosen_labels]


def compute_centroids(
    embeddings: np.ndarray,
    labels: Sequence[Any],
) -> Dict[Any, np.ndarray]:
    """
    Compute centroid for each label.

    Args:
        embeddings: (N, D) array.
        labels: sequence of length N.

    Returns:
        dict[label] -> centroid (D,)
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D (N, D)")
    if len(labels) != embeddings.shape[0]:
        raise ValueError("labels length must match embeddings rows")

    centroids: Dict[Any, np.ndarray] = {}
    labels_arr = np.asarray(labels, dtype=object)
    for lab in sorted(set(labels_arr.tolist()), key=lambda x: str(x)):
        idx = labels_arr == lab
        if not np.any(idx):
            continue
        centroids[lab] = embeddings[idx].mean(axis=0)
    return centroids


def compute_weights_for_chosen_labels(
    embeddings: np.ndarray,
    labels: Sequence[Any],
    chosen_labels: Any,
    *,
    pick_type: str = "imbalance",
    eps: float = 1e-8,
    use_inverse: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """
    Compute weights for samples belonging to chosen labels.

    pick_type:
      - imbalance: weights are all ones
      - compact: weights are w1 (distance to own centroid, or inverse)
      - diversity: weights are w2 (sum distance to other centroids, or inverse)

    Returns:
        weights, chosen_indices, chosen_label_list
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D (N, D)")
    if len(labels) != embeddings.shape[0]:
        raise ValueError("labels length must match embeddings rows")

    chosen_list = _as_label_list(chosen_labels)
    if not chosen_list:
        raise ValueError("chosen_labels must be provided (non-empty)")

    centroids = compute_centroids(embeddings, labels)
    if not centroids:
        raise ValueError("no centroids computed (empty labels?)")

    labels_arr = np.asarray(labels, dtype=object)
    chosen_mask = np.isin(labels_arr, np.asarray(chosen_list, dtype=object))
    chosen_indices = np.nonzero(chosen_mask)[0]
    if chosen_indices.size == 0:
        raise ValueError("no samples found for chosen_labels")

    # Build centroid matrix and mapping
    centroid_labels = list(centroids.keys())
    centroid_matrix = np.stack([centroids[lab] for lab in centroid_labels], axis=0)  # (K, D)
    label_to_centroid_idx = {lab: i for i, lab in enumerate(centroid_labels)}

    # Distances from each chosen sample to each centroid
    x = embeddings[chosen_indices]  # (M, D)
    diffs = x[:, None, :] - centroid_matrix[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)  # (M, K)

    # Own-centroid distance
    own_idx = np.array([label_to_centroid_idx[lab] for lab in labels_arr[chosen_indices]])
    weight_1 = dists[np.arange(dists.shape[0]), own_idx]

    # Min distance to other centroids
    if dists.shape[1] > 1:
        dists_other = dists.copy()
        dists_other[np.arange(dists.shape[0]), own_idx] = np.inf
        weight_2 = dists_other.min(axis=1)
    else:
        weight_2 = np.zeros_like(weight_1)

    if use_inverse:
        weight_1 = 1.0 / (weight_1 + eps)
        weight_2 = 1.0 / (weight_2 + eps)

    ptype = pick_type.strip().lower()
    if ptype == "imbalance":
        weights = np.ones_like(weight_1, dtype=np.float64)
    elif ptype == "compact":
        weights = weight_1
    elif ptype == "diversity":
        weights = weight_2
    else:
        raise ValueError("pick_type must be one of: 'imbalance', 'compact', 'diversity'")

    return weights, chosen_indices, chosen_list


def compute_weights_for_chosen_labels_cosine(
    embeddings: np.ndarray,
    labels: Sequence[Any],
    chosen_labels: Any,
    *,
    eps: float = 1e-8,
    use_inverse: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """
    Compute weights using cosine distance (1 - cosine similarity).

    weight_1: distance to its own cluster centroid (or inverse if use_inverse=True)
    weight_2: sum of distances to other centroids (or inverse if use_inverse=True)
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D (N, D)")
    if len(labels) != embeddings.shape[0]:
        raise ValueError("labels length must match embeddings rows")

    chosen_list = _as_label_list(chosen_labels)
    if not chosen_list:
        raise ValueError("chosen_labels must be provided (non-empty)")

    centroids = compute_centroids(embeddings, labels)
    if not centroids:
        raise ValueError("no centroids computed (empty labels?)")

    labels_arr = np.asarray(labels, dtype=object)
    chosen_mask = np.isin(labels_arr, np.asarray(chosen_list, dtype=object))
    chosen_indices = np.nonzero(chosen_mask)[0]
    if chosen_indices.size == 0:
        raise ValueError("no samples found for chosen_labels")

    centroid_labels = list(centroids.keys())
    centroid_matrix = np.stack([centroids[lab] for lab in centroid_labels], axis=0)  # (K, D)
    label_to_centroid_idx = {lab: i for i, lab in enumerate(centroid_labels)}

    x = embeddings[chosen_indices]  # (M, D)

    # Normalize for cosine distance
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)
    c_norm = centroid_matrix / (np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + eps)
    sims = x_norm @ c_norm.T  # (M, K)
    dists = 1.0 - sims

    own_idx = np.array([label_to_centroid_idx[lab] for lab in labels_arr[chosen_indices]])
    weight_1 = dists[np.arange(dists.shape[0]), own_idx]

    if dists.shape[1] > 1:
        dists_other = dists.copy()
        dists_other[np.arange(dists.shape[0]), own_idx] = np.inf
        weight_2 = dists_other.min(axis=1)
    else:
        weight_2 = np.zeros_like(weight_1)

    if use_inverse:
        weight_1 = 1.0 / (weight_1 + eps)
        weight_2 = 1.0 / (weight_2 + eps)

    return weight_1, weight_2, chosen_indices, chosen_list


def weighted_random_sample(
    samples: Sequence[Any],
    weights: Sequence[float],
    k: int,
    *,
    replace: bool = False,
    seed: int | None = None,
) -> List[Any]:
    """
    Weighted random selection of samples.

    Args:
        samples: items to sample from.
        weights: non-negative weights, same length as samples.
        k: number of items to select.
        replace: sample with replacement if True.
        seed: optional RNG seed.

    Returns:
        list of selected samples.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if len(samples) != len(weights):
        raise ValueError("samples and weights must have the same length")
    if len(samples) == 0:
        return []
    if not replace and k > len(samples):
        raise ValueError("k cannot exceed number of samples when replace=False")

    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if np.all(w == 0):
        w = np.ones_like(w, dtype=np.float64)

    rng = np.random.default_rng(seed)
    N = len(w)  # 10
    wmax, wmin = w.max(), w.min()
    if N == 1 or wmax == wmin:
        p = np.ones_like(w, dtype=np.float64)
        p = p / p.sum()
    else:
        # Stabilize gamma and probabilities to avoid overflow/NaN.
        wpos = w[w > 0]
        wmin_pos = wpos.min() if wpos.size else 0.0
        if wmin_pos <= 0:
            p = w / w.sum()
        else:
            ratio = wmax / wmin_pos
            if np.isclose(ratio, 1.0, rtol=1e-6):
                p = np.ones_like(w, dtype=np.float64)
                p = p / p.sum()
            else:
                gamma = np.log(N) / np.log(ratio)
                logw = np.where(w > 0, np.log(w), -np.inf)
                logp = gamma * logw
                logp -= np.max(logp)
                p = np.exp(logp)
                p = p / p.sum()
    
    idx = rng.choice(len(samples), size=k, replace=replace, p=p)
    # print("chosen idx:", idx, "chosen w:", w[idx], "chosen p:", p[idx])
    return [samples[i] for i in idx]


def example_pick(
    *,
    samples: Sequence[Any],
    weights: Sequence[float],
    num_examples: int,
    seed: int | None = None,
) -> List[Any]:
    """
    Pick examples using provided weights.
    """
    if num_examples < 0:
        raise ValueError("num_examples must be >= 0")
    if num_examples == 0:
        return []
    if len(samples) == 0:
        return []

    k = min(num_examples, len(samples))
    return weighted_random_sample(samples, weights, k, replace=False, seed=seed)


def remove_task_clinc(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    If record has {'task': 'CLINC'} (case-insensitive), remove the 'task' key.
    """
    if "task" in rec and isinstance(rec["task"], str) and rec["task"].strip().lower() == "clinc":
        rec = dict(rec)
        rec.pop("task", None)
    return rec


def merge_synthetic_with_original(
    *,
    data_name: str,
    data_choice: str,
    synthetic_dir: Path | str = Path("./code/DataGene/synthetic_data"),
    merged_path: Path | str | None = None,
    duplicate: bool = True,
) -> Path:
    """
    Merge all synthetic JSONL files in synthetic_dir with the original dataset.

    If duplicate=False, remove original rows whose labels are present in synthetic data
    before merging. Always applies remove_task_clinc to merged output.
    """
    synthetic_dir = Path(synthetic_dir)
    if merged_path is None:
        merged_path = synthetic_dir / "merge.jsonl"
    merged_path = Path(merged_path)

    original_path = Path("./code/DataGene/datasets") / data_name / f"{data_choice}.jsonl"
    original_rows = load_jsonl(original_path)
    if not original_rows:
        raise ValueError(f"No rows found in original dataset: {original_path}")

    _, label_key = infer_text_and_label_keys(original_rows)
    original_labels = {str(r.get(label_key, "")).strip() for r in original_rows if str(r.get(label_key, "")).strip()}

    syn_rows: List[Dict[str, Any]] = []
    for p in sorted(synthetic_dir.glob("*.jsonl")):
        if p.name == merged_path.name:
            continue
        syn_rows.extend(load_jsonl(p))

    if not syn_rows:
        raise ValueError(f"No synthetic JSONL files found in {synthetic_dir}")

    _, syn_label_key = infer_text_and_label_keys(syn_rows)
    syn_labels = sorted({str(r.get(syn_label_key, "")).strip() for r in syn_rows if str(r.get(syn_label_key, "")).strip()})

    missing = [lab for lab in syn_labels if lab not in original_labels]
    if missing:
        raise ValueError(f"Synthetic label(s) not found in original dataset: {missing}")

    if duplicate:
        merged = list(original_rows) + list(syn_rows)
    else:
        syn_set = set(syn_labels)
        filtered = [r for r in original_rows if str(r.get(label_key, "")).strip() not in syn_set]
        merged = list(filtered) + list(syn_rows)

    merged = [remove_task_clinc(r) for r in merged]
    # write_jsonl(merged_path, merged)
    return merged


def extract_data(
    rows: List[Dict[str, Any]],
    select_label: Any,
) -> List[Dict[str, Any]]:
    """
    Extract rows for a single label.

    Args:
        rows: list[dict] loaded from JSONL.
        select_label: label name to extract (must exist in data).

    Returns:
        list[dict] containing all rows for the selected label.
    """
    if not rows:
        return []

    _, label_key = infer_text_and_label_keys(rows)
    labels_all = sorted({str(r.get(label_key, "")).strip() for r in rows if str(r.get(label_key, "")).strip()})
    if not labels_all:
        raise ValueError("No labels found in rows.")

    select_label_str = str(select_label)
    if select_label_str not in labels_all:
        raise ValueError(f"select_label {select_label!r} not found in labels.")

    extracted = [r for r in rows if str(r.get(label_key, "")).strip() == select_label_str]
    return extracted
