"""
Compare clustering methods across datasets in comparecluster.

Runs KMeans, HDBSCAN, and Agglomerative clustering on each dataset and
reports accuracy, ARI, NMI, silhouette, Davies-Bouldin, and Calinski-Harabasz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

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

from src.embedding.pipeline import get_all_embeddings  # noqa: E402

from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = contingency_matrix(y_true, y_pred)
    if cm.size == 0:
        return 0.0
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scipy is required for clustering accuracy") from e
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    return float(cm[row_ind, col_ind].sum() / cm.sum())


def compute_metrics(emb: np.ndarray, y_true: List, y_pred: np.ndarray) -> Dict[str, float | None]:
    le_true = LabelEncoder()
    le_pred = LabelEncoder()
    y_true_enc = le_true.fit_transform(y_true)
    y_pred_enc = le_pred.fit_transform(y_pred)

    acc = clustering_accuracy(y_true_enc, y_pred_enc)
    ari = float(adjusted_rand_score(y_true_enc, y_pred_enc))
    nmi = float(normalized_mutual_info_score(y_true_enc, y_pred_enc))

    sil = None
    db = None
    ch = None
    n_clusters = np.unique(y_pred_enc).shape[0]
    if n_clusters >= 2 and emb.shape[0] >= 3:
        try:
            sil = float(silhouette_score(emb, y_pred_enc, metric="cosine"))
        except Exception:
            sil = None
        try:
            db = float(davies_bouldin_score(emb, y_pred_enc))
        except Exception:
            db = None
        try:
            ch = float(calinski_harabasz_score(emb, y_pred_enc))
        except Exception:
            ch = None

    return {
        "accuracy": acc,
        "ARI": ari,
        "NMI": nmi,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
    }


def cluster_kmeans(emb: np.ndarray, n_clusters: int, seed: int, n_init: int) -> np.ndarray:
    if n_clusters <= 1 or emb.shape[0] == 0:
        return np.zeros(emb.shape[0], dtype=int)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    return km.fit_predict(emb)


def cluster_agglomerative(emb: np.ndarray, n_clusters: int, linkage: str) -> np.ndarray:
    if n_clusters <= 1 or emb.shape[0] == 0:
        return np.zeros(emb.shape[0], dtype=int)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return agg.fit_predict(emb)


def cluster_hdbscan(emb: np.ndarray, min_cluster_size: int, min_samples: int | None) -> np.ndarray:
    if emb.shape[0] < 2:
        return np.zeros(emb.shape[0], dtype=int)
    mcs = min(min_cluster_size, emb.shape[0])
    if mcs < 2:
        return np.zeros(emb.shape[0], dtype=int)
    try:
        import hdbscan
    except Exception as e:  # pragma: no cover
        raise RuntimeError("hdbscan is not installed. Try: pip install hdbscan") from e
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples)
    return clusterer.fit_predict(emb)


def format_metric(x: float | None) -> str:
    return "None" if x is None else f"{x:.4f}"


def resolve_datasets(data_dir: Path, names: str | None) -> List[Path]:
    if names:
        parts = [p.strip() for p in names.split(",") if p.strip()]
        paths = []
        for name in parts:
            p = Path(name)
            if not p.suffix:
                p = p.with_suffix(".jsonl")
            if not p.is_absolute():
                p = data_dir / p
            if not p.exists():
                raise FileNotFoundError(f"dataset not found: {p}")
            paths.append(p)
        return paths

    paths = sorted(data_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no .jsonl datasets found in {data_dir}")
    return paths


def run() -> None:
    parser = argparse.ArgumentParser(description="Compare clustering methods on datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/wdm/code/DataGene/comparecluster"),
        help="Directory with JSONL datasets (default: comparecluster).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names or paths (e.g., balance,compact).",
    )
    parser.add_argument("--model", type=str, default="instructor", help="Embedding model name.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--instruction", type=str, default="Represent the intent of this utterance for clustering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--kmeans-n-init", type=int, default=10, help="KMeans n_init.")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--hdbscan-min-samples", type=int, default=None, help="HDBSCAN min_samples.")
    parser.add_argument("--agglom-linkage", type=str, default="ward", help="Agglomerative linkage.")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize", help="Disable embedding normalization.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/wdm/code/DataGene/outputs/result.xlsx"),
        help="Output Excel file path.",
    )
    parser.add_argument(
        "--float-format",
        type=str,
        default="%.8f",
        help="Float format for Excel output (default: %.8f).",
    )
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    paths = resolve_datasets(data_dir, args.datasets)

    methods: List[str] = ["kmeans", "hdbscan", "agglomerative"]
    rows: List[Dict[str, object]] = []

    for path in paths:
        emb, labels = get_all_embeddings(
            data_path=str(path),
            model_name=args.model,
            batch_size=args.batch_size,
            normalize=args.normalize,
            instruction=args.instruction,
        )
        n_samples = emb.shape[0]
        n_labels = len(set(labels))

        print("\nDataset:", path.name)
        print("samples:", n_samples, "labels:", n_labels, "model:", args.model)

        for method in methods:
            if method == "kmeans":
                y_pred = cluster_kmeans(emb, n_labels, args.seed, args.kmeans_n_init)
            elif method == "hdbscan":
                y_pred = cluster_hdbscan(emb, args.hdbscan_min_cluster_size, args.hdbscan_min_samples)
            else:
                y_pred = cluster_agglomerative(emb, n_labels, args.agglom_linkage)

            metrics = compute_metrics(emb, labels, y_pred)
            print(
                f"{method:14s}",
                "acc=", format_metric(metrics["accuracy"]),
                "ari=", format_metric(metrics["ARI"]),
                "nmi=", format_metric(metrics["NMI"]),
                "sil=", format_metric(metrics["silhouette"]),
                "db=", format_metric(metrics["davies_bouldin"]),
                "ch=", format_metric(metrics["calinski_harabasz"]),
            )
            rows.append(
                {
                    "dataset": path.name,
                    "model": args.model,
                    "method": method,
                    "samples": n_samples,
                    "labels": n_labels,
                    "accuracy": metrics["accuracy"],
                    "ARI": metrics["ARI"],
                    "NMI": metrics["NMI"],
                    "silhouette": metrics["silhouette"],
                    "davies_bouldin": metrics["davies_bouldin"],
                    "calinski_harabasz": metrics["calinski_harabasz"],
                }
            )

    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required to write result.xlsx") from e

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False, float_format=args.float_format)
    print("Saved results to:", output_path)


if __name__ == "__main__":
    run()
