"""
Clustering + metrics utilities.

Implements:
- KMeans clustering (k = number of unique labels)
- Metrics: ARI, NMI, Silhouette, Davies–Bouldin
"""

from __future__ import annotations

from typing import Any, List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
)


def kmeans_cluster(
    embeddings: np.ndarray,
    *,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
) -> np.ndarray:
    """Run KMeans and return predicted cluster ids."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    return km.fit_predict(embeddings)


def compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: List[Any],
    *,
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple[np.ndarray, float, float, Optional[float], Optional[float]]:
    """
    Cluster with KMeans using k = #unique labels, then compute ARI/NMI/Silhouette/DB.

    Returns:
      y_pred, ari, nmi, silhouette (or None), davies_bouldin (or None)
    """
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    n_clusters = int(np.unique(y_true).shape[0])

    y_pred = kmeans_cluster(embeddings, n_clusters=n_clusters, random_state=random_state, n_init=n_init)

    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))

    # Silhouette/DB require at least 2 clusters and at least 2 samples per cluster in practice
    sil = None
    db = None
    if n_clusters >= 2 and embeddings.shape[0] >= 3:
        try:
            sil = float(silhouette_score(embeddings, y_pred, metric="cosine"))
        except Exception:
            sil = None
        try:
            db = float(davies_bouldin_score(embeddings, y_pred))
        except Exception:
            db = None

    return y_pred, ari, nmi, sil, db
