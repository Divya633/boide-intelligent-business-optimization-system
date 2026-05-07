from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data_loader import get_order_level


@dataclass
class SegmentationResult:
    customers: pd.DataFrame
    order_level: pd.DataFrame
    silhouette_score: float
    selected_k: int
    candidate_scores: List[Dict[str, float]]
    segment_counts: Dict[str, int]
    segment_summary: pd.DataFrame


def _build_customer_rfm(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order_level = (
        get_order_level(df)
        [["customer_unique_id", "order_id", "payment_value", "order_purchase_timestamp"]]
        .dropna(subset=["customer_unique_id", "order_id", "payment_value", "order_purchase_timestamp"])
    )
    order_level = order_level[order_level["payment_value"] > 0].copy()
    snapshot = order_level["order_purchase_timestamp"].max()

    customer = (
        order_level.groupby("customer_unique_id")
        .agg(
            monetary=("payment_value", "sum"),
            frequency=("order_id", "nunique"),
            recency=("order_purchase_timestamp", lambda x: (snapshot - x.max()).days),
        )
        .reset_index()
    )
    customer = customer[(customer["monetary"] > 0) & (customer["frequency"] > 0)].reset_index(drop=True)
    return customer, order_level


def _prepare_features(customer: pd.DataFrame) -> np.ndarray:
    features = customer[["monetary", "frequency", "recency"]].copy()
    features["monetary"] = np.log1p(features["monetary"])
    features["frequency"] = np.log1p(features["frequency"])
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def _sample_for_scoring(values: np.ndarray, max_rows: int = 10000) -> np.ndarray:
    if len(values) <= max_rows:
        return np.arange(len(values))
    rng = np.random.default_rng(42)
    return rng.choice(len(values), size=max_rows, replace=False)


def _label_segments(customer: pd.DataFrame) -> pd.DataFrame:
    cluster_mean = customer.groupby("cluster")["monetary"].mean()
    freq_mean = customer.groupby("cluster")["frequency"].mean()
    rec_mean = customer.groupby("cluster")["recency"].mean()

    high_cluster = int(cluster_mean.idxmax())
    low_cluster = int(cluster_mean.idxmin())

    def label_row(row: pd.Series) -> str:
        cluster = int(row["cluster"])
        if cluster == high_cluster:
            return "High Value"
        if cluster == low_cluster:
            if freq_mean[cluster] >= freq_mean.median():
                return "Discount-Sensitive"
            return "Occasional"
        if row["recency"] <= rec_mean.median() and row["frequency"] >= freq_mean.median():
            return "Loyal"
        return "Mid Value"

    labelled = customer.copy()
    labelled["segment"] = labelled.apply(label_row, axis=1)
    return labelled


def run_segmentation(df: pd.DataFrame, candidate_k: tuple[int, ...] = (3, 4, 5)) -> SegmentationResult:
    customer, order_level = _build_customer_rfm(df)
    scaled = _prepare_features(customer)
    sample_idx = _sample_for_scoring(scaled)

    candidate_scores: List[Dict[str, float]] = []
    best_score = -1.0
    best_k = candidate_k[0]
    best_labels = None

    for k in candidate_k:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        score = float(silhouette_score(scaled[sample_idx], labels[sample_idx]))
        candidate_scores.append({"k": int(k), "silhouette_score": round(score, 4)})
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    customer = customer.copy()
    customer["cluster"] = best_labels if best_labels is not None else np.zeros(len(customer), dtype=int)
    customer = _label_segments(customer)

    segment_counts_series = customer["segment"].value_counts()
    segment_counts = {name: int(value) for name, value in segment_counts_series.items()}
    segment_summary = (
        customer.groupby("segment")
        .agg(
            customers=("customer_unique_id", "count"),
            avg_spend=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
            avg_orders=("frequency", "mean"),
            avg_recency=("recency", "mean"),
        )
        .reset_index()
    )

    return SegmentationResult(
        customers=customer,
        order_level=order_level,
        silhouette_score=float(best_score),
        selected_k=int(best_k),
        candidate_scores=candidate_scores,
        segment_counts=segment_counts,
        segment_summary=segment_summary,
    )
