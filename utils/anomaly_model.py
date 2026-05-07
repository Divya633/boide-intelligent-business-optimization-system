from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .data_loader import get_order_level


@dataclass
class AnomalyDetectionResult:
    daily_features: pd.DataFrame
    anomalies: pd.DataFrame
    anomaly_count: int
    anomaly_rate: float
    high_severity_count: int
    severity_counts: Dict[str, int]
    contamination: float


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    order_level = get_order_level(df).copy()
    order_level["date"] = pd.to_datetime(order_level["order_purchase_timestamp"]).dt.floor("D")

    daily = (
        order_level.groupby("date")
        .agg(
            revenue=("payment_value", "sum"),
            orders=("order_id", "nunique"),
            avg_order_value=("payment_value", "mean"),
            cancel_rate=("order_status", lambda x: (x == "canceled").mean() * 100),
            avg_review=("review_score", "mean"),
            late_delivery_rate=("delivery_delay_days", lambda x: (x > 0).mean() * 100),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    numeric_cols = ["revenue", "orders", "avg_order_value", "cancel_rate", "avg_review", "late_delivery_rate"]
    for col in numeric_cols:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
        daily[col] = daily[col].interpolate().ffill().bfill()

    return daily


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> AnomalyDetectionResult:
    daily = build_daily_features(df)
    feature_cols = ["revenue", "orders", "avg_order_value", "cancel_rate", "avg_review", "late_delivery_rate"]

    model = IsolationForest(contamination=contamination, random_state=42)
    daily = daily.copy()
    daily["anomaly"] = model.fit_predict(daily[feature_cols])
    daily["anomaly_score"] = -model.decision_function(daily[feature_cols])

    revenue_mean = float(daily["revenue"].mean())
    revenue_std = float(daily["revenue"].std()) if float(daily["revenue"].std()) > 0 else 1.0
    daily["z_score"] = ((daily["revenue"] - revenue_mean) / revenue_std).abs()

    score_mean = float(daily["anomaly_score"].mean())
    score_std = float(daily["anomaly_score"].std()) if float(daily["anomaly_score"].std()) > 0 else 1.0
    daily["severity_score"] = 0.65 * daily["z_score"] + 0.35 * ((daily["anomaly_score"] - score_mean) / score_std).abs()

    def classify_severity(row: pd.Series) -> str:
        score = float(row["severity_score"])
        if score >= 3.0 or float(row["z_score"]) >= 3.0:
            return "High"
        if score >= 2.0 or float(row["z_score"]) >= 2.0:
            return "Medium"
        return "Low"

    anomalies = daily[daily["anomaly"] == -1].copy()
    anomalies["severity"] = anomalies.apply(classify_severity, axis=1)
    anomalies["possible_cause"] = anomalies["severity"].apply(
        lambda x: "System issue / payment fraud spike" if x == "High" else "Demand spike / campaign / seasonal shift"
    )
    anomalies["impact"] = anomalies["severity"].apply(
        lambda x: "High business risk - investigate immediately" if x == "High" else "Moderate variance - monitor and validate"
    )

    severity_counts = anomalies["severity"].value_counts().to_dict()

    return AnomalyDetectionResult(
        daily_features=daily,
        anomalies=anomalies,
        anomaly_count=int(len(anomalies)),
        anomaly_rate=float(len(anomalies) / max(len(daily), 1) * 100),
        high_severity_count=int((anomalies["severity"] == "High").sum()),
        severity_counts={k: int(v) for k, v in severity_counts.items()},
        contamination=float(contamination),
    )
