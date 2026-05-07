"""
context_builder.py
------------------
Layer 1 of the Mini-LLM: feature extraction.

Builds a traceable context dictionary from the merged Olist dataframe plus
cross-module session signals.
"""

import pandas as pd

from ..data_loader import get_order_level


def build_context(
    df: pd.DataFrame,
    forecast: float | None = None,
    segments: dict | None = None,
    anomaly_count: int = 0,
    forecast_details: dict | None = None,
    segment_details: dict | None = None,
    anomaly_details: dict | None = None,
) -> dict:
    """Extract the typed signals used by the rule engine."""
    order_level = get_order_level(df)

    revenue = float(order_level["payment_value"].sum())
    avg_order = float(order_level["payment_value"].mean())
    std_order = float(order_level["payment_value"].std())
    cv = std_order / avg_order if avg_order > 0 else 0.0

    monthly = (
        order_level.groupby(order_level["order_purchase_timestamp"].dt.to_period("M"))["payment_value"]
        .sum()
    )
    if len(monthly) >= 2 and monthly.iloc[-2] != 0:
        revenue_trend_pct = float((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100)
    else:
        revenue_trend_pct = 0.0

    total_orders = int(order_level["order_id"].nunique())
    cancel_rate = float((order_level["order_status"] == "canceled").sum() / len(order_level) * 100)

    if "review_score" in order_level.columns:
        review_non_null = order_level["review_score"].notna().sum()
        avg_review = float(order_level["review_score"].mean())
        low_review_pct = float(
            (order_level["review_score"] <= 2).sum() / review_non_null * 100
        ) if review_non_null else 0.0
    else:
        avg_review = 4.0
        low_review_pct = 0.0

    if "delivery_delay_days" in order_level.columns:
        avg_delay = float(order_level["delivery_delay_days"].mean())
        late_pct = float((order_level["delivery_delay_days"] > 0).sum() / len(order_level) * 100)
        delay_review_corr = float(
            order_level["delivery_delay_days"].corr(order_level["review_score"])
        ) if "review_score" in order_level.columns else 0.0
    else:
        avg_delay = 0.0
        late_pct = 0.0
        delay_review_corr = 0.0

    if "category_en" in df.columns and "item_revenue" in df.columns:
        cat_rev = df.groupby("category_en")["item_revenue"].sum()
        top3_share = float(cat_rev.nlargest(3).sum() / cat_rev.sum() * 100) if not cat_rev.empty else 0.0
        top_category = str(cat_rev.idxmax()) if not cat_rev.empty else "unknown"
    else:
        top3_share = 0.0
        top_category = "unknown"

    observed_days = max(
        (order_level["order_purchase_timestamp"].max() - order_level["order_purchase_timestamp"].min()).days + 1,
        1,
    )
    daily_avg_revenue = revenue / observed_days
    if forecast is not None and daily_avg_revenue > 0:
        forecast_vs_baseline = float((forecast - daily_avg_revenue * 7) / (daily_avg_revenue * 7) * 100)
    else:
        forecast_vs_baseline = None

    forecast_accuracy = None
    forecast_confidence = None
    forecast_model = None
    if forecast_details:
        forecast_accuracy = forecast_details.get("mape")
        forecast_confidence = forecast_details.get("confidence")
        forecast_model = forecast_details.get("model_label")

    if segments is not None:
        total_customers = max(segments.get("total", 1), 1)
        hv_pct = float(segments.get("high_value", 0) / total_customers * 100)
        lv_pct = float(segments.get("low_value", 0) / total_customers * 100)
    else:
        hv_pct = None
        lv_pct = None

    segment_quality = None
    segment_k = None
    if segment_details:
        segment_quality = segment_details.get("silhouette_score")
        segment_k = segment_details.get("selected_k")

    cust_orders = order_level.groupby("customer_unique_id")["order_id"].nunique()
    repeat_customers = int((cust_orders > 1).sum())
    repeat_rate = float(repeat_customers / max(len(cust_orders), 1) * 100)

    if "freight_value" in df.columns and "price" in df.columns:
        valid = df[(df["price"] > 0) & df["freight_value"].notna()]
        freight_ratio = float((valid["freight_value"] / valid["price"]).mean() * 100) if not valid.empty else 0.0
        high_freight_pct = float((valid["freight_value"] / valid["price"] > 0.4).mean() * 100) if not valid.empty else 0.0
    else:
        freight_ratio = 0.0
        high_freight_pct = 0.0

    if "payment_installments" in order_level.columns:
        avg_installments = float(order_level["payment_installments"].mean())
        high_installment_pct = float((order_level["payment_installments"] >= 6).sum() / len(order_level) * 100)
    else:
        avg_installments = 1.0
        high_installment_pct = 0.0

    if "seller_id" in df.columns and "item_revenue" in df.columns:
        seller_rev = df.groupby("seller_id")["item_revenue"].sum()
        top10_seller_share = (
            float(seller_rev.nlargest(10).sum() / seller_rev.sum() * 100) if not seller_rev.empty else 0.0
        )
        total_sellers = int(df["seller_id"].nunique())
    else:
        top10_seller_share = 0.0
        total_sellers = 0

    if "review_comment_message" in df.columns:
        delivery_kw = ["atraso", "demora", "prazo", "entrega", "nao chegou", "chegou errado"]
        quality_kw = ["produto", "qualidade", "quebrado", "errado", "defeito", "ruim"]

        def _classify(text):
            if not isinstance(text, str) or text.strip() == "":
                return "no_comment"
            text = text.lower()
            if any(word in text for word in delivery_kw):
                return "delivery_issue"
            if any(word in text for word in quality_kw):
                return "quality_issue"
            return "positive"

        sentiment = df["review_comment_message"].apply(_classify)
        delivery_complaint_pct = float((sentiment == "delivery_issue").sum() / len(df) * 100)
        quality_complaint_pct = float((sentiment == "quality_issue").sum() / len(df) * 100)
    else:
        delivery_complaint_pct = 0.0
        quality_complaint_pct = 0.0

    anomaly_rate = None
    high_severity_anomalies = None
    anomaly_severity_mix = None
    if anomaly_details:
        anomaly_rate = anomaly_details.get("anomaly_rate")
        high_severity_anomalies = anomaly_details.get("high_severity_count")
        anomaly_severity_mix = anomaly_details.get("severity_counts")

    return {
        "revenue": revenue,
        "avg_order": avg_order,
        "cv": cv,
        "revenue_trend_pct": revenue_trend_pct,
        "total_orders": total_orders,
        "cancel_rate": cancel_rate,
        "avg_review": avg_review,
        "low_review_pct": low_review_pct,
        "avg_delay": avg_delay,
        "late_pct": late_pct,
        "delay_review_corr": delay_review_corr,
        "top3_share": top3_share,
        "top_category": top_category,
        "forecast_vs_baseline": forecast_vs_baseline,
        "forecast_accuracy": forecast_accuracy,
        "forecast_confidence": forecast_confidence,
        "forecast_model": forecast_model,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "high_severity_anomalies": high_severity_anomalies,
        "anomaly_severity_mix": anomaly_severity_mix,
        "hv_pct": hv_pct,
        "lv_pct": lv_pct,
        "segment_quality": segment_quality,
        "segment_k": segment_k,
        "repeat_rate": repeat_rate,
        "repeat_customers": repeat_customers,
        "freight_ratio": freight_ratio,
        "high_freight_pct": high_freight_pct,
        "avg_installments": avg_installments,
        "high_installment_pct": high_installment_pct,
        "top10_seller_share": top10_seller_share,
        "total_sellers": total_sellers,
        "delivery_complaint_pct": delivery_complaint_pct,
        "quality_complaint_pct": quality_complaint_pct,
    }
