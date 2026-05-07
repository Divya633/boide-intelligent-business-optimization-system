import numpy as np
from .data_loader import get_order_level


def compute_elasticity(df):
    """Price elasticity derived from real order data."""
    if "price" not in df.columns or "item_revenue" not in df.columns:
        return 0.4
    valid = df[["price", "item_revenue"]].dropna()
    if valid.empty:
        return 0.4
    corr = valid["price"].corr(valid["item_revenue"])
    return float(np.clip(abs(corr), 0.1, 0.9))


def simulate_business(df, price_change_pct: float,
                      marketing_spend_pct: float,
                      inventory_change_pct: float) -> dict:
    order_level = get_order_level(df)
    base_revenue = order_level["payment_value"].sum()
    base_orders  = order_level["order_id"].nunique()
    avg_order    = base_revenue / max(base_orders, 1)

    elasticity = compute_elasticity(df)

    price_effect     = -elasticity * (price_change_pct / 100)
    marketing_effect = 0.003 * marketing_spend_pct
    inventory_effect = min(0.15, 0.0015 * inventory_change_pct)

    projected_revenue = base_revenue * (
        1 + price_effect + marketing_effect + inventory_effect
    )

    base_cv    = order_level["payment_value"].std() / max(avg_order, 1)
    risk_score = min(100, base_cv * 10 + abs(price_change_pct) * 1.5)

    delta = projected_revenue - base_revenue

    return {
        "projected_revenue":    projected_revenue,
        "base_revenue":         base_revenue,
        "delta":                delta,
        "delta_pct":            (delta / base_revenue) * 100,
        "risk_score":           risk_score,
        "elasticity":           elasticity,
        "price_effect_pct":     price_effect * 100,
        "marketing_effect_pct": marketing_effect * 100,
        "inventory_effect_pct": inventory_effect * 100,
    }
