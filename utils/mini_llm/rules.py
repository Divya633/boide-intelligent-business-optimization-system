"""
rules.py
--------
Layer 2 of the Mini-LLM: Rule Engine.

Contains 20+ weighted IF-THEN rules across 5 business domains.
Each rule returns a RuleResult with:
  - tag       : "risk" | "growth" | "opportunity" | "neutral"
  - priority  : 1 (highest) to 5 (lowest) — controls display order
  - weight    : float, contributes to confidence score
  - insight   : the human-readable finding
  - action    : the recommended action
  - evidence  : the exact metric that triggered this rule (traceable)

All thresholds are annotated with their rationale so you can defend them
in a viva without memorising magic numbers.
"""

from dataclasses import dataclass


@dataclass
class RuleResult:
    tag:      str    # "risk" | "growth" | "opportunity" | "neutral"
    priority: int    # 1 = most urgent, 5 = informational
    weight:   float  # how much this rule contributes to confidence (0–1)
    insight:  str    # what the data shows
    action:   str    # what to do about it
    evidence: str    # the exact metric value that fired this rule


def _fmt(value, prefix="", suffix="", decimals=2):
    """Format a numeric value for display in insight strings."""
    return f"{prefix}{round(value, decimals)}{suffix}"


# ── DOMAIN 1: Revenue rules ──────────────────────────────────

def rule_revenue_scale(ctx: dict) -> RuleResult | None:
    """
    Revenue > R$1M is a signal to scale distribution.
    Threshold: R$1M chosen as the inflection point where logistics
    complexity typically justifies a dedicated fulfilment partner.
    """
    if ctx["revenue"] > 1_000_000:
        return RuleResult(
            tag="growth", priority=2, weight=0.8,
            insight=f"Total revenue {_fmt(ctx['revenue']/1e6, 'R$', 'M')} exceeds scale threshold",
            action="Invest in distribution partnerships and fulfilment automation",
            evidence=f"payment_value.sum() = R${ctx['revenue']:,.0f}"
        )
    return None


def rule_low_aov(ctx: dict) -> RuleResult | None:
    """
    AOV < R$150 signals a low-ticket product mix.
    Threshold: Olist dataset mean AOV ~R$120; below that, bundling is needed.
    """
    if ctx["avg_order"] < 150:
        return RuleResult(
            tag="risk", priority=2, weight=0.7,
            insight=f"Avg order value {_fmt(ctx['avg_order'], 'R$')} is below the R$150 threshold",
            action="Bundle complementary products; introduce minimum-order incentives",
            evidence=f"payment_value.mean() = R${ctx['avg_order']:.2f}"
        )
    return None


def rule_high_aov(ctx: dict) -> RuleResult | None:
    """
    AOV > R$300 signals a premium customer base worth nurturing.
    """
    if ctx["avg_order"] > 300:
        return RuleResult(
            tag="growth", priority=3, weight=0.6,
            insight=f"High AOV {_fmt(ctx['avg_order'], 'R$')} indicates premium buyer segment",
            action="Introduce a premium loyalty tier with early access and concierge service",
            evidence=f"payment_value.mean() = R${ctx['avg_order']:.2f}"
        )
    return None


def rule_revenue_volatility(ctx: dict) -> RuleResult | None:
    """
    Coefficient of Variation > 2 means demand is erratic.
    CV = std / mean; CV > 2 is considered high volatility in retail.
    """
    if ctx["cv"] > 2.0:
        return RuleResult(
            tag="risk", priority=1, weight=0.9,
            insight=f"Revenue CV={_fmt(ctx['cv'])} — demand is highly erratic",
            action="Introduce subscription products or long-term contracts to smooth revenue",
            evidence=f"std/mean = {ctx['cv']:.3f}"
        )
    elif ctx["cv"] > 1.0:
        return RuleResult(
            tag="risk", priority=3, weight=0.5,
            insight=f"Revenue CV={_fmt(ctx['cv'])} — moderate demand variability",
            action="Add demand forecasting alerts and safety stock buffers",
            evidence=f"std/mean = {ctx['cv']:.3f}"
        )
    return None


def rule_revenue_trend(ctx: dict) -> RuleResult | None:
    """
    Month-over-month revenue change: strong growth or decline signals.
    """
    t = ctx["revenue_trend_pct"]
    if t > 15:
        return RuleResult(
            tag="growth", priority=1, weight=0.85,
            insight=f"Revenue grew {_fmt(t, suffix='%')} last month — strong momentum",
            action="Scale inventory and ad spend now to capitalise on the trend",
            evidence=f"Monthly MoM = +{t:.1f}%"
        )
    elif t < -10:
        return RuleResult(
            tag="risk", priority=1, weight=0.85,
            insight=f"Revenue fell {_fmt(abs(t), suffix='%')} last month — demand contraction",
            action="Investigate root cause: price, competition, or seasonal effect",
            evidence=f"Monthly MoM = {t:.1f}%"
        )
    return None


# ── DOMAIN 2: Operations rules ───────────────────────────────

def rule_cancellation_rate(ctx: dict) -> RuleResult | None:
    """
    Cancel rate > 3%: industry benchmark for e-commerce is 1–2%.
    Above 3% signals a systemic problem in payment or logistics.
    """
    if ctx["cancel_rate"] > 5:
        return RuleResult(
            tag="risk", priority=1, weight=0.95,
            insight=f"Cancellation rate {_fmt(ctx['cancel_rate'], suffix='%')} is critically high",
            action="Audit payment gateway failures and contact logistics partner immediately",
            evidence=f"order_status == 'canceled' rate = {ctx['cancel_rate']:.2f}%"
        )
    elif ctx["cancel_rate"] > 3:
        return RuleResult(
            tag="risk", priority=2, weight=0.75,
            insight=f"Cancellation rate {_fmt(ctx['cancel_rate'], suffix='%')} exceeds 3% benchmark",
            action="Review checkout UX and payment failure notifications",
            evidence=f"order_status == 'canceled' rate = {ctx['cancel_rate']:.2f}%"
        )
    return None


def rule_anomaly_count(ctx: dict) -> RuleResult | None:
    """
    More than 5 anomalous days suggests operational instability.
    """
    n = ctx["anomaly_count"]
    if n > 10:
        return RuleResult(
            tag="risk", priority=1, weight=0.9,
            insight=f"{n} revenue anomalies detected — system is chronically unstable",
            action="Set up real-time revenue monitoring and automated alerts",
            evidence=f"IsolationForest detected {n} outlier days"
        )
    elif n > 5:
        return RuleResult(
            tag="risk", priority=2, weight=0.7,
            insight=f"{n} anomalous revenue days indicate frequent disruptions",
            action="Investigate anomaly dates for correlating events (holidays, outages)",
            evidence=f"IsolationForest detected {n} outlier days"
        )
    return None


# ── DOMAIN 3: Customer / Segment rules ───────────────────────

def rule_vip_segment(ctx: dict) -> RuleResult | None:
    """
    High-value customer % drives loyalty ROI decisions.
    < 5%: urgently need retention; > 10%: can scale loyalty programme.
    """
    if ctx["hv_pct"] is None:
        return None
    if ctx["hv_pct"] < 5:
        return RuleResult(
            tag="risk", priority=2, weight=0.75,
            insight=f"Only {_fmt(ctx['hv_pct'], suffix='%')} of customers are high-value",
            action="Launch retention programme: personalised offers, dedicated support",
            evidence=f"High-value cluster = {ctx['hv_pct']:.1f}% of total customers"
        )
    elif ctx["hv_pct"] > 10:
        return RuleResult(
            tag="opportunity", priority=3, weight=0.65,
            insight=f"{_fmt(ctx['hv_pct'], suffix='%')} VIP customer base is strong",
            action="Deploy exclusive VIP programme with early access and cashback rewards",
            evidence=f"High-value cluster = {ctx['hv_pct']:.1f}% of total customers"
        )
    return None


def rule_low_value_dominance(ctx: dict) -> RuleResult | None:
    """
    If > 60% of customers are low-value, acquisition strategy needs revision.
    """
    if ctx["lv_pct"] is None:
        return None
    if ctx["lv_pct"] > 60:
        return RuleResult(
            tag="risk", priority=2, weight=0.7,
            insight=f"{_fmt(ctx['lv_pct'], suffix='%')} of customers are low-value",
            action="Shift acquisition to premium channels; add upsell journey at checkout",
            evidence=f"Low-value cluster = {ctx['lv_pct']:.1f}% of total customers"
        )
    return None


# ── DOMAIN 4: Forecast rules ─────────────────────────────────

def rule_forecast_uplift(ctx: dict) -> RuleResult | None:
    """
    Forecast > baseline+10%: pre-position inventory now.
    Forecast < baseline-10%: take defensive action.
    """
    f = ctx["forecast_vs_baseline"]
    if f is None:
        return None
    if f > 10:
        return RuleResult(
            tag="growth", priority=1, weight=0.85,
            insight=f"7-day forecast is {_fmt(f, suffix='%')} above baseline",
            action="Pre-position stock for top 3 categories; alert logistics team",
            evidence=f"ARIMA forecast vs 7-day rolling avg = +{f:.1f}%"
        )
    elif f < -10:
        return RuleResult(
            tag="risk", priority=1, weight=0.85,
            insight=f"7-day forecast is {_fmt(abs(f), suffix='%')} below baseline",
            action="Run promotional campaign on slow-moving SKUs; review pricing",
            evidence=f"ARIMA forecast vs 7-day rolling avg = {f:.1f}%"
        )
    return None


# ── DOMAIN 5: Product / Review rules ─────────────────────────

def rule_avg_review(ctx: dict) -> RuleResult | None:
    """
    Avg review < 3.5 signals product or fulfilment quality issues.
    > 4.2 is a marketing asset worth promoting.
    """
    r = ctx["avg_review"]
    if r < 3.5:
        return RuleResult(
            tag="risk", priority=1, weight=0.9,
            insight=f"Avg review score {_fmt(r)}/5 is below acceptable threshold",
            action="Identify low-rated categories; address product quality or description accuracy",
            evidence=f"review_score.mean() = {r:.2f}"
        )
    elif r > 4.2:
        return RuleResult(
            tag="opportunity", priority=4, weight=0.5,
            insight=f"Strong review score {_fmt(r)}/5 — a marketable asset",
            action="Feature review score in ad creatives and product pages",
            evidence=f"review_score.mean() = {r:.2f}"
        )
    return None


def rule_low_review_concentration(ctx: dict) -> RuleResult | None:
    """
    If > 15% of reviews are 1 or 2 stars, there's a structural quality problem.
    """
    if ctx["low_review_pct"] > 15:
        return RuleResult(
            tag="risk", priority=2, weight=0.8,
            insight=f"{_fmt(ctx['low_review_pct'], suffix='%')} of reviews are 1–2 stars",
            action="Deploy post-purchase survey; identify top complaint categories",
            evidence=f"review_score <= 2 rate = {ctx['low_review_pct']:.1f}%"
        )
    return None


def rule_delivery_delay(ctx: dict) -> RuleResult | None:
    """
    Late delivery > 20% of orders is a significant customer experience risk.
    Olist data shows ~0.35 negative correlation between delay and review score.
    """
    if ctx["late_pct"] > 20:
        return RuleResult(
            tag="risk", priority=1, weight=0.85,
            insight=f"{_fmt(ctx['late_pct'], suffix='%')} of orders delivered late",
            action="Renegotiate carrier SLAs; add estimated delivery date buffer",
            evidence=f"delivery_delay_days > 0 rate = {ctx['late_pct']:.1f}%"
        )
    elif ctx["late_pct"] > 10:
        return RuleResult(
            tag="risk", priority=3, weight=0.6,
            insight=f"{_fmt(ctx['late_pct'], suffix='%')} late delivery rate — above benchmark",
            action="Monitor carrier performance weekly; set up delay alerts",
            evidence=f"delivery_delay_days > 0 rate = {ctx['late_pct']:.1f}%"
        )
    return None


def rule_category_concentration(ctx: dict) -> RuleResult | None:
    """
    Top-3 categories > 35% of revenue = concentration risk.
    """
    if ctx["top3_share"] > 50:
        return RuleResult(
            tag="risk", priority=2, weight=0.7,
            insight=f"Top 3 categories = {_fmt(ctx['top3_share'], suffix='%')} of revenue",
            action="Diversify category mix; promote underperforming categories",
            evidence=f"top3_share = {ctx['top3_share']:.1f}%"
        )
    elif ctx["top3_share"] > 35:
        return RuleResult(
            tag="opportunity", priority=4, weight=0.45,
            insight=f"Moderate category concentration at {_fmt(ctx['top3_share'], suffix='%')}",
            action="Expand top category depth while testing adjacent categories",
            evidence=f"top3_share = {ctx['top3_share']:.1f}%"
        )
    return None


# ── UPGRADE RULES ────────────────────────────────────────────

def rule_repeat_purchase(ctx: dict) -> RuleResult | None:
    """
    Repeat purchase rate < 5% is critically low.
    Olist dataset: only 3.12% of customers reorder.
    E-commerce industry average: 20-30%.
    This is the single most important retention signal.
    """
    r = ctx.get("repeat_rate", 0)
    if r < 5:
        return RuleResult(
            tag="risk", priority=1, weight=0.95,
            insight=f"Only {r:.1f}% of customers reorder — industry average is 20-30%",
            action="Launch 7-day post-purchase email with personalised recommendations. "
                   "Add loyalty points redeemable on next order. Target Occasional segment first.",
            evidence=f"repeat_rate = {r:.2f}% ({ctx.get('repeat_customers',0):,} repeat customers)"
        )
    elif r < 15:
        return RuleResult(
            tag="risk", priority=2, weight=0.7,
            insight=f"Repeat purchase rate {r:.1f}% is below the 20-30% industry benchmark",
            action="Introduce a subscription tier for top categories. Send re-engagement at day 30 post-delivery.",
            evidence=f"repeat_rate = {r:.2f}%"
        )
    return None


def rule_freight_burden(ctx: dict) -> RuleResult | None:
    """
    Freight = 32.2% of product price on average.
    When > 40%, customers perceive poor value — churn risk.
    Threshold: >35% freight ratio is an industry warning sign.
    """
    fr = ctx.get("freight_ratio", 0)
    hf = ctx.get("high_freight_pct", 0)
    if fr > 40:
        return RuleResult(
            tag="risk", priority=2, weight=0.8,
            insight=f"Freight cost is {fr:.1f}% of product price — customers perceive poor value",
            action="Introduce free shipping threshold at R$150+. Negotiate carrier rates for top-volume categories. "
                   "Highlight 'free shipping' in ad copy to improve conversion.",
            evidence=f"avg freight/price ratio = {fr:.1f}% | {hf:.1f}% of orders have freight > 40% of price"
        )
    elif fr > 30:
        return RuleResult(
            tag="risk", priority=3, weight=0.55,
            insight=f"Freight at {fr:.1f}% of price is high — may suppress repeat purchases",
            action="A/B test free shipping on orders above R$100 to measure conversion uplift.",
            evidence=f"avg freight/price ratio = {fr:.1f}%"
        )
    return None


def rule_installment_risk(ctx: dict) -> RuleResult | None:
    """
    16.6% of orders use 6+ installments — long-term credit exposure.
    High installment usage correlates with lower customer lifetime value
    and higher churn probability after final payment.
    """
    hi = ctx.get("high_installment_pct", 0)
    ai = ctx.get("avg_installments", 1)
    if hi > 20:
        return RuleResult(
            tag="risk", priority=2, weight=0.7,
            insight=f"{hi:.1f}% of orders use 6+ installments — high credit exposure",
            action="Flag 12+ installment customers for proactive retention outreach at month 6. "
                   "Offer early payoff incentive (discount on next order) to reduce churn risk.",
            evidence=f"orders with 6+ installments = {hi:.1f}% | avg installments = {ai:.1f}"
        )
    return None


def rule_seller_concentration(ctx: dict) -> RuleResult | None:
    """
    Top 10 sellers = 14.1% of revenue across 3,095 sellers.
    If top sellers leave, significant revenue is at risk.
    Seller diversification is a marketplace health metric.
    """
    share = ctx.get("top10_seller_share", 0)
    total = ctx.get("total_sellers", 0)
    if share > 20:
        return RuleResult(
            tag="risk", priority=2, weight=0.65,
            insight=f"Top 10 sellers account for {share:.1f}% of revenue — seller dependency risk",
            action="Build seller retention programme for top performers. "
                   "Diversify by promoting mid-tier sellers with incentivised listing fees.",
            evidence=f"top10_seller_share = {share:.1f}% across {total:,} total sellers"
        )
    return None


def rule_review_sentiment(ctx: dict) -> RuleResult | None:
    """
    Keyword-based NLP on review_comment_message.
    Delivery complaints and quality complaints are separate signals
    requiring different business responses.
    """
    dc = ctx.get("delivery_complaint_pct", 0)
    qc = ctx.get("quality_complaint_pct", 0)
    if dc > 5:
        return RuleResult(
            tag="risk", priority=1, weight=0.85,
            insight=f"{dc:.1f}% of reviews mention delivery issues in comments",
            action="Escalate to logistics partner. Add proactive delay SMS alerts. "
                   "Offer automatic R$20 voucher for late deliveries > 7 days.",
            evidence=f"delivery_complaint_pct = {dc:.1f}% (keywords: atraso, demora, entrega)"
        )
    if qc > 3:
        return RuleResult(
            tag="risk", priority=2, weight=0.75,
            insight=f"{qc:.1f}% of reviews mention product quality issues in comments",
            action="Audit sellers in flagged categories. Require photo verification on listings. "
                   "Implement 3-strike seller quality policy.",
            evidence=f"quality_complaint_pct = {qc:.1f}% (keywords: quebrado, defeito, errado)"
        )
    return None


# ── Rule registry ─────────────────────────────────────────────

ALL_RULES = [
    # Original 14 rules
    rule_revenue_scale,
    rule_low_aov,
    rule_high_aov,
    rule_revenue_volatility,
    rule_revenue_trend,
    rule_cancellation_rate,
    rule_anomaly_count,
    rule_vip_segment,
    rule_low_value_dominance,
    rule_forecast_uplift,
    rule_avg_review,
    rule_low_review_concentration,
    rule_delivery_delay,
    rule_category_concentration,
    # Upgrade rules (5 new)
    rule_repeat_purchase,
    rule_freight_burden,
    rule_installment_risk,
    rule_seller_concentration,
    rule_review_sentiment,
]


def run_all_rules(ctx: dict) -> list[RuleResult]:
    """Fire all rules against the context. Return only those that triggered."""
    results = []
    for rule_fn in ALL_RULES:
        result = rule_fn(ctx)
        if result is not None:
            results.append(result)
    return results


    