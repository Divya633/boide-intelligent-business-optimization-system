"""
scorer.py
---------
Layer 3 of the Mini-LLM: Confidence Scorer + Insight Ranker.

Two responsibilities:
  1. Compute a confidence score (0100) based on data completeness
     and the cumulative weight of fired rules.
  2. Rank and deduplicate insights for display: risk first,
     then growth, then opportunity  within each tag sorted by priority.

Confidence formula (explainable in a viva):
  base        = 50  (we always have at least some data)
  + module bonus: +10 per connected session signal (forecast, segments, anomalies)
  + rule bonus:   sum of weights of fired rules  15, capped at 30
  - penalty:      10 if any high-priority risk rules fired (uncertainty goes up)
  clamped to [20, 95]
"""

from .rules import RuleResult


#  Confidence scorer 

def compute_confidence(
    fired_rules: list[RuleResult],
    has_forecast: bool,
    has_segments: bool,
    has_anomalies: bool,
) -> int:
    """
    Returns an integer confidence score in [20, 95].

    Higher score = more data signals available + rules agree on direction.
    Lower score  = missing module data or high-priority risks dominate.
    """
    score = 50  # base

    # Data completeness bonus
    if has_forecast:  score += 10
    if has_segments:  score += 10
    if has_anomalies: score += 5

    # Rule weight bonus (how many strong rules fired)
    total_weight = sum(r.weight for r in fired_rules)
    score += min(30, int(total_weight * 15))

    # Penalty: if priority-1 risk rules exist, confidence drops
    # (the system is surfacing genuine uncertainty)
    p1_risks = [r for r in fired_rules if r.tag == "risk" and r.priority == 1]
    score -= len(p1_risks) * 10

    return max(20, min(95, score))


#  Insight ranker 

_TAG_ORDER = {"risk": 0, "growth": 1, "opportunity": 2, "neutral": 3}


def rank_insights(fired_rules: list[RuleResult]) -> list[RuleResult]:
    """
    Sort by: tag priority (risk first)  rule priority (1 = most urgent).
    Deduplicate: if two rules fire in the same tag with the same priority,
    keep only the one with higher weight.
    """
    # Sort: tag order first, then rule priority ascending
    sorted_rules = sorted(
        fired_rules,
        key=lambda r: (_TAG_ORDER.get(r.tag, 9), r.priority, -r.weight)
    )

    # Deduplicate by (tag, priority)  keep highest weight
    seen = {}
    deduped = []
    for r in sorted_rules:
        key = (r.tag, r.priority)
        if key not in seen:
            seen[key] = True
            deduped.append(r)

    return deduped


#  Recommendation generator 

def generate_recommendations(ranked: list[RuleResult]) -> list[dict]:
    """
    Convert ranked RuleResults into structured recommendation dicts
    suitable for Streamlit display.
    Each dict has: tag, priority, insight, action, evidence, badge_label.
    """
    badge_map = {
        1: "CRITICAL",
        2: "HIGH",
        3: "MEDIUM",
        4: "LOW",
        5: "INFO",
    }
    recs = []
    for r in ranked:
        recs.append({
            "tag":         r.tag,
            "priority":    r.priority,
            "badge":       badge_map.get(r.priority, "INFO"),
            "insight":     r.insight,
            "action":      r.action,
            "evidence":    r.evidence,
        })
    return recs