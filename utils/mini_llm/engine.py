"""
engine.py
---------
The MiniLLM class  orchestrates all four layers:

  1. context_builder   build_context()
  2. rules             run_all_rules()
  3. scorer            compute_confidence() + rank_insights()
  4. scorer            generate_recommendations()

Usage in any Streamlit page:

    from utils.mini_llm import MiniLLM

    llm = MiniLLM()
    result = llm.run(
        df        = df,
        forecast  = st.session_state.get("forecast"),
        segments  = st.session_state.get("segments"),
        anomalies = st.session_state.get("anomalies", 0),
    )

    result["confidence"]       # int, 0100
    result["recommendations"]  # list of dicts
    result["context"]          # raw signal dict (for display)
    result["risk_score"]       # int, 0100 (separate from confidence)
    result["summary"]          # one-sentence executive summary string
"""

import pandas as pd
from .context_builder import build_context
from .rules           import run_all_rules
from .scorer          import compute_confidence, rank_insights, generate_recommendations


class MiniLLM:
    """
    Explainable, rule-based Mini-LLM for business decision intelligence.
    No external API calls. Every insight traces to a real data column.
    """

    def run(
        self,
        df:        pd.DataFrame,
        forecast:  float | None = None,
        segments:  dict  | None = None,
        anomalies: int          = 0,
        forecast_details: dict | None = None,
        segment_details: dict | None = None,
        anomaly_details: dict | None = None,
    ) -> dict:
        """
        Run the full inference pipeline.

        Returns
        -------
        dict with keys:
          context         : raw signal dict from context_builder
          fired_rules     : list[RuleResult]  all rules that triggered
          ranked          : list[RuleResult]  sorted and deduplicated
          recommendations : list[dict]        formatted for Streamlit
          confidence      : int (0100)
          risk_score      : int (0100)
          summary         : str
        """

        #  Layer 1: Feature extraction 
        ctx = build_context(
            df,
            forecast,
            segments,
            anomalies,
            forecast_details=forecast_details,
            segment_details=segment_details,
            anomaly_details=anomaly_details,
        )

        #  Layer 2: Rule engine 
        fired = run_all_rules(ctx)

        #  Layer 3: Rank + Score 
        ranked     = rank_insights(fired)
        confidence = compute_confidence(
            fired,
            has_forecast  = forecast  is not None,
            has_segments  = segments  is not None,
            has_anomalies = anomalies > 0,
        )

        #  Layer 4: Format output 
        recs       = generate_recommendations(ranked)
        risk_score = self._compute_risk_score(ctx, fired)
        summary    = self._generate_summary(ctx, ranked, confidence)

        return {
            "context":         ctx,
            "fired_rules":     fired,
            "ranked":          ranked,
            "recommendations": recs,
            "confidence":      confidence,
            "risk_score":      risk_score,
            "summary":         summary,
        }

    #  Private helpers 

    @staticmethod
    def _compute_risk_score(ctx: dict, fired: list) -> int:
        """
        Composite risk score 0100.
        Weights four dimensions equally (25 pts each):
          - Operational risk  (cancel_rate, anomaly_count)
          - Revenue risk      (cv, revenue_trend_pct)
          - Quality risk      (avg_review, low_review_pct)
          - Delivery risk     (late_pct, avg_delay)
        """
        # Operational (025)
        op  = min(25, ctx["cancel_rate"] * 3 + ctx["anomaly_count"] * 1.5)
        # Revenue (025)
        rev = min(25, ctx["cv"] * 5 + max(0, -ctx["revenue_trend_pct"]) * 0.5)
        # Quality (025)
        qual = min(25, max(0, (4.0 - ctx["avg_review"]) * 10) + ctx["low_review_pct"] * 0.5)
        # Delivery (025)
        deliv = min(25, ctx["late_pct"] * 0.5 + max(0, ctx["avg_delay"]) * 2)

        return min(100, int(op + rev + qual + deliv))

    @staticmethod
    def _generate_summary(ctx: dict, ranked: list, confidence: int) -> str:
        """
        One-sentence executive summary combining the top insight and confidence.
        """
        if not ranked:
            return (
                f"Business performance is stable with R${ctx['revenue']/1e6:.1f}M revenue "
                f"and {ctx['avg_review']:.1f}/5 avg review (confidence: {confidence}%)."
            )
        top = ranked[0]
        tone = {"risk": "Attention needed", "growth": "Positive signal",
                "opportunity": "Opportunity", "neutral": "Note"}.get(top.tag, "")
        return (
            f"{tone}: {top.insight}. "
            f"Recommended: {top.action} (confidence: {confidence}%)."
        )
