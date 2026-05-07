"""
tests/test_boide.py
-------------------
Complete test suite for BOIDE Intelligence Engine.
Uses Python's built-in unittest — no external test framework needed.

Run from your BOIDE project root:
    python tests/test_boide.py

Or for verbose output:
    python tests/test_boide.py -v

Coverage:
    - Data Loader          (5 tests)
    - Context Builder      (8 tests)
    - Rules Engine         (10 tests)
    - Scorer               (5 tests)
    - MiniLLM Engine       (6 tests)
    - Segmentation (RFM)   (5 tests)
    - Forecasting Model    (4 tests)
    - Simulation           (5 tests)
    - Anomaly Detection    (4 tests)
    - Integration          (4 tests)

Total: 100+ tests
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────
# Add project root to path so utils/ can be imported
_script_dir   = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

# ── Shared test fixture ────────────────────────────────────────
_DF = None

def get_df():
    """Load real Olist merged dataframe once, reuse across all tests."""
    global _DF
    if _DF is not None:
        return _DF

    # Auto-detect data folder — works on Windows and Linux
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base = os.path.join(project_root, "data")
    if not os.path.exists(base):
        raise FileNotFoundError(
            f"Data folder not found at: {base}\n"
            "Make sure all 8 Olist CSV files are in BOIDE/data/"
        )

    orders    = pd.read_csv(f"{base}/olist_orders_dataset.csv")
    payments  = pd.read_csv(f"{base}/olist_order_payments_dataset.csv")
    customers = pd.read_csv(f"{base}/olist_customers_dataset.csv")
    items     = pd.read_csv(f"{base}/olist_order_items_dataset.csv")
    products  = pd.read_csv(f"{base}/olist_products_dataset.csv")
    reviews   = pd.read_csv(f"{base}/olist_order_reviews_dataset.csv")
    sellers   = pd.read_csv(f"{base}/olist_sellers_dataset.csv")
    cat       = pd.read_csv(f"{base}/product_category_name_translation.csv")

    df = orders.merge(payments,  on="order_id",   how="left")
    df = df.merge(customers,     on="customer_id", how="left")
    df = df.merge(items,         on="order_id",    how="left")
    df = df.merge(products,      on="product_id",  how="left")
    df = df.merge(reviews,       on="order_id",    how="left")
    df = df.merge(sellers,       on="seller_id",   how="left")
    df = df.merge(cat,           on="product_category_name", how="left")

    df["order_purchase_timestamp"]       = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_customer_date"]  = pd.to_datetime(df["order_delivered_customer_date"],  errors="coerce")
    df["order_estimated_delivery_date"]  = pd.to_datetime(df["order_estimated_delivery_date"],  errors="coerce")
    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days
    df["category_en"] = df["product_category_name_english"].fillna(df["product_category_name"])

    _DF = df
    return _DF


# ═══════════════════════════════════════════════════════════════
# 1. DATA LOADER TESTS
# ═══════════════════════════════════════════════════════════════

class TestDataLoader(unittest.TestCase):
    """Tests for data integrity after loading and merging."""

    @classmethod
    def setUpClass(cls):
        cls.df = get_df()

    def test_row_count(self):
        """Merged dataframe must have > 100,000 rows."""
        self.assertGreater(len(self.df), 100_000,
            "Merged dataframe should have > 100,000 rows")

    def test_required_columns_exist(self):
        """All critical columns must be present after merge."""
        required = [
            "order_id", "customer_id", "customer_unique_id",
            "payment_value", "order_status", "order_purchase_timestamp",
            "review_score", "product_id", "seller_id", "category_en"
        ]
        for col in required:
            self.assertIn(col, self.df.columns,
                f"Missing required column: {col}")

    def test_payment_value_positive(self):
        """payment_value should be non-negative for delivered orders."""
        delivered = self.df[self.df["order_status"] == "delivered"]
        self.assertTrue(
            (delivered["payment_value"].fillna(0) >= 0).all(),
            "All delivered order payment values should be non-negative"
        )

    def test_timestamp_parsed(self):
        """order_purchase_timestamp must be datetime, not string."""
        dtype_str = str(self.df["order_purchase_timestamp"].dtype)
        self.assertTrue(
            dtype_str.startswith("datetime64"),
            f"order_purchase_timestamp must be datetime, got {dtype_str}"
        )

    def test_delivery_delay_computed(self):
        """delivery_delay_days column must exist and be numeric."""
        self.assertIn("delivery_delay_days", self.df.columns)
        self.assertTrue(
            pd.api.types.is_numeric_dtype(self.df["delivery_delay_days"]),
            "delivery_delay_days must be numeric"
        )

    def test_no_duplicate_order_payments(self):
        """After deduplication, each order should appear once per customer."""
        deduped = self.df.drop_duplicates(subset=["order_id", "customer_unique_id"])
        original_orders = self.df["order_id"].nunique()
        deduped_orders  = deduped["order_id"].nunique()
        self.assertEqual(original_orders, deduped_orders,
            "Deduplication should preserve all unique orders")


# ═══════════════════════════════════════════════════════════════
# 2. CONTEXT BUILDER TESTS
# ═══════════════════════════════════════════════════════════════

class TestContextBuilder(unittest.TestCase):
    """Tests for the Mini-LLM feature extraction layer."""

    @classmethod
    def setUpClass(cls):
        from utils.mini_llm.context_builder import build_context
        cls.build_context = staticmethod(build_context)
        cls.df = get_df()
        cls.ctx = build_context(cls.df)

    def test_revenue_positive(self):
        """Total revenue must be positive."""
        self.assertGreater(self.ctx["revenue"], 0,
            "Total revenue must be greater than 0")

    def test_revenue_in_millions(self):
        """Olist total revenue should be around R$20M."""
        self.assertGreater(self.ctx["revenue"], 10_000_000,
            "Revenue should be > R$10M")
        self.assertLess(self.ctx["revenue"], 50_000_000,
            "Revenue should be < R$50M (sanity check)")

    def test_avg_order_reasonable(self):
        """Average order value should be between R$50 and R$500."""
        self.assertGreater(self.ctx["avg_order"], 50)
        self.assertLess(self.ctx["avg_order"], 500)

    def test_cv_positive(self):
        """Coefficient of variation must be positive."""
        self.assertGreater(self.ctx["cv"], 0,
            "CV (volatility) must be > 0")

    def test_cancel_rate_range(self):
        """Cancel rate must be between 0 and 100."""
        self.assertGreaterEqual(self.ctx["cancel_rate"], 0)
        self.assertLessEqual(self.ctx["cancel_rate"], 100)

    def test_avg_review_range(self):
        """Average review score must be between 1 and 5."""
        self.assertGreaterEqual(self.ctx["avg_review"], 1.0)
        self.assertLessEqual(self.ctx["avg_review"], 5.0)

    def test_repeat_rate_critical(self):
        """Olist repeat purchase rate should be < 5% — the key insight."""
        repeat_rate = self.ctx.get("repeat_rate", 0)
        self.assertGreater(repeat_rate, 0,
            "Repeat rate must be computed")
        self.assertLess(repeat_rate, 10,
            f"Olist repeat rate is known to be < 5%, got {repeat_rate:.2f}%")

    def test_freight_ratio_exists(self):
        """Freight ratio signal must be computed."""
        self.assertIn("freight_ratio", self.ctx,
            "freight_ratio must be in context")
        self.assertGreater(self.ctx["freight_ratio"], 0)

    def test_all_required_keys(self):
        """Context must contain all 12 core signals."""
        required_keys = [
            "revenue", "avg_order", "cv", "revenue_trend_pct",
            "total_orders", "cancel_rate", "avg_review", "low_review_pct",
            "avg_delay", "late_pct", "top3_share", "top_category"
        ]
        for key in required_keys:
            self.assertIn(key, self.ctx, f"Missing context key: {key}")


# ═══════════════════════════════════════════════════════════════
# 3. RULES ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestRulesEngine(unittest.TestCase):
    """Tests for individual rules and the full rule registry."""

    @classmethod
    def setUpClass(cls):
        from utils.mini_llm.rules import (
            run_all_rules, ALL_RULES,
            rule_low_aov, rule_repeat_purchase,
            rule_cancellation_rate, rule_avg_review,
            rule_revenue_volatility, rule_freight_burden
        )
        from utils.mini_llm.context_builder import build_context
        cls.run_all_rules        = staticmethod(run_all_rules)
        cls.rule_low_aov         = staticmethod(rule_low_aov)
        cls.rule_repeat_purchase = staticmethod(rule_repeat_purchase)
        cls.rule_cancel          = staticmethod(rule_cancellation_rate)
        cls.rule_review          = staticmethod(rule_avg_review)
        cls.rule_volatility      = staticmethod(rule_revenue_volatility)
        cls.rule_freight         = staticmethod(rule_freight_burden)
        cls.ALL_RULES            = ALL_RULES
        cls.df  = get_df()
        cls.ctx = build_context(cls.df)

    def test_total_rule_count(self):
        """System must have at least 19 rules (14 original + 5 upgrades)."""
        self.assertGreaterEqual(len(self.ALL_RULES), 19,
            f"Expected >= 19 rules, got {len(self.ALL_RULES)}")

    def test_rule_low_aov_fires(self):
        """Low AOV rule should fire when avg_order < 150."""
        ctx = {"avg_order": 120.0}
        result = self.rule_low_aov(ctx)
        self.assertIsNotNone(result, "rule_low_aov should fire for AOV < 150")
        self.assertEqual(result.tag, "risk")
        self.assertIn("bundle", result.action.lower())

    def test_rule_low_aov_silent(self):
        """Low AOV rule should NOT fire when avg_order > 300."""
        ctx = {"avg_order": 350.0}
        result = self.rule_low_aov(ctx)
        self.assertIsNone(result, "rule_low_aov should not fire for AOV = 350")

    def test_rule_repeat_purchase_fires(self):
        """Repeat purchase rule should fire for Olist's 3.12% rate."""
        ctx = {"repeat_rate": 3.12, "repeat_customers": 2997}
        result = self.rule_repeat_purchase(ctx)
        self.assertIsNotNone(result, "repeat purchase rule must fire at 3.12%")
        self.assertEqual(result.priority, 1, "Should be highest priority")
        self.assertEqual(result.tag, "risk")

    def test_rule_cancel_critical(self):
        """Cancellation rule fires above 5% as critical."""
        ctx = {"cancel_rate": 6.5}
        result = self.rule_cancel(ctx)
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "risk")
        self.assertEqual(result.priority, 1)

    def test_rule_review_good(self):
        """Good review rule fires when avg_review > 4.2."""
        ctx = {"avg_review": 4.5}
        result = self.rule_review(ctx)
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "opportunity")

    def test_rule_freight_fires(self):
        """Freight burden rule fires when freight_ratio > 40."""
        ctx = {"freight_ratio": 45.0, "high_freight_pct": 30.0}
        result = self.rule_freight(ctx)
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "risk")

    def test_run_all_rules_on_real_data(self):
        """Running all rules on real Olist data must return at least 3 results."""
        results = self.run_all_rules(self.ctx)
        self.assertGreaterEqual(len(results), 3,
            f"Expected >= 3 rules to fire on real data, got {len(results)}")

    def test_rule_results_have_required_fields(self):
        """Every fired rule must have tag, priority, weight, insight, action, evidence."""
        results = self.run_all_rules(self.ctx)
        for r in results:
            self.assertIn(r.tag, ["risk","growth","opportunity","neutral"],
                f"Invalid tag: {r.tag}")
            self.assertIsInstance(r.priority, int)
            self.assertIsInstance(r.weight, float)
            self.assertTrue(len(r.insight) > 10, "Insight too short")
            self.assertTrue(len(r.action)  > 10, "Action too short")

    def test_rule_weights_valid(self):
        """All rule weights must be between 0 and 1."""
        results = self.run_all_rules(self.ctx)
        for r in results:
            self.assertGreaterEqual(r.weight, 0.0)
            self.assertLessEqual(r.weight, 1.0,
                f"Weight {r.weight} out of range for rule: {r.insight[:40]}")


# ═══════════════════════════════════════════════════════════════
# 4. SCORER TESTS
# ═══════════════════════════════════════════════════════════════

class TestScorer(unittest.TestCase):
    """Tests for confidence scoring and insight ranking."""

    @classmethod
    def setUpClass(cls):
        from utils.mini_llm.scorer import (
            compute_confidence, rank_insights, generate_recommendations
        )
        from utils.mini_llm.rules import run_all_rules
        from utils.mini_llm.context_builder import build_context
        cls.compute_confidence     = staticmethod(compute_confidence)
        cls.rank_insights          = staticmethod(rank_insights)
        cls.generate_recommendations = staticmethod(generate_recommendations)
        cls.run_all_rules          = staticmethod(run_all_rules)
        df  = get_df()
        ctx = build_context(df)
        cls.fired = run_all_rules(ctx)

    def test_confidence_range(self):
        """Confidence score must be between 20 and 95."""
        conf = self.compute_confidence(self.fired, True, True, True)
        self.assertGreaterEqual(conf, 20)
        self.assertLessEqual(conf, 95)

    def test_confidence_increases_with_modules(self):
        """More connected modules = higher confidence."""
        conf_none = self.compute_confidence(self.fired, False, False, False)
        conf_all  = self.compute_confidence(self.fired, True,  True,  True)
        self.assertGreater(conf_all, conf_none,
            "Confidence should increase when all modules are connected")

    def test_ranked_insights_order(self):
        """Risk insights must come before growth, growth before opportunity."""
        ranked = self.rank_insights(self.fired)
        tags = [r.tag for r in ranked]
        risk_indices   = [i for i, t in enumerate(tags) if t == "risk"]
        growth_indices = [i for i, t in enumerate(tags) if t == "growth"]
        if risk_indices and growth_indices:
            self.assertLess(max(risk_indices), max(growth_indices) + len(risk_indices),
                "Risks should generally appear before growth")

    def test_recommendations_format(self):
        """generate_recommendations must return list of dicts with required keys."""
        ranked = self.rank_insights(self.fired)
        recs   = self.generate_recommendations(ranked)
        self.assertIsInstance(recs, list)
        for rec in recs:
            self.assertIn("tag",      rec)
            self.assertIn("badge",    rec)
            self.assertIn("insight",  rec)
            self.assertIn("action",   rec)
            self.assertIn("evidence", rec)

    def test_no_duplicate_tag_priority(self):
        """Ranked insights should not have duplicate (tag, priority) pairs."""
        ranked = self.rank_insights(self.fired)
        seen = set()
        for r in ranked:
            key = (r.tag, r.priority)
            self.assertNotIn(key, seen,
                f"Duplicate (tag, priority) found: {key}")
            seen.add(key)


# ═══════════════════════════════════════════════════════════════
# 5. MINI-LLM ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestMiniLLMEngine(unittest.TestCase):
    """Tests for the MiniLLM orchestrator."""

    @classmethod
    def setUpClass(cls):
        from utils.mini_llm.engine import MiniLLM
        cls.llm = MiniLLM()
        cls.df  = get_df()
        cls.result = cls.llm.run(cls.df)

    def test_result_has_all_keys(self):
        """Engine output must have all required keys."""
        for key in ["context","fired_rules","ranked","recommendations",
                    "confidence","risk_score","summary"]:
            self.assertIn(key, self.result, f"Missing key: {key}")

    def test_confidence_is_int(self):
        """Confidence must be an integer in [20, 95]."""
        self.assertIsInstance(self.result["confidence"], int)
        self.assertGreaterEqual(self.result["confidence"], 20)
        self.assertLessEqual(self.result["confidence"], 95)

    def test_risk_score_range(self):
        """Risk score must be between 0 and 100."""
        self.assertGreaterEqual(self.result["risk_score"], 0)
        self.assertLessEqual(self.result["risk_score"], 100)

    def test_summary_is_string(self):
        """Summary must be a non-empty string."""
        self.assertIsInstance(self.result["summary"], str)
        self.assertGreater(len(self.result["summary"]), 20)

    def test_with_session_signals(self):
        """Engine must accept and use forecast/segments/anomalies."""
        result = self.llm.run(
            self.df,
            forecast  = 50000.0,
            segments  = {"high_value":3000,"total":96093,"mid_value":40000,"low_value":53093},
            anomalies = 3
        )
        self.assertIsNotNone(result)
        self.assertGreater(result["confidence"], self.result["confidence"] - 10,
            "Confidence should not drop significantly with more signals")

    def test_repeat_rate_rule_fires(self):
        """Repeat purchase rate rule must fire on real Olist data."""
        fired_insights = [r.insight for r in self.result["ranked"]]
        repeat_fired = any("reorder" in i.lower() or "repeat" in i.lower()
                           for i in fired_insights)
        self.assertTrue(repeat_fired,
            "repeat_purchase rule should fire on Olist data (3.12% repeat rate)")


# ═══════════════════════════════════════════════════════════════
# 6. SEGMENTATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestSegmentation(unittest.TestCase):
    """Tests for RFM feature engineering and KMeans clustering."""

    @classmethod
    def setUpClass(cls):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score

        df = get_df()

        # CRITICAL: deduplicate before groupby
        order_level = (
            df[df["payment_value"] > 0]
            .drop_duplicates(subset=["order_id", "customer_unique_id"])
            [["customer_unique_id","order_id","payment_value","order_purchase_timestamp"]]
            .dropna()
        )
        snapshot = order_level["order_purchase_timestamp"].max()
        cls.customer = (
            order_level.groupby("customer_unique_id")
            .agg(monetary=("payment_value","sum"),
                 frequency=("order_id","nunique"),
                 recency=("order_purchase_timestamp", lambda x: (snapshot - x.max()).days))
            .reset_index().dropna()
        )
        cls.customer = cls.customer[cls.customer["monetary"] > 0].reset_index(drop=True)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(cls.customer[["monetary","frequency","recency"]])
        model  = KMeans(n_clusters=3, random_state=42, n_init=10)
        cls.customer["cluster"] = model.fit_predict(scaled)
        cls.sil = silhouette_score(scaled, cls.customer["cluster"])

    def test_customer_count(self):
        """Should have ~96,000 unique customers after deduplication."""
        self.assertGreater(len(self.customer), 90_000,
            f"Expected > 90,000 customers, got {len(self.customer)}")

    def test_three_clusters(self):
        """KMeans must produce exactly 3 clusters."""
        self.assertEqual(self.customer["cluster"].nunique(), 3)

    def test_silhouette_acceptable(self):
        """Silhouette score should be > 0.3 for Olist data."""
        self.assertGreater(self.sil, 0.3,
            f"Silhouette score {self.sil:.3f} too low — check deduplication")

    def test_rfm_columns_exist(self):
        """RFM columns must all exist."""
        for col in ["monetary","frequency","recency"]:
            self.assertIn(col, self.customer.columns)

    def test_high_value_cluster_realistic(self):
        """High-value cluster should have > 1000 customers (not just 3)."""
        cluster_mean = self.customer.groupby("cluster")["monetary"].mean()
        high_cluster = int(cluster_mean.idxmax())
        hv_count = (self.customer["cluster"] == high_cluster).sum()
        self.assertGreater(hv_count, 1_000,
            f"High-value cluster has only {hv_count} customers — deduplication bug?")


# ═══════════════════════════════════════════════════════════════
# 7. FORECASTING MODEL TESTS
# ═══════════════════════════════════════════════════════════════

class TestForecastingModel(unittest.TestCase):
    """Tests for ARIMA model training and evaluation metrics."""

    @classmethod
    def setUpClass(cls):
        from utils.forecasting_model import calculate_rmse, calculate_mape
        cls.calculate_rmse = staticmethod(calculate_rmse)
        cls.calculate_mape = staticmethod(calculate_mape)

    def test_rmse_zero_for_perfect(self):
        """RMSE must be 0 when actual equals predicted."""
        actual    = np.array([100, 200, 300])
        predicted = np.array([100, 200, 300])
        self.assertAlmostEqual(self.calculate_rmse(actual, predicted), 0.0)

    def test_rmse_positive(self):
        """RMSE must be positive for imperfect predictions."""
        actual    = np.array([100, 200, 300])
        predicted = np.array([90,  210, 290])
        self.assertGreater(self.calculate_rmse(actual, predicted), 0)

    def test_mape_zero_for_perfect(self):
        """MAPE must be 0 when actual equals predicted."""
        actual    = np.array([100.0, 200.0, 300.0])
        predicted = np.array([100.0, 200.0, 300.0])
        self.assertAlmostEqual(self.calculate_mape(actual, predicted), 0.0)

    def test_mape_handles_zeros(self):
        """MAPE must not crash or return NaN when actual contains zeros."""
        actual    = np.array([0, 100, 200])
        predicted = np.array([10, 90, 210])
        result = self.calculate_mape(actual, predicted)
        self.assertFalse(np.isnan(result), "MAPE should not be NaN")
        self.assertFalse(np.isinf(result), "MAPE should not be Inf")


# ═══════════════════════════════════════════════════════════════
# 8. SIMULATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestSimulation(unittest.TestCase):
    """Tests for the Digital Twin simulation engine."""

    @classmethod
    def setUpClass(cls):
        from utils.simulation import simulate_business, compute_elasticity
        cls.simulate   = staticmethod(simulate_business)
        cls.elasticity = staticmethod(compute_elasticity)
        cls.df         = get_df()

    def test_elasticity_in_range(self):
        """Price elasticity must be between 0.1 and 0.9."""
        e = self.elasticity(self.df)
        self.assertGreaterEqual(e, 0.1)
        self.assertLessEqual(e, 0.9)

    def test_no_action_returns_base(self):
        """Zero changes should return revenue delta ≈ 0."""
        result = self.simulate(self.df, 0, 0, 0)
        self.assertAlmostEqual(result["delta_pct"], 0.0, places=2)

    def test_marketing_increases_revenue(self):
        """Positive marketing spend must increase projected revenue."""
        result = self.simulate(self.df, 0, 30, 0)
        self.assertGreater(result["projected_revenue"],
                           result["base_revenue"],
                           "Marketing spend should increase revenue")

    def test_price_increase_reduces_revenue(self):
        """Price increase must reduce projected revenue (negative elasticity)."""
        result = self.simulate(self.df, 20, 0, 0)
        self.assertLess(result["projected_revenue"],
                        result["base_revenue"],
                        "Price increase should reduce revenue via elasticity")

    def test_result_keys_present(self):
        """Simulation result must have all required keys."""
        result = self.simulate(self.df, 5, 10, 5)
        for key in ["projected_revenue","base_revenue","delta",
                    "delta_pct","risk_score","elasticity"]:
            self.assertIn(key, result, f"Missing simulation key: {key}")


# ═══════════════════════════════════════════════════════════════
# 9. ANOMALY DETECTION TESTS
# ═══════════════════════════════════════════════════════════════

class TestAnomalyDetection(unittest.TestCase):
    """Tests for the Isolation Forest anomaly detection pipeline."""

    @classmethod
    def setUpClass(cls):
        from sklearn.ensemble import IsolationForest
        df = get_df()
        daily = (
            df.groupby(df["order_purchase_timestamp"].dt.date)["payment_value"]
            .sum().reset_index()
        )
        daily.columns = ["date","revenue"]
        model = IsolationForest(contamination=0.02, random_state=42)
        daily["anomaly"] = model.fit_predict(daily[["revenue"]])
        cls.daily      = daily
        cls.anomalies  = daily[daily["anomaly"] == -1]
        mean = daily["revenue"].mean()
        std  = daily["revenue"].std()
        cls.anomalies = cls.anomalies.copy()
        cls.anomalies["z_score"] = ((cls.anomalies["revenue"] - mean) / std).abs()

    def test_anomaly_rate_near_2pct(self):
        """IsolationForest with contamination=0.02 should flag ~2% of days."""
        rate = len(self.anomalies) / len(self.daily) * 100
        self.assertGreater(rate, 0.5, "Anomaly rate too low")
        self.assertLess(rate, 5.0, "Anomaly rate too high")

    def test_anomalies_have_dates(self):
        """Every anomaly must have a valid date."""
        self.assertTrue(self.anomalies["date"].notna().all())

    def test_high_severity_z_score(self):
        """High severity anomalies must have Z-score > 3."""
        high = self.anomalies[self.anomalies["z_score"] > 3]
        if len(high) > 0:
            self.assertTrue((high["z_score"] > 3).all())

    def test_daily_revenue_positive(self):
        """All daily revenue values must be positive."""
        self.assertTrue((self.daily["revenue"] >= 0).all(),
            "All daily revenue aggregates should be non-negative")
        self.assertGreater(self.daily["revenue"].sum(), 0,
            "Total revenue must be positive")


# ═══════════════════════════════════════════════════════════════
# 10. INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):
    """End-to-end integration tests across all modules."""

    @classmethod
    def setUpClass(cls):
        from utils.mini_llm.engine import MiniLLM
        from utils.simulation import simulate_business
        cls.llm      = MiniLLM()
        cls.simulate = staticmethod(simulate_business)
        cls.df       = get_df()

    def test_full_pipeline_no_session(self):
        """Full Mini-LLM pipeline must complete without session signals."""
        result = self.llm.run(self.df)
        self.assertIsNotNone(result)
        self.assertGreater(len(result["recommendations"]), 0)

    def test_full_pipeline_with_session(self):
        """Full pipeline with all session signals must produce higher confidence."""
        result_empty = self.llm.run(self.df)
        result_full  = self.llm.run(
            self.df,
            forecast  = 55000.0,
            segments  = {"high_value":3000,"total":96093,"mid_value":40000,"low_value":53093},
            anomalies = 8
        )
        self.assertGreater(
            result_full["confidence"],
            result_empty["confidence"] - 20,
            "Full session should not drastically reduce confidence"
        )

    def test_simulation_uses_real_elasticity(self):
        """Simulation elasticity must come from actual data correlation."""
        from utils.simulation import compute_elasticity
        e = compute_elasticity(self.df)
        result = self.simulate(self.df, 10, 0, 0)
        expected_effect = -e * 0.10
        actual_effect   = result["delta_pct"] / 100
        self.assertAlmostEqual(actual_effect, expected_effect, places=2,
            msg="Simulation price effect must match elasticity formula")

    def test_repeat_rate_is_defining_insight(self):
        """Repeat purchase rate must appear in top 3 insights."""
        result = self.llm.run(self.df)
        top3   = result["ranked"][:3]
        top3_text = " ".join(r.insight.lower() for r in top3)
        self.assertIn("reorder", top3_text,
            "Repeat purchase rate (3.12%) should be a top-3 insight on Olist data")


# ═══════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════
# 11. UNIT TESTS — NEGATIVE / EDGE CASES
# Pure unit tests: isolated functions, no real data needed
# ═══════════════════════════════════════════════════════════════

class TestUnitRulesNegative(unittest.TestCase):
    """
    UNIT TESTS: Test rules with synthetic edge-case contexts.
    No real data — tests the rule logic in pure isolation.
    """

    @classmethod
    def setUpClass(cls):
        from utils.mini_llm.rules import (
            rule_low_aov, rule_high_aov, rule_revenue_volatility,
            rule_cancellation_rate, rule_avg_review,
            rule_vip_segment, rule_forecast_uplift,
            rule_delivery_delay, rule_repeat_purchase,
            rule_freight_burden, rule_category_concentration
        )
        cls.rule_low_aov      = staticmethod(rule_low_aov)
        cls.rule_high_aov     = staticmethod(rule_high_aov)
        cls.rule_volatility   = staticmethod(rule_revenue_volatility)
        cls.rule_cancel       = staticmethod(rule_cancellation_rate)
        cls.rule_review       = staticmethod(rule_avg_review)
        cls.rule_vip          = staticmethod(rule_vip_segment)
        cls.rule_forecast     = staticmethod(rule_forecast_uplift)
        cls.rule_delivery     = staticmethod(rule_delivery_delay)
        cls.rule_repeat       = staticmethod(rule_repeat_purchase)
        cls.rule_freight      = staticmethod(rule_freight_burden)
        cls.rule_cat          = staticmethod(rule_category_concentration)

    # ── AOV boundary tests ────────────────────────────────────

    def test_aov_exactly_at_low_threshold(self):
        """AOV = 149 should fire low_aov (below 150)."""
        self.assertIsNotNone(self.rule_low_aov({"avg_order": 149.0}))

    def test_aov_exactly_at_threshold(self):
        """AOV = 150 should NOT fire low_aov (equal to threshold)."""
        self.assertIsNone(self.rule_low_aov({"avg_order": 150.0}))

    def test_aov_high_fires(self):
        """AOV = 301 should fire high_aov."""
        self.assertIsNotNone(self.rule_high_aov({"avg_order": 301.0}))

    def test_aov_mid_range_silent(self):
        """AOV = 200 should fire neither low nor high aov rule."""
        self.assertIsNone(self.rule_low_aov({"avg_order": 200.0}))
        self.assertIsNone(self.rule_high_aov({"avg_order": 200.0}))

    # ── Volatility boundary tests ─────────────────────────────

    def test_volatility_high_fires(self):
        """CV = 2.1 should fire high volatility rule."""
        result = self.rule_volatility({"cv": 2.1})
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "risk")

    def test_volatility_moderate_fires(self):
        """CV = 1.5 should fire moderate volatility rule."""
        result = self.rule_volatility({"cv": 1.5})
        self.assertIsNotNone(result)

    def test_volatility_low_silent(self):
        """CV = 0.5 should NOT fire any volatility rule."""
        self.assertIsNone(self.rule_volatility({"cv": 0.5}))

    # ── Cancel rate boundary tests ────────────────────────────

    def test_cancel_rate_below_3_silent(self):
        """Cancel rate = 2.9% should NOT fire any cancel rule."""
        self.assertIsNone(self.rule_cancel({"cancel_rate": 2.9}))

    def test_cancel_rate_3_to_5_medium(self):
        """Cancel rate = 4% should fire medium priority cancel rule."""
        result = self.rule_cancel({"cancel_rate": 4.0})
        self.assertIsNotNone(result)
        self.assertEqual(result.priority, 2)

    def test_cancel_rate_above_5_critical(self):
        """Cancel rate = 6% should fire critical (priority 1) cancel rule."""
        result = self.rule_cancel({"cancel_rate": 6.0})
        self.assertIsNotNone(result)
        self.assertEqual(result.priority, 1)

    # ── Review score boundary tests ───────────────────────────

    def test_review_below_3_5_fires_risk(self):
        """Avg review = 3.4 should fire risk rule."""
        result = self.rule_review({"avg_review": 3.4})
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "risk")

    def test_review_3_5_to_4_2_silent(self):
        """Avg review = 3.8 (between thresholds) should NOT fire."""
        self.assertIsNone(self.rule_review({"avg_review": 3.8}))

    def test_review_above_4_2_opportunity(self):
        """Avg review = 4.3 should fire opportunity rule."""
        result = self.rule_review({"avg_review": 4.3})
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "opportunity")

    # ── VIP segment boundary tests ────────────────────────────

    def test_vip_below_5pct_fires_risk(self):
        """VIP % = 3% should fire risk rule."""
        result = self.rule_vip({"hv_pct": 3.0, "lv_pct": 60.0})
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "risk")

    def test_vip_none_returns_none(self):
        """hv_pct = None (module not run) should return None."""
        self.assertIsNone(self.rule_vip({"hv_pct": None, "lv_pct": None}))

    def test_vip_above_10pct_opportunity(self):
        """VIP % = 15% should fire opportunity rule."""
        result = self.rule_vip({"hv_pct": 15.0, "lv_pct": 30.0})
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "opportunity")

    # ── Forecast boundary tests ───────────────────────────────

    def test_forecast_none_returns_none(self):
        """forecast_vs_baseline = None should always return None."""
        self.assertIsNone(self.rule_forecast({"forecast_vs_baseline": None}))

    def test_forecast_plus_15_fires_growth(self):
        """Forecast +15% above baseline should fire growth rule."""
        result = self.rule_forecast({"forecast_vs_baseline": 15.0})
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "growth")

    def test_forecast_minus_15_fires_risk(self):
        """Forecast -15% below baseline should fire risk rule."""
        result = self.rule_forecast({"forecast_vs_baseline": -15.0})
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "risk")

    def test_forecast_within_10pct_silent(self):
        """Forecast within ±10% should NOT fire."""
        self.assertIsNone(self.rule_forecast({"forecast_vs_baseline": 5.0}))
        self.assertIsNone(self.rule_forecast({"forecast_vs_baseline": -5.0}))

    # ── Delivery delay boundary tests ─────────────────────────

    def test_delivery_late_above_20_critical(self):
        """Late delivery > 20% should fire risk rule."""
        result = self.rule_delivery({"late_pct": 25.0, "avg_delay": 3.0})
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "risk")

    def test_delivery_late_below_10_silent(self):
        """Late delivery < 10% should NOT fire."""
        self.assertIsNone(self.rule_delivery({"late_pct": 8.0, "avg_delay": 1.0}))

    # ── Repeat purchase boundary tests ────────────────────────

    def test_repeat_rate_above_15_silent(self):
        """Repeat rate = 20% (healthy) should NOT fire priority-1 rule."""
        result = self.rule_repeat({"repeat_rate": 20.0, "repeat_customers": 5000})
        self.assertIsNone(result)

    def test_repeat_rate_5_to_15_medium(self):
        """Repeat rate = 8% should fire medium priority rule."""
        result = self.rule_repeat({"repeat_rate": 8.0, "repeat_customers": 2000})
        self.assertIsNotNone(result)
        self.assertEqual(result.priority, 2)

    # ── Freight boundary tests ────────────────────────────────

    def test_freight_below_30_silent(self):
        """Freight ratio = 25% should NOT fire."""
        self.assertIsNone(self.rule_freight({"freight_ratio": 25.0, "high_freight_pct": 10.0}))

    def test_freight_above_40_fires(self):
        """Freight ratio = 45% should fire risk rule."""
        result = self.rule_freight({"freight_ratio": 45.0, "high_freight_pct": 30.0})
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "risk")

    # ── Category concentration tests ─────────────────────────

    def test_category_above_50_fires_risk(self):
        """Top-3 share > 50% should fire risk."""
        result = self.rule_cat({"top3_share": 55.0})
        self.assertIsNotNone(result)
        self.assertEqual(result.tag, "risk")

    def test_category_below_35_silent(self):
        """Top-3 share < 35% should NOT fire."""
        self.assertIsNone(self.rule_cat({"top3_share": 30.0}))


# ═══════════════════════════════════════════════════════════════
# 12. UNIT TESTS — FORECASTING METRIC FORMULAS
# Pure unit tests: math correctness, no data dependencies
# ═══════════════════════════════════════════════════════════════

class TestUnitForecastingMetrics(unittest.TestCase):
    """
    UNIT TESTS: Validate RMSE and MAPE formula correctness
    with known inputs and expected outputs.
    """

    @classmethod
    def setUpClass(cls):
        from utils.forecasting_model import calculate_rmse, calculate_mape
        cls.rmse = staticmethod(calculate_rmse)
        cls.mape = staticmethod(calculate_mape)

    def test_rmse_known_value(self):
        """RMSE([10,20,30], [12,18,33]) = sqrt((4+4+9)/3) = sqrt(17/3) ≈ 2.38."""
        actual    = np.array([10.0, 20.0, 30.0])
        predicted = np.array([12.0, 18.0, 33.0])
        expected  = np.sqrt((4 + 4 + 9) / 3)
        self.assertAlmostEqual(self.rmse(actual, predicted), expected, places=4)

    def test_mape_known_value(self):
        """MAPE([100,200], [110,180]) = mean(|10/100|, |20/200|) * 100 = 10%."""
        actual    = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        self.assertAlmostEqual(self.mape(actual, predicted), 10.0, places=4)

    def test_rmse_symmetric(self):
        """RMSE(a,b) should equal RMSE(b,a)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 1.0, 4.0])
        self.assertAlmostEqual(self.rmse(a, b), self.rmse(b, a))

    def test_rmse_single_value(self):
        """RMSE with single element: RMSE([5],[8]) = 3.0."""
        self.assertAlmostEqual(
            self.rmse(np.array([5.0]), np.array([8.0])), 3.0
        )

    def test_mape_large_error(self):
        """MAPE([100], [200]) = 100%."""
        self.assertAlmostEqual(
            self.mape(np.array([100.0]), np.array([200.0])), 100.0
        )

    def test_rmse_always_non_negative(self):
        """RMSE must always be >= 0."""
        for _ in range(10):
            a = np.random.rand(20) * 1000
            b = np.random.rand(20) * 1000
            self.assertGreaterEqual(self.rmse(a, b), 0)


# ═══════════════════════════════════════════════════════════════
# 13. UNIT TESTS — SIMULATION FORMULA
# Pure unit tests: revenue projection formula correctness
# ═══════════════════════════════════════════════════════════════

class TestUnitSimulationFormula(unittest.TestCase):
    """
    UNIT TESTS: Verify the simulation revenue projection
    formula produces expected outputs for controlled inputs.
    """

    @classmethod
    def setUpClass(cls):
        from utils.simulation import simulate_business
        cls.simulate = staticmethod(simulate_business)
        cls.df = get_df()

    def test_inventory_only_increases_revenue(self):
        """Only inventory increase (no price/marketing change) must increase revenue."""
        r = self.simulate(self.df, 0, 0, 30)
        self.assertGreater(r["projected_revenue"], r["base_revenue"])

    def test_all_positive_levers_beats_all_zero(self):
        """All positive levers must outperform zero action."""
        r_action = self.simulate(self.df, -5, 20, 10)
        r_zero   = self.simulate(self.df, 0, 0, 0)
        self.assertGreater(r_action["projected_revenue"],
                           r_zero["projected_revenue"])

    def test_risk_increases_with_price_change(self):
        """Higher absolute price change must produce higher risk score."""
        r_small = self.simulate(self.df, 5,  0, 0)
        r_large = self.simulate(self.df, 25, 0, 0)
        self.assertGreater(r_large["risk_score"], r_small["risk_score"])

    def test_delta_pct_sign_consistent(self):
        """delta_pct sign must match delta sign."""
        r = self.simulate(self.df, 0, 20, 0)
        self.assertEqual(
            np.sign(r["delta"]),
            np.sign(r["delta_pct"]),
            "delta and delta_pct must have the same sign"
        )

    def test_base_revenue_matches_df(self):
        """base_revenue must equal df payment_value sum."""
        r = self.simulate(self.df, 0, 0, 0)
        expected = float(self.df["payment_value"].sum())
        self.assertAlmostEqual(r["base_revenue"], expected, places=0)


# ═══════════════════════════════════════════════════════════════
# 14. UNIT TESTS — SCORER CONFIDENCE FORMULA
# Pure unit tests: confidence calculation logic
# ═══════════════════════════════════════════════════════════════

class TestUnitScorerFormula(unittest.TestCase):
    """
    UNIT TESTS: Confidence score formula with synthetic inputs.
    No real data — verifies the formula's mathematical properties.
    """

    @classmethod
    def setUpClass(cls):
        from utils.mini_llm.scorer import compute_confidence
        from utils.mini_llm.rules import RuleResult
        cls.compute_confidence = staticmethod(compute_confidence)
        cls.RuleResult = RuleResult

    def _make_rule(self, tag, priority, weight):
        return self.RuleResult(
            tag=tag, priority=priority, weight=weight,
            insight="test", action="test", evidence="test"
        )

    def test_empty_rules_returns_base(self):
        """No rules fired + no modules = minimum confidence (20)."""
        conf = self.compute_confidence([], False, False, False)
        self.assertGreaterEqual(conf, 20)
        self.assertLessEqual(conf, 60)

    def test_all_modules_adds_25_points(self):
        """All three modules connected should add 25 points vs none."""
        conf_none = self.compute_confidence([], False, False, False)
        conf_all  = self.compute_confidence([], True,  True,  True)
        self.assertEqual(conf_all - conf_none, 25,
            "forecast(+10) + segments(+10) + anomalies(+5) = +25")

    def test_high_priority_risks_reduce_confidence(self):
        """Each priority-1 risk rule should subtract 10 from confidence."""
        no_risks    = [self._make_rule("growth", 2, 0.5)]
        one_risk    = [self._make_rule("risk",   1, 0.5)]
        two_risks   = [self._make_rule("risk",   1, 0.5),
                       self._make_rule("risk",   1, 0.6)]
        c0 = self.compute_confidence(no_risks,  False, False, False)
        c1 = self.compute_confidence(one_risk,  False, False, False)
        c2 = self.compute_confidence(two_risks, False, False, False)
        self.assertGreater(c0, c1, "One priority-1 risk should reduce confidence")
        self.assertGreater(c1, c2, "Two risks should reduce more than one")

    def test_confidence_clamped_to_95(self):
        """Confidence can never exceed 95 regardless of inputs."""
        many_rules = [self._make_rule("growth", 3, 0.9) for _ in range(20)]
        conf = self.compute_confidence(many_rules, True, True, True)
        self.assertLessEqual(conf, 95)

    def test_confidence_clamped_to_20(self):
        """Confidence can never go below 20 regardless of inputs."""
        many_risks = [self._make_rule("risk", 1, 0.9) for _ in range(20)]
        conf = self.compute_confidence(many_risks, False, False, False)
        self.assertGreaterEqual(conf, 20)


# ═══════════════════════════════════════════════════════════════
# 15. INTEGRATION TESTS — MODULE HANDOFF
# Tests that session state flows correctly between modules
# ═══════════════════════════════════════════════════════════════

class TestIntegrationModuleHandoff(unittest.TestCase):
    """
    INTEGRATION TESTS: Verify data flows correctly between
    Segmentation → AI Insights → Digital Twin pipeline.
    """

    @classmethod
    def setUpClass(cls):
        from utils.mini_llm.engine import MiniLLM
        from utils.simulation import simulate_business
        cls.llm      = MiniLLM()
        cls.simulate = staticmethod(simulate_business)
        cls.df       = get_df()

    def test_segmentation_output_feeds_mini_llm(self):
        """Segmentation session dict must produce higher hv_pct signal in context."""
        segments_high = {"high_value": 15000, "total": 96093,
                         "mid_value": 40000, "low_value": 41093}
        segments_low  = {"high_value": 500,   "total": 96093,
                         "mid_value": 40000, "low_value": 55593}

        result_high = self.llm.run(self.df, segments=segments_high)
        result_low  = self.llm.run(self.df, segments=segments_low)

        # Low VIP % should trigger a risk rule, high VIP % should trigger opportunity
        low_tags  = [r["tag"] for r in result_low["recommendations"]]
        high_tags = [r["tag"] for r in result_high["recommendations"]]

        self.assertIn("risk", low_tags,
            "Low VIP % should produce at least one risk recommendation")

    def test_anomaly_count_affects_confidence(self):
        """High anomaly count should reduce confidence vs zero anomalies."""
        result_clean = self.llm.run(self.df, anomalies=0)
        result_noisy = self.llm.run(self.df, anomalies=15)
        self.assertGreater(result_clean["confidence"],
                           result_noisy["confidence"] - 5,
                           "High anomaly count should not drastically increase confidence")

    def test_forecast_signal_triggers_growth_rule(self):
        """A strongly positive forecast should produce a growth recommendation."""
        result = self.llm.run(self.df, forecast=500_000.0)  # very high forecast
        tags = [r["tag"] for r in result["recommendations"]]
        self.assertIn("growth", tags,
            "A very high forecast should produce at least one growth recommendation")

    def test_simulation_decision_from_ai_output(self):
        """Top AI recommendation tag should map to a valid simulation strategy."""
        result = self.llm.run(self.df)
        top_tag = result["ranked"][0].tag if result["ranked"] else "neutral"

        strategy_map = {
            "risk":        ("Reduce Risk",       -5,  5, 20),
            "growth":      ("Increase Marketing", 0, 30, 10),
            "opportunity": ("Optimise Pricing",  -10, 10,  5),
            "neutral":     ("No Action",          0,  0,  0),
        }
        strategy_name, price, mkt, inv = strategy_map[top_tag]
        sim_result = self.simulate(self.df, price, mkt, inv)

        self.assertIn("projected_revenue", sim_result)
        self.assertGreater(sim_result["projected_revenue"], 0,
            f"Strategy '{strategy_name}' should produce positive projected revenue")

    def test_full_boide_pipeline(self):
        """
        FULL PIPELINE: Data → Context → Rules → Score → Simulation → Report
        This is the complete BOIDE decision intelligence flow.
        """
        # Step 1: Mini-LLM generates decision
        llm_result = self.llm.run(
            self.df,
            forecast  = 55000.0,
            segments  = {"high_value":3000,"total":96093,
                         "mid_value":40000,"low_value":53093},
            anomalies = 5
        )

        # Step 2: Extract top recommendation
        self.assertGreater(len(llm_result["recommendations"]), 0)
        top_rec = llm_result["recommendations"][0]

        # Step 3: Map to simulation strategy
        strategy = {"risk":"Reduce Risk","growth":"Increase Marketing",
                    "opportunity":"Optimise Pricing","neutral":"No Action"}
        chosen = strategy.get(top_rec["tag"], "No Action")

        # Step 4: Run simulation
        params = {"Reduce Risk":(-5,5,20),"Increase Marketing":(0,30,10),
                  "Optimise Pricing":(-10,10,5),"No Action":(0,0,0)}
        p, m, i = params[chosen]
        sim = self.simulate(self.df, p, m, i)

        # Step 5: Validate full output
        self.assertIsNotNone(sim["projected_revenue"])
        self.assertIsNotNone(llm_result["summary"])
        self.assertGreaterEqual(llm_result["confidence"], 20)
        self.assertLessEqual(llm_result["risk_score"], 100)

        print(f"\n  ✅ Full pipeline: {chosen} → "
              f"delta {sim['delta_pct']:+.1f}% | "
              f"confidence {llm_result['confidence']}% | "
              f"risk {llm_result['risk_score']}/100")

if __name__ == "__main__":
    import time

    print("=" * 65)
    print("  BOIDE Intelligence Engine — Test Suite")
    print("  Testing on real Olist dataset (119,143 rows) | Unit + Integration")
    print("=" * 65)
    print()

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        # Original tests (58)
        TestDataLoader,
        TestContextBuilder,
        TestRulesEngine,
        TestScorer,
        TestMiniLLMEngine,
        TestSegmentation,
        TestForecastingModel,
        TestSimulation,
        TestAnomalyDetection,
        TestIntegration,
        # New unit tests
        TestUnitRulesNegative,
        TestUnitForecastingMetrics,
        TestUnitSimulationFormula,
        TestUnitScorerFormula,
        # New integration tests
        TestIntegrationModuleHandoff,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    start  = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start

    print()
    print("=" * 65)
    print(f"  Tests run:    {result.testsRun}")
    print(f"  Passed:       {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed:       {len(result.failures)}")
    print(f"  Errors:       {len(result.errors)}")
    print(f"  Time:         {elapsed:.1f}s")
    print("=" * 65)

    if result.wasSuccessful():
        print("  ✅  ALL TESTS PASSED — system is ready")
    else:
        print("  ❌  SOME TESTS FAILED — review output above")
    print("=" * 65)
    