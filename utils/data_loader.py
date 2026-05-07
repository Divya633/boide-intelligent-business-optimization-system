import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    orders   = pd.read_csv("data/olist_orders_dataset.csv")
    payments = pd.read_csv("data/olist_order_payments_dataset.csv")
    customers= pd.read_csv("data/olist_customers_dataset.csv")
    items    = pd.read_csv("data/olist_order_items_dataset.csv")
    products = pd.read_csv("data/olist_products_dataset.csv")
    reviews  = pd.read_csv("data/olist_order_reviews_dataset.csv")
    sellers  = pd.read_csv("data/olist_sellers_dataset.csv")
    cat_trans= pd.read_csv("data/product_category_name_translation.csv")

    payments_agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=(
                "payment_type",
                lambda s: s.mode().iat[0] if not s.mode().empty else (s.dropna().iloc[0] if not s.dropna().empty else None),
            ),
        )
    )

    df = orders.merge(payments_agg,  on="order_id",   how="left")
    df = df.merge(customers,     on="customer_id", how="left")
    df = df.merge(items,         on="order_id",    how="left")
    df = df.merge(products,      on="product_id",  how="left")
    df = df.merge(reviews,       on="order_id",    how="left")
    df = df.merge(sellers,       on="seller_id",   how="left")
    df = df.merge(cat_trans,     on="product_category_name", how="left")

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")

    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    # Use English category names where available
    df["category_en"] = df["product_category_name_english"].fillna(df["product_category_name"])
    df["item_revenue"] = df["price"].fillna(0) + df["freight_value"].fillna(0)

    return df


def get_order_level(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per order for order-level KPIs and forecasting."""
    return df.drop_duplicates(subset=["order_id"]).copy()


def get_daily_order_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Daily order-level revenue without item-level double counting."""
    order_level = get_order_level(df)
    daily = (
        order_level.groupby(order_level["order_purchase_timestamp"].dt.date)["payment_value"]
        .sum()
        .reset_index()
    )
    daily.columns = ["date", "revenue"]
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


def get_monthly_order_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly order-level revenue without item-level double counting."""
    order_level = get_order_level(df)
    monthly = (
        order_level.groupby(order_level["order_purchase_timestamp"].dt.to_period("M"))["payment_value"]
        .sum()
        .reset_index()
    )
    monthly.columns = ["period", "revenue"]
    return monthly


def init_session_state():
    defaults = {
        "forecast": None,
        "forecast_growth": None,
        "forecast_details": None,
        "segments": None,
        "segment_details": None,
        "anomalies": 0,
        "anomaly_details": None,
        "decision": None,
        "last_run": {}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def next_page_button(label: str, page_path: str):
    """Render a Next button at the bottom of a page."""
    st.divider()
    col = st.columns([3, 1])[1]
    with col:
        if st.button(f"Next: {label}", use_container_width=True):
            st.switch_page(page_path)
