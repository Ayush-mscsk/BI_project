from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


EXPECTED_COLUMNS = [
    "order_id",
    "order_date",
    "product_id",
    "product_category",
    "price",
    "discount_percent",
    "quantity_sold",
    "customer_region",
    "payment_method",
    "rating",
    "review_count",
    "discounted_price",
    "total_revenue",
]


@dataclass(frozen=True)
class KPIBundle:
    total_orders: int
    total_revenue: float
    total_units_sold: int
    average_order_value: float
    average_discount_percent: float
    average_rating: float


@dataclass(frozen=True)
class DataQualityReport:
    input_rows: int
    output_rows: int
    rows_dropped: int
    invalid_order_date_rows: int
    missing_dimension_rows: int
    missing_total_revenue_rows: int
    imputed_total_revenue_rows: int
    negative_price_rows: int
    discount_out_of_range_rows: int
    negative_quantity_rows: int
    out_of_range_rating_rows: int
    negative_revenue_rows: int

    def as_dict(self) -> dict[str, int]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "rows_dropped": self.rows_dropped,
            "invalid_order_date_rows": self.invalid_order_date_rows,
            "missing_dimension_rows": self.missing_dimension_rows,
            "missing_total_revenue_rows": self.missing_total_revenue_rows,
            "imputed_total_revenue_rows": self.imputed_total_revenue_rows,
            "negative_price_rows": self.negative_price_rows,
            "discount_out_of_range_rows": self.discount_out_of_range_rows,
            "negative_quantity_rows": self.negative_quantity_rows,
            "out_of_range_rating_rows": self.out_of_range_rating_rows,
            "negative_revenue_rows": self.negative_revenue_rows,
        }


def _prepare_data_with_report(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Prepare sales data and return both cleaned data and quality diagnostics."""
    input_rows = int(len(df))
    working = df.copy()

    working["order_date"] = pd.to_datetime(working["order_date"], errors="coerce")

    numeric_columns = [
        "price",
        "discount_percent",
        "quantity_sold",
        "rating",
        "review_count",
        "discounted_price",
        "total_revenue",
    ]
    for col in numeric_columns:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    invalid_order_date_rows = int(working["order_date"].isna().sum())
    missing_dimension_mask = working[["product_category", "customer_region", "payment_method"]].isna().any(axis=1)
    missing_dimension_rows = int(missing_dimension_mask.sum())

    missing_total_revenue_rows = int(working["total_revenue"].isna().sum())
    can_impute_revenue = (
        working["total_revenue"].isna()
        & working["discounted_price"].notna()
        & working["quantity_sold"].notna()
    )
    imputed_total_revenue_rows = int(can_impute_revenue.sum())
    working.loc[can_impute_revenue, "total_revenue"] = (
        working.loc[can_impute_revenue, "discounted_price"]
        * working.loc[can_impute_revenue, "quantity_sold"]
    )

    negative_price_rows = int((working["price"] < 0).fillna(False).sum())
    discount_out_of_range_rows = int(((working["discount_percent"] < 0) | (working["discount_percent"] > 100)).fillna(False).sum())
    negative_quantity_rows = int((working["quantity_sold"] < 0).fillna(False).sum())
    out_of_range_rating_rows = int(((working["rating"] < 0) | (working["rating"] > 5)).fillna(False).sum())
    negative_revenue_rows = int((working["total_revenue"] < 0).fillna(False).sum())

    drop_mask = working["order_date"].isna() | missing_dimension_mask
    cleaned = working.loc[~drop_mask].copy()

    cleaned["order_month"] = cleaned["order_date"].dt.to_period("M").astype(str)
    cleaned["order_year"] = cleaned["order_date"].dt.year
    cleaned["order_quarter"] = "Q" + cleaned["order_date"].dt.quarter.astype(str)
    cleaned["discount_band"] = pd.cut(
        cleaned["discount_percent"].fillna(0),
        bins=[-0.1, 0, 10, 20, 30, 100],
        labels=["0%", "1-10%", "11-20%", "21-30%", ">30%"],
    )

    output_rows = int(len(cleaned))
    report = DataQualityReport(
        input_rows=input_rows,
        output_rows=output_rows,
        rows_dropped=input_rows - output_rows,
        invalid_order_date_rows=invalid_order_date_rows,
        missing_dimension_rows=missing_dimension_rows,
        missing_total_revenue_rows=missing_total_revenue_rows,
        imputed_total_revenue_rows=imputed_total_revenue_rows,
        negative_price_rows=negative_price_rows,
        discount_out_of_range_rows=discount_out_of_range_rows,
        negative_quantity_rows=negative_quantity_rows,
        out_of_range_rating_rows=out_of_range_rating_rows,
        negative_revenue_rows=negative_revenue_rows,
    )

    return cleaned, report



def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the sales data with derived date fields."""
    df = pd.read_csv(csv_path)

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    prepared_df, _ = _prepare_data_with_report(df)
    return prepared_df


def load_data_with_quality(csv_path: str) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load, prepare, and return data quality diagnostics for BI reporting."""
    df = pd.read_csv(csv_path)

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return _prepare_data_with_report(df)


def quality_report_to_frame(report: DataQualityReport) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": list(report.as_dict().keys()),
            "value": list(report.as_dict().values()),
        }
    )



def apply_filters(
    df: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    categories: Iterable[str] | None = None,
    regions: Iterable[str] | None = None,
    payment_methods: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return filtered data for dashboard interactions and report views."""
    filtered = df.copy()

    if start_date is not None:
        filtered = filtered[filtered["order_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        filtered = filtered[filtered["order_date"] <= pd.Timestamp(end_date)]

    if categories:
        filtered = filtered[filtered["product_category"].isin(categories)]
    if regions:
        filtered = filtered[filtered["customer_region"].isin(regions)]
    if payment_methods:
        filtered = filtered[filtered["payment_method"].isin(payment_methods)]

    return filtered



def compute_kpis(df: pd.DataFrame) -> KPIBundle:
    """Compute top-line KPI metrics for the active dataset slice."""
    if df.empty:
        return KPIBundle(0, 0.0, 0, 0.0, 0.0, 0.0)

    total_orders = int(df["order_id"].nunique())
    total_revenue = float(df["total_revenue"].sum())
    total_units_sold = int(df["quantity_sold"].sum())
    average_order_value = float(total_revenue / total_orders) if total_orders else 0.0
    average_discount_percent = float(df["discount_percent"].mean())
    average_rating = float(df["rating"].mean())

    return KPIBundle(
        total_orders=total_orders,
        total_revenue=total_revenue,
        total_units_sold=total_units_sold,
        average_order_value=average_order_value,
        average_discount_percent=average_discount_percent,
        average_rating=average_rating,
    )



def monthly_revenue_trend(df: pd.DataFrame) -> pd.DataFrame:
    trend = (
        df.groupby("order_month", as_index=False)
        .agg(total_revenue=("total_revenue", "sum"), total_orders=("order_id", "nunique"))
        .sort_values("order_month")
    )
    return trend



def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("product_category", as_index=False)
        .agg(
            total_revenue=("total_revenue", "sum"),
            total_units=("quantity_sold", "sum"),
            avg_rating=("rating", "mean"),
            avg_discount=("discount_percent", "mean"),
            order_count=("order_id", "nunique"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    return summary



def region_performance(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("customer_region", as_index=False)
        .agg(
            total_revenue=("total_revenue", "sum"),
            total_units=("quantity_sold", "sum"),
            avg_rating=("rating", "mean"),
            order_count=("order_id", "nunique"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    return summary



def payment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("payment_method", as_index=False)
        .agg(total_revenue=("total_revenue", "sum"), order_count=("order_id", "nunique"))
        .sort_values("total_revenue", ascending=False)
    )
    return summary



def discount_impact(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("discount_band", observed=False, as_index=False)
        .agg(
            total_revenue=("total_revenue", "sum"),
            total_units=("quantity_sold", "sum"),
            avg_rating=("rating", "mean"),
            order_count=("order_id", "nunique"),
        )
        .sort_values("discount_band")
    )
    return summary



def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    summary = (
        df.groupby(["product_id", "product_category"], as_index=False)
        .agg(total_revenue=("total_revenue", "sum"), total_units=("quantity_sold", "sum"))
        .sort_values("total_revenue", ascending=False)
        .head(n)
    )
    return summary



def customer_sentiment_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact table for rating and review behavior analysis."""
    working = df.copy()
    working["rating_bucket"] = pd.cut(
        working["rating"],
        bins=[0, 2, 3, 4, 5],
        labels=["Low (<=2)", "Fair (2-3)", "Good (3-4)", "Excellent (4-5)"],
        include_lowest=True,
    )

    summary = (
        working.groupby("rating_bucket", observed=False, as_index=False)
        .agg(avg_reviews=("review_count", "mean"), orders=("order_id", "nunique"))
        .sort_values("rating_bucket")
    )
    return summary



def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "price",
        "discount_percent",
        "quantity_sold",
        "rating",
        "review_count",
        "discounted_price",
        "total_revenue",
    ]
    corr = df[numeric_cols].corr(numeric_only=True)
    return corr


def forecast_monthly_revenue(trend: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    """
    Forecast monthly revenue using exponential smoothing.
    
    Args:
        trend: DataFrame with order_month and total_revenue columns (from monthly_revenue_trend)
        periods: Number of months to forecast (default 3)
    
    Returns:
        DataFrame with historical data + forecast rows
    """
    if len(trend) < 2:
        return trend
    
    try:
        # Convert month string to datetime for proper ordering
        trend_copy = trend.copy()
        trend_copy["order_month_dt"] = pd.to_datetime(trend_copy["order_month"] + "-01")
        trend_copy = trend_copy.sort_values("order_month_dt")
        
        revenue_series = trend_copy["total_revenue"].values
        
        # Use exponential smoothing for forecast
        if len(revenue_series) >= 4:
            model = ExponentialSmoothing(
                revenue_series,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
            forecast_values = fitted.forecast(steps=periods)
        else:
            # Fallback to simple linear trend for small datasets
            x = np.arange(len(revenue_series))
            z = np.polyfit(x, revenue_series, 1)
            p = np.poly1d(z)
            last_idx = len(revenue_series) - 1
            forecast_values = p(np.arange(last_idx + 1, last_idx + 1 + periods))
        
        # Ensure no negative forecasts
        forecast_values = np.maximum(forecast_values, 0)
        
        # Build forecast dataframe
        last_date = trend_copy["order_month_dt"].max()
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq="MS")
        forecast_months = forecast_dates.strftime("%Y-%m").tolist()
        
        forecast_df = pd.DataFrame({
            "order_month": forecast_months,
            "total_revenue": forecast_values,
            "is_forecast": True,
        })
        
        # Keep all original columns for historical data, add NaN for forecast
        historical_cols = [c for c in trend_copy.columns if c != "order_month_dt"]
        result = pd.concat([
            trend_copy[historical_cols].assign(is_forecast=False),
            forecast_df.assign(total_orders=np.nan),
        ], ignore_index=True)
        
        return result
    
    except Exception:
        # Fallback: return original trend if forecast fails
        return trend.assign(is_forecast=False)


def format_currency(value: float) -> str:
    return f"${value:,.2f}"



def format_number(value: float | int) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:,.2f}"
