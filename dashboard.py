from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bi_utils import (
    apply_filters,
    category_performance,
    compute_kpis,
    correlation_matrix,
    discount_impact,
    forecast_monthly_revenue,
    format_currency,
    format_number,
    load_data_with_quality,
    monthly_revenue_trend,
    quality_report_to_frame,
    region_performance,
)

st.set_page_config(page_title="Sales BI Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def get_data(path: str):
    return load_data_with_quality(path)


def build_sidebar_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")

    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()

    date_range = st.sidebar.date_input(
        "Order date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    selected_categories = st.sidebar.multiselect(
        "Product categories",
        options=sorted(df["product_category"].dropna().unique().tolist()),
        default=sorted(df["product_category"].dropna().unique().tolist()),
    )

    selected_regions = st.sidebar.multiselect(
        "Customer regions",
        options=sorted(df["customer_region"].dropna().unique().tolist()),
        default=sorted(df["customer_region"].dropna().unique().tolist()),
    )

    selected_payments = st.sidebar.multiselect(
        "Payment methods",
        options=sorted(df["payment_method"].dropna().unique().tolist()),
        default=sorted(df["payment_method"].dropna().unique().tolist()),
    )

    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[-1])

    return start_date, end_date, selected_categories, selected_regions, selected_payments


def render_kpi_row(df: pd.DataFrame) -> None:
    kpis = compute_kpis(df)
    top1, top2, top3 = st.columns(3)
    bot1, bot2, bot3 = st.columns(3)

    top1.metric("Total Revenue", format_currency(kpis.total_revenue))
    top2.metric("Orders", format_number(kpis.total_orders))
    top3.metric("Units Sold", format_number(kpis.total_units_sold))
    bot1.metric("Avg Order Value", format_currency(kpis.average_order_value))
    bot2.metric("Avg Discount", f"{kpis.average_discount_percent:.2f}%")
    bot3.metric("Avg Rating", f"{kpis.average_rating:.2f}/5")


def overview_insights(trend: pd.DataFrame, cat: pd.DataFrame, region: pd.DataFrame) -> list[str]:
    """Generate key insights for Executive Overview tab."""
    insights = []

    if not cat.empty and cat["total_revenue"].sum() > 0:
        top_cat = cat.iloc[0]
        revenue_share = (top_cat["total_revenue"] / cat["total_revenue"].sum() * 100)
        insights.append(
            f"Top category '{top_cat['product_category']}' drives {revenue_share:.1f}% of revenue."
        )

    if len(trend) >= 2:
        latest_rev = trend.iloc[-1]["total_revenue"]
        prev_rev = trend.iloc[-2]["total_revenue"]
        if latest_rev > 0:
            mom_change = ((latest_rev - prev_rev) / prev_rev * 100) if prev_rev > 0 else 0
            direction = "↑ increased" if mom_change > 0 else "↓ decreased"
            insights.append(f"Latest month revenue {direction} by {abs(mom_change):.1f}% month-over-month.")
        
        # Add forecast insight
        try:
            trend_with_forecast = forecast_monthly_revenue(trend, periods=3)
            forecast_data = trend_with_forecast[trend_with_forecast["is_forecast"]]
            if not forecast_data.empty:
                avg_forecast = forecast_data["total_revenue"].mean()
                forecast_change = ((avg_forecast - latest_rev) / latest_rev * 100) if latest_rev > 0 else 0
                forecast_direction = "growth" if forecast_change > 0 else "decline"
                insights.append(
                    f"3-month forecast predicts {forecast_direction}: expected avg revenue {forecast_change:+.1f}% vs latest month."
                )
        except Exception:
            pass

    if not region.empty:
        best_region = region.iloc[0]
        worst_region = region.iloc[-1]
        rating_gap = best_region["avg_rating"] - worst_region["avg_rating"]
        if rating_gap >= 0.5:
            insights.append(
                f"Customer satisfaction gap: {best_region['customer_region']} ({best_region['avg_rating']:.2f}) outperforms "
                f"{worst_region['customer_region']} ({worst_region['avg_rating']:.2f}) by {rating_gap:.2f} points."
            )

    return insights


def product_customer_insights(df: pd.DataFrame, metric_mode: str) -> list[str]:
    """Generate key insights for Product & Customer tab."""
    insights = []

    cat_perf = category_performance(df)
    if not cat_perf.empty:
        high_vol = cat_perf[cat_perf["total_units"] > cat_perf["total_units"].quantile(0.75)]
        if len(high_vol) > 0:
            insights.append(
                f"{len(high_vol)} categories achieve high unit volume; "
                f"{high_vol.iloc[0]['product_category']} leads with {format_number(high_vol.iloc[0]['total_units'])} units."
            )

    payment_dist = df.groupby("payment_method")["order_id"].nunique().sort_values(ascending=False)
    if len(payment_dist) > 1:
        dominant = payment_dist.iloc[0]
        second = payment_dist.iloc[1]
        pct_diff = ((dominant - second) / second * 100) if second > 0 else 0
        insights.append(
            f"Payment preference: {payment_dist.index[0]} dominates with "
            f"{pct_diff:.1f}% more orders than {payment_dist.index[1]}."
        )

    region_category = df.groupby(["customer_region", "product_category"])["order_id"].nunique().unstack(fill_value=0)
    if not region_category.empty:
        zero_combos = (region_category == 0).sum().sum()
        total_combos = region_category.size
        penetration = ((total_combos - zero_combos) / total_combos * 100)
        if penetration < 100:
            insights.append(
                f"Market penetration: {penetration:.0f}% of region-category combos have sales. "
                f"Opportunity to expand into {int(zero_combos)} underserved markets."
            )

    return insights


def discount_insights(df: pd.DataFrame, impact: pd.DataFrame) -> list[str]:
    """Generate key insights for Discount Analysis tab."""
    insights = []

    if not impact.empty:
        impact_sorted = impact.sort_values("avg_revenue_per_order", ascending=False)
        best_band = impact_sorted.iloc[0]
        worst_band = impact_sorted.iloc[-1]
        yield_diff = best_band["avg_revenue_per_order"] - worst_band["avg_revenue_per_order"]
        
        best_yield = best_band["avg_revenue_per_order"]
        worst_yield = worst_band["avg_revenue_per_order"]
        best_name = best_band["discount_band"]
        worst_name = worst_band["discount_band"]
        
        insight_txt = f"Discount yield: {best_name} band generates {best_yield:.2f}/order vs {worst_name} at {worst_yield:.2f}/order (difference: {yield_diff:.2f})."
        insights.append(insight_txt)

        volume_boost = impact.sort_values("total_units", ascending=False).iloc[0]
        no_discount = impact[impact["discount_band"] == "0%"]
        if not no_discount.empty:
            base_units_per_order = no_discount.iloc[0]["total_units"] / no_discount.iloc[0]["order_count"] if no_discount.iloc[0]["order_count"] > 0 else 0
            boost_units_per_order = volume_boost["total_units"] / volume_boost["order_count"] if volume_boost["order_count"] > 0 else 0
            if boost_units_per_order > base_units_per_order:
                pct_boost = ((boost_units_per_order - base_units_per_order) / base_units_per_order * 100) if base_units_per_order > 0 else 0
                boost_band = volume_boost["discount_band"]
                insight_txt2 = f"Volume driver: {boost_band} band increases units-per-order by {pct_boost:.0f}% vs no-discount."
                insights.append(insight_txt2)

    return insights


def display_insights(insights: list[str]) -> None:
    """Display insights as formatted bullet points."""
    if insights:
        st.markdown("**Key Insights:**")
        for insight in insights:
            st.markdown(f"• {insight}")


def render_overview_tab(df: pd.DataFrame) -> None:
    st.subheader("Revenue and Demand Overview")

    trend = monthly_revenue_trend(df)
    cat = category_performance(df)
    region = region_performance(df)

    c1, c2 = st.columns(2)

    with c1:
        trend = trend.copy()
        trend["order_month_dt"] = pd.to_datetime(trend["order_month"] + "-01")

        # Add revenue forecast
        trend_with_forecast = forecast_monthly_revenue(trend, periods=3)
        trend_with_forecast["order_month_dt"] = pd.to_datetime(trend_with_forecast["order_month"] + "-01")
        
        historical = trend_with_forecast[~trend_with_forecast["is_forecast"]]
        forecast_data = trend_with_forecast[trend_with_forecast["is_forecast"]]

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Historical revenue
        fig_trend.add_trace(
            go.Scatter(
                x=historical["order_month_dt"],
                y=historical["total_revenue"],
                name="Revenue (Historical)",
                mode="lines+markers",
                line={"width": 3, "color": "#0b6e4f"},
            ),
            secondary_y=False,
        )
        
        # Forecast revenue
        if not forecast_data.empty:
            combined_for_forecast = pd.concat([
                historical.tail(1),
                forecast_data,
            ], ignore_index=True)
            fig_trend.add_trace(
                go.Scatter(
                    x=combined_for_forecast["order_month_dt"],
                    y=combined_for_forecast["total_revenue"],
                    name="Revenue (Forecast)",
                    mode="lines+markers",
                    line={"width": 3, "color": "#c8553d", "dash": "dash"},
                ),
                secondary_y=False,
            )
        
        # Orders bars
        fig_trend.add_trace(
            go.Bar(
                x=historical["order_month_dt"],
                y=historical["total_orders"],
                name="Orders",
                marker_color="#f4a259",
                opacity=0.55,
            ),
            secondary_y=True,
        )
        fig_trend.update_layout(
            title="Monthly Revenue (line) vs Orders (bars) + 3-Month Forecast",
            legend={"orientation": "h", "y": 1.12, "x": 0},
        )
        fig_trend.update_yaxes(title_text="Revenue", secondary_y=False)
        fig_trend.update_yaxes(title_text="Orders", secondary_y=True)
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        pareto = cat.copy()
        total_rev = pareto["total_revenue"].sum()
        if total_rev > 0:
            pareto["cum_pct"] = (pareto["total_revenue"].cumsum() / total_rev * 100).round(2)
        else:
            pareto["cum_pct"] = 0.0

        fig_cat = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cat.add_trace(
            go.Bar(
                x=pareto["product_category"],
                y=pareto["total_revenue"],
                name="Revenue",
                marker_color="#1f77b4",
            ),
            secondary_y=False,
        )
        fig_cat.add_trace(
            go.Scatter(
                x=pareto["product_category"],
                y=pareto["cum_pct"],
                name="Cumulative %",
                mode="lines+markers",
                line={"color": "#d1495b", "width": 3},
            ),
            secondary_y=True,
        )
        fig_cat.update_layout(
            title="Category Pareto: Revenue Contribution",
            legend={"orientation": "h", "y": 1.12, "x": 0},
        )
        fig_cat.update_yaxes(title_text="Revenue", secondary_y=False)
        fig_cat.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
        st.plotly_chart(fig_cat, use_container_width=True)

    region_coords = {
        "North America": {"lat": 39.0, "lon": -98.0},
        "Europe": {"lat": 54.0, "lon": 15.0},
        "Asia": {"lat": 34.0, "lon": 100.0},
        "Middle East": {"lat": 29.0, "lon": 45.0},
    }
    region_map = region.copy()
    region_map["lat"] = region_map["customer_region"].map(lambda r: region_coords.get(r, {}).get("lat"))
    region_map["lon"] = region_map["customer_region"].map(lambda r: region_coords.get(r, {}).get("lon"))
    region_map = region_map.dropna(subset=["lat", "lon"])

    fig_region = px.scatter_geo(
        region_map,
        lat="lat",
        lon="lon",
        size="total_revenue",
        color="avg_rating",
        hover_name="customer_region",
        hover_data={"total_revenue": ":.2f", "order_count": True, "avg_rating": ":.2f", "lat": False, "lon": False},
        text="customer_region",
        title="Regional Revenue Map (size = revenue, color = avg rating)",
        color_continuous_scale="Tealgrn",
        projection="natural earth",
    )
    fig_region.update_traces(textposition="top center")
    fig_region.update_geos(showcoastlines=True, coastlinecolor="LightGray", showcountries=True)
    st.plotly_chart(fig_region, use_container_width=True)

    st.divider()
    insights = overview_insights(trend, cat, region)
    display_insights(insights)


def render_product_customer_tab(df: pd.DataFrame) -> None:
    st.subheader("Product and Customer Insights")

    metric_mode = st.radio(
        "View metric",
        options=["Revenue", "Orders"],
        index=0,
        horizontal=True,
        key="product_customer_metric_mode",
    )
    is_revenue_mode = metric_mode == "Revenue"

    category = category_performance(df)
    payment_region = (
        df.groupby(["customer_region", "payment_method"], as_index=False)
        .agg(order_count=("order_id", "nunique"), total_revenue=("total_revenue", "sum"))
    )
    region_category = (
        df.groupby(["customer_region", "product_category"], as_index=False)
        .agg(order_count=("order_id", "nunique"), total_revenue=("total_revenue", "sum"))
    )

    scatter_y = "total_revenue" if is_revenue_mode else "order_count"
    scatter_y_label = "Revenue" if is_revenue_mode else "Orders"
    scatter_size = "order_count" if is_revenue_mode else "total_units"
    scatter_size_label = "orders" if is_revenue_mode else "units sold"

    payment_y = "total_revenue" if is_revenue_mode else "order_count"
    payment_y_label = "Revenue" if is_revenue_mode else "Orders"

    heatmap_value = "total_revenue" if is_revenue_mode else "order_count"
    heatmap_title = (
        "Revenue Heatmap: Region x Product Category"
        if is_revenue_mode
        else "Order Heatmap: Region x Product Category"
    )

    c1, c2 = st.columns(2)

    with c1:
        fig_category_mix = px.scatter(
            category,
            x="total_units",
            y=scatter_y,
            size=scatter_size,
            color="avg_rating",
            text="product_category",
            title=f"Category Demand vs {scatter_y_label} (bubble size = {scatter_size_label})",
            labels={"total_units": "Units Sold", scatter_y: scatter_y_label, "avg_rating": "Avg Rating"},
            color_continuous_scale="Viridis",
        )
        fig_category_mix.update_traces(textposition="top center")
        st.plotly_chart(fig_category_mix, use_container_width=True)

    with c2:
        fig_payment = px.bar(
            payment_region,
            x="customer_region",
            y=payment_y,
            color="payment_method",
            barmode="stack",
            title=f"Payment Method Mix by Region ({payment_y_label})",
            labels={"customer_region": "Region", payment_y: payment_y_label},
        )
        st.plotly_chart(fig_payment, use_container_width=True)

    heatmap_matrix = (
        region_category.pivot(index="customer_region", columns="product_category", values=heatmap_value)
        .fillna(0)
        .sort_index()
    )

    fig_heatmap = px.imshow(
        heatmap_matrix,
        text_auto=".2s",
        aspect="auto",
        title=heatmap_title,
        color_continuous_scale="YlGnBu",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()
    insights = product_customer_insights(df, metric_mode)
    display_insights(insights)


def render_discount_tab(df: pd.DataFrame) -> None:
    st.subheader("Discount and Revenue Impact")

    impact = discount_impact(df)
    impact = impact.copy()
    impact["avg_revenue_per_order"] = impact.apply(
        lambda row: (row["total_revenue"] / row["order_count"]) if row["order_count"] else 0.0,
        axis=1,
    )
    corr = correlation_matrix(df)

    c1, c2 = st.columns(2)

    with c1:
        fig_impact = make_subplots(specs=[[{"secondary_y": True}]])
        fig_impact.add_trace(
            go.Bar(
                x=impact["discount_band"],
                y=impact["total_revenue"],
                name="Total Revenue",
                marker_color="#287271",
            ),
            secondary_y=False,
        )
        fig_impact.add_trace(
            go.Scatter(
                x=impact["discount_band"],
                y=impact["avg_revenue_per_order"],
                name="Avg Revenue / Order",
                mode="lines+markers",
                line={"width": 3, "color": "#c8553d"},
            ),
            secondary_y=True,
        )
        fig_impact.update_layout(
            title="Discount Band Performance: Revenue and Order Yield",
            legend={"orientation": "h", "y": 1.12, "x": 0},
        )
        fig_impact.update_yaxes(title_text="Total Revenue", secondary_y=False)
        fig_impact.update_yaxes(title_text="Avg Revenue per Order", secondary_y=True)
        st.plotly_chart(fig_impact, use_container_width=True)

    with c2:
        fig_units = px.bar(
            impact,
            x="discount_band",
            y="total_units",
            color="avg_rating",
            title="Units Sold by Discount Band (color = avg rating)",
            color_continuous_scale="Temps",
        )
        st.plotly_chart(fig_units, use_container_width=True)

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Matrix",
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()
    insights = discount_insights(df, impact)
    display_insights(insights)


def render_export_tab(df: pd.DataFrame) -> None:
    st.subheader("Data Export")
    st.write("Download the filtered data currently in view.")

    export_df = df.copy()
    export_df["order_date"] = export_df["order_date"].dt.strftime("%Y-%m-%d")
    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered CSV",
        data=csv_data,
        file_name="filtered_sales_data.csv",
        mime="text/csv",
    )

    st.dataframe(export_df.head(100), use_container_width=True)


def main() -> None:
    st.title("Amazon Sales Business Intelligence Dashboard")
    st.caption("Interactive analytics across revenue, demand, discounting, and customer behavior")

    data_path = st.sidebar.text_input("Dataset path", value="amazon_sales_dataset.csv")

    try:
        df, quality_report = get_data(data_path)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return

    with st.sidebar.expander("Data quality summary", expanded=False):
        st.caption("Validation snapshot from ingestion step")
        st.write(f"Input rows: {quality_report.input_rows:,}")
        st.write(f"Output rows: {quality_report.output_rows:,}")
        st.write(f"Rows dropped: {quality_report.rows_dropped:,}")
        st.write(f"Invalid order dates: {quality_report.invalid_order_date_rows:,}")
        st.write(f"Missing core dimensions: {quality_report.missing_dimension_rows:,}")
        st.write(f"Missing total revenue: {quality_report.missing_total_revenue_rows:,}")
        st.write(f"Imputed total revenue: {quality_report.imputed_total_revenue_rows:,}")
        st.write(f"Negative price rows: {quality_report.negative_price_rows:,}")
        st.write(f"Discount out-of-range rows: {quality_report.discount_out_of_range_rows:,}")
        st.write(f"Negative quantity rows: {quality_report.negative_quantity_rows:,}")
        st.write(f"Rating out-of-range rows: {quality_report.out_of_range_rating_rows:,}")
        st.write(f"Negative revenue rows: {quality_report.negative_revenue_rows:,}")
        st.dataframe(quality_report_to_frame(quality_report), use_container_width=True)

    start_date, end_date, categories, regions, payments = build_sidebar_filters(df)
    filtered_df = apply_filters(
        df,
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        regions=regions,
        payment_methods=payments,
    )

    if filtered_df.empty:
        st.warning("No records found for the selected filters.")
        return

    render_kpi_row(filtered_df)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Executive Overview", "Product & Customer", "Discount Analysis", "Export"]
    )

    with tab1:
        render_overview_tab(filtered_df)
    with tab2:
        render_product_customer_tab(filtered_df)
    with tab3:
        render_discount_tab(filtered_df)
    with tab4:
        render_export_tab(filtered_df)


if __name__ == "__main__":
    main()
