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


def render_overview_tab(df: pd.DataFrame) -> None:
    st.subheader("Revenue and Demand Overview")

    trend = monthly_revenue_trend(df)
    cat = category_performance(df)
    region = region_performance(df)

    c1, c2 = st.columns(2)

    with c1:
        trend = trend.copy()
        trend["order_month_dt"] = pd.to_datetime(trend["order_month"] + "-01")

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(
            go.Scatter(
                x=trend["order_month_dt"],
                y=trend["total_revenue"],
                name="Revenue",
                mode="lines+markers",
                line={"width": 3, "color": "#0b6e4f"},
            ),
            secondary_y=False,
        )
        fig_trend.add_trace(
            go.Bar(
                x=trend["order_month_dt"],
                y=trend["total_orders"],
                name="Orders",
                marker_color="#f4a259",
                opacity=0.55,
            ),
            secondary_y=True,
        )
        fig_trend.update_layout(
            title="Monthly Revenue (line) vs Orders (bars)",
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
