from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bi_utils import (
    category_performance,
    compute_kpis,
    discount_impact,
    format_currency,
    format_number,
    load_data_with_quality,
    monthly_revenue_trend,
    payment_distribution,
    quality_report_to_frame,
    region_performance,
    top_products,
)


def create_report(data_path: str, output_dir: str) -> Path:
    df, quality_report = load_data_with_quality(data_path)
    kpis = compute_kpis(df)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    monthly = monthly_revenue_trend(df)
    category = category_performance(df)
    region = region_performance(df)
    payment = payment_distribution(df)
    discount = discount_impact(df)
    products = top_products(df, n=10)
    quality = quality_report_to_frame(quality_report)

    monthly.to_csv(out / "monthly_revenue_trend.csv", index=False)
    category.to_csv(out / "category_performance.csv", index=False)
    region.to_csv(out / "region_performance.csv", index=False)
    payment.to_csv(out / "payment_distribution.csv", index=False)
    discount.to_csv(out / "discount_impact.csv", index=False)
    products.to_csv(out / "top_products.csv", index=False)
    quality.to_csv(out / "data_quality_summary.csv", index=False)

    monthly = monthly.copy()
    monthly["order_month_dt"] = pd.to_datetime(monthly["order_month"] + "-01")

    fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_monthly.add_trace(
        go.Scatter(
            x=monthly["order_month_dt"],
            y=monthly["total_revenue"],
            mode="lines+markers",
            name="Revenue",
            line={"width": 3, "color": "#0b6e4f"},
        ),
        secondary_y=False,
    )
    fig_monthly.add_trace(
        go.Bar(
            x=monthly["order_month_dt"],
            y=monthly["total_orders"],
            name="Orders",
            marker_color="#f4a259",
            opacity=0.55,
        ),
        secondary_y=True,
    )
    fig_monthly.update_layout(title="Monthly Revenue (line) vs Orders (bars)")
    fig_monthly.update_yaxes(title_text="Revenue", secondary_y=False)
    fig_monthly.update_yaxes(title_text="Orders", secondary_y=True)
    fig_monthly.write_html(out / "monthly_revenue_trend.html", include_plotlyjs="cdn")

    category = category.copy()
    total_rev = category["total_revenue"].sum()
    if total_rev > 0:
        category["cum_pct"] = (category["total_revenue"].cumsum() / total_rev * 100).round(2)
    else:
        category["cum_pct"] = 0.0

    fig_category = make_subplots(specs=[[{"secondary_y": True}]])
    fig_category.add_trace(
        go.Bar(
            x=category["product_category"],
            y=category["total_revenue"],
            name="Revenue",
            marker_color="#1f77b4",
        ),
        secondary_y=False,
    )
    fig_category.add_trace(
        go.Scatter(
            x=category["product_category"],
            y=category["cum_pct"],
            mode="lines+markers",
            name="Cumulative %",
            line={"color": "#d1495b", "width": 3},
        ),
        secondary_y=True,
    )
    fig_category.update_layout(title="Category Pareto: Revenue Contribution")
    fig_category.update_yaxes(title_text="Revenue", secondary_y=False)
    fig_category.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
    fig_category.write_html(out / "category_revenue.html", include_plotlyjs="cdn")

    fig_region = px.bar(
        region,
        x="total_revenue",
        y="customer_region",
        color="avg_rating",
        orientation="h",
        title="Regional Revenue Ranking (color = avg rating)",
        color_continuous_scale="Tealgrn",
    )
    fig_region.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_region.write_html(out / "region_revenue.html", include_plotlyjs="cdn")

    top_category = category.iloc[0]["product_category"] if not category.empty else "N/A"
    top_region = region.iloc[0]["customer_region"] if not region.empty else "N/A"
    top_payment = payment.iloc[0]["payment_method"] if not payment.empty else "N/A"

    report_md = out / "executive_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Sales BI Executive Report",
                "",
                "## KPI Snapshot",
                f"- Total Revenue: {format_currency(kpis.total_revenue)}",
                f"- Total Orders: {format_number(kpis.total_orders)}",
                f"- Units Sold: {format_number(kpis.total_units_sold)}",
                f"- Average Order Value: {format_currency(kpis.average_order_value)}",
                f"- Average Discount: {kpis.average_discount_percent:.2f}%",
                f"- Average Rating: {kpis.average_rating:.2f}/5",
                "",
                "## Key Insights",
                f"- Top revenue category: {top_category}.",
                f"- Strongest region by revenue: {top_region}.",
                f"- Most used payment method by revenue contribution: {top_payment}.",
                "- Review discount impact table for elasticity signals across discount bands.",
                "",
                "## Data Quality Summary",
                f"- Input rows: {quality_report.input_rows:,}",
                f"- Output rows: {quality_report.output_rows:,}",
                f"- Rows dropped: {quality_report.rows_dropped:,}",
                f"- Invalid order dates: {quality_report.invalid_order_date_rows:,}",
                f"- Missing dimensions (category/region/payment): {quality_report.missing_dimension_rows:,}",
                f"- Missing total_revenue rows: {quality_report.missing_total_revenue_rows:,}",
                f"- Imputed total_revenue rows: {quality_report.imputed_total_revenue_rows:,}",
                f"- Discount out-of-range rows (<0 or >100): {quality_report.discount_out_of_range_rows:,}",
                f"- Rating out-of-range rows (<0 or >5): {quality_report.out_of_range_rating_rows:,}",
                "",
                "## Generated Artifacts",
                "- `monthly_revenue_trend.csv`",
                "- `category_performance.csv`",
                "- `region_performance.csv`",
                "- `payment_distribution.csv`",
                "- `discount_impact.csv`",
                "- `top_products.csv`",
                "- `data_quality_summary.csv`",
                "- `monthly_revenue_trend.html`",
                "- `category_revenue.html`",
                "- `region_revenue.html`",
                "",
                "## Recommended Next Actions",
                "1. Validate discount strategy for high-volume categories.",
                "2. Launch targeted campaigns in low-revenue regions.",
                "3. Monitor payment-method conversion trends monthly.",
            ]
        ),
        encoding="utf-8",
    )

    return report_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a BI summary report from sales data")
    parser.add_argument(
        "--data",
        default="amazon_sales_dataset.csv",
        help="Path to sales dataset CSV",
    )
    parser.add_argument(
        "--output",
        default="reports",
        help="Directory where report outputs should be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = create_report(args.data, args.output)
    print(f"Report created: {report_path}")


if __name__ == "__main__":
    main()
