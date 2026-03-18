# Amazon Sales BI Project

End-to-end Business Intelligence project built on `amazon_sales_dataset.csv` with:

- Interactive dashboard (Streamlit + Plotly)
- Automated executive reporting pipeline
- Reusable analytics code for KPIs and summary tables

## Project Goals

- Track revenue, orders, and units sold over time
- Compare performance across product categories and regions
- Analyze discount impact on demand and revenue
- Understand payment-method contribution and customer sentiment proxies

## Project Structure

```text
.
├── amazon_sales_dataset.csv
├── bi_utils.py
├── dashboard.py
├── generate_report.py
├── requirements.txt
└── README.md
```

## Dataset Fields Used

- `order_id`
- `order_date`
- `product_id`
- `product_category`
- `price`
- `discount_percent`
- `quantity_sold`
- `customer_region`
- `payment_method`
- `rating`
- `review_count`
- `discounted_price`
- `total_revenue`

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Interactive Dashboard

```bash
streamlit run dashboard.py
```

The dashboard includes:

- Sidebar filters (date, category, region, payment method)
- Data quality summary panel (invalid, dropped, and imputed rows)
- Executive KPI row with trend chart (historical + 3-month forecast overlay)
- Monthly revenue forecasting with exponential smoothing
- Dynamic insights (key factors, anomalies, forecast predictions)
- Product & Customer analysis with revenue/orders toggle
- Discount analysis with yield breakdown
- Filtered data export to CSV

## Generate Automated Report

```bash
python generate_report.py --data amazon_sales_dataset.csv --output reports
```

This creates:

- `reports/executive_report.md`
- CSV summary tables for BI reporting
- `reports/data_quality_summary.csv`
- HTML charts for shareable visuals

## KPI Definitions

- **Total Revenue**: Sum of `total_revenue`
- **Orders**: Distinct `order_id`
- **Units Sold**: Sum of `quantity_sold`
- **Average Order Value**: Total Revenue / Orders
- **Average Discount**: Mean of `discount_percent`
- **Average Rating**: Mean of `rating`

## Business Questions Answered

- Which product categories drive the most revenue?
- Which regions perform best and where is growth opportunity?
- How do discount bands affect units sold and revenue?
- Which payment methods contribute most to revenue?

## Recent Enhancements

✓ **Forecast Models** - Added 3-month revenue forecasts using exponential smoothing with fallback to polynomial regression for small datasets. Forecast visualization overlaid on historical trend with dynamic insights.

## Next BI Enhancements

1. Add confidence intervals to forecast models
2. Introduce cohort analysis for customer behavior
3. Add testing infrastructure (unit tests for KPI calculations and forecast accuracy)
3. Publish report outputs to a BI tool (Power BI/Tableau/Looker).