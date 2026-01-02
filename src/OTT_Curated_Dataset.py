import OTT_Dataset_Creation as ott
import pandas as pd
import numpy as np

try:
    customers_df = ott.customers_df
    usage_df = ott.usage_logs_df
    payments_df = ott.payments_df
    tickets_df = ott.support_df
    labels_df = ott.churn_labels_df

except Exception as e:
    customers_df = pd.read_csv("data/customers.csv")
    usage_df = pd.read_csv("data/usage_logs.csv")
    payments_df = pd.read_csv("data/payments.csv")
    tickets_df = pd.read_csv("data/support_tickets.csv")
    labels_df = pd.read_csv("data/churn_labels.csv")


usage_agg = (
    usage_df.groupby("customer_id", as_index=False)
    .agg(
        avg_watch_minutes=("watch_minutes", "mean"),
        avg_sessions=("sessions", "mean"),
        binge_days=("binge_flag", "sum"),
        active_days=("date", "nunique"),
    )
)

# -----------------------------
# Aggregate payments (customer level)
# -----------------------------
payments_agg = (
    payments_df.groupby("customer_id", as_index=False)
    .agg(
        late_ratio=("late_payment", "mean"),
        payment_count=("bill_month", "nunique"),
        avg_amount_due=("amount_due", "mean"),
        avg_amount_paid=("amount_paid", "mean"),
    )
)

# -----------------------------
# Aggregate tickets (customer level)
# -----------------------------
tickets_agg = (
    tickets_df.groupby("customer_id", as_index=False)
    .agg(
        ticket_count=("ticket_id", "count"),
        avg_resolution=("resolution_time_hrs", "mean"),
        unresolved_count=("resolved", lambda x: int((x == 0).sum())),
    )
)

# -----------------------------
# Join to curated dataset
# -----------------------------
curated = customers_df.merge(usage_agg, on="customer_id", how="left") \
                   .merge(payments_agg, on="customer_id", how="left") \
                   .merge(tickets_agg, on="customer_id", how="left") \
                   .merge(labels_df, on="customer_id", how="left")

# -----------------------------
# Fill missing values
# -----------------------------
fill_map = {
    "avg_watch_minutes": 0.0,
    "avg_sessions": 0.0,
    "binge_days": 0,
    "active_days": 0,
    "late_ratio": 0.0,
    "payment_count": 0,
    "avg_amount_due": 0.0,
    "avg_amount_paid": 0.0,
    "ticket_count": 0,
    "avg_resolution": 0.0,
    "unresolved_count": 0,
    "churn": 0,
}
curated = curated.fillna(value=fill_map)

# Optional: enforce churn int
curated["churn"] = curated["churn"].astype(int)
curated = curated.drop(columns=["is_active"])

curated.to_csv("data/OTT_curated_dataset.csv", index=False)