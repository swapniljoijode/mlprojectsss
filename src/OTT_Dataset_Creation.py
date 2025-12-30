import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
N_CUSTOMERS = 10000
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2024, 12, 31)

# create output folder
os.makedirs("data", exist_ok=True)

# helper to random date between two
def random_date(start, end):
    delta = end - start
    return start + timedelta(days=np.random.randint(0, delta.days + 1))

# -----------------------------
# 1) CUSTOMERS TABLE
# -----------------------------
customer_ids = [f"CUST_{i:06d}" for i in range(1, N_CUSTOMERS + 1)]

ages = np.random.randint(18, 70, size=N_CUSTOMERS)
genders = np.random.choice(["Male", "Female", "Other"], size=N_CUSTOMERS, p=[0.47, 0.47, 0.06])
countries = np.random.choice(["US", "CA", "UK", "IN", "BR", "DE", "AU"], size=N_CUSTOMERS,
                             p=[0.35, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1])
membership_types = np.random.choice(["Basic", "Standard", "Premium"], size=N_CUSTOMERS,
                                    p=[0.4, 0.4, 0.2])
devices_owned = np.random.randint(1, 5, size=N_CUSTOMERS)
profile_count = np.random.randint(1, 5, size=N_CUSTOMERS)

join_dates = [random_date(START_DATE, END_DATE - timedelta(days=30))  # joined at least 30 days before end
              for _ in range(N_CUSTOMERS)]

customers_df = pd.DataFrame({
    "customer_id": customer_ids,
    "age": ages,
    "gender": genders,
    "country": countries,
    "join_date": join_dates,
    "membership_type": membership_types,
    "devices_owned": devices_owned,
    "profile_count": profile_count,
})

# is_active we will fill later based on churn
customers_df["is_active"] = 1

print("Customers:", customers_df.shape)
customers_df.head()

# -----------------------------
# 2) USAGE LOGS TABLE (last 90 days)
# -----------------------------
usage_rows = []

usage_start = END_DATE - timedelta(days=90)

for cid, mtype, jdate in zip(customers_df["customer_id"],
                             customers_df["membership_type"],
                             customers_df["join_date"]):
    # active period for logging
    user_start = max(jdate, usage_start)
    if user_start > END_DATE:
        continue

    days = (END_DATE - user_start).days + 1

    # base engagement by membership
    if mtype == "Premium":
        base_watch = np.random.normal(140, 40)  # minutes/day
    elif mtype == "Standard":
        base_watch = np.random.normal(100, 30)
    else:  # Basic
        base_watch = np.random.normal(70, 25)
    base_watch = max(10, base_watch)

    for d in range(days):
        day = user_start + timedelta(days=d)
        # some users skip days
        if np.random.rand() < 0.15:
            watch_minutes = 0
            sessions = 0
            unique_titles = 0
            binge_flag = 0
        else:
            # noise
            today_watch = max(0, np.random.normal(base_watch, base_watch * 0.4))
            watch_minutes = int(today_watch)
            sessions = np.random.randint(1, 5)
            unique_titles = np.random.randint(1, min(5, sessions + 1))
            binge_flag = int(watch_minutes > 180)

        device_type = np.random.choice(["TV", "Mobile", "Web", "Tablet"], p=[0.5, 0.25, 0.15, 0.10])

        usage_rows.append([
            cid, day.date(), watch_minutes, sessions, unique_titles, device_type, binge_flag
        ])

usage_logs_df = pd.DataFrame(usage_rows, columns=[
    "customer_id", "date", "watch_minutes", "sessions",
    "unique_titles", "device_type", "binge_flag"
])

print("Usage logs:", usage_logs_df.shape)
usage_logs_df.head()

# -----------------------------
# 3) PAYMENTS TABLE (last 12 months)
# -----------------------------
plan_price = {"Basic": 9.99, "Standard": 15.49, "Premium": 19.99}

payment_rows = []
billing_start = END_DATE - timedelta(days=365)

for cid, mtype, jdate in zip(customers_df["customer_id"],
                             customers_df["membership_type"],
                             customers_df["join_date"]):
    # bills only after join date
    start_month = max(jdate.replace(day=1), billing_start.replace(day=1))

    bill_date = start_month
    while bill_date <= END_DATE:
        amount_due = plan_price[mtype]

        # late/missed payments more likely for younger + Basic plan
        age = customers_df.loc[customers_df["customer_id"] == cid, "age"].iloc[0]
        base_late_prob = 0.03
        if mtype == "Basic":
            base_late_prob += 0.05
        if age < 25:
            base_late_prob += 0.04

        is_late = np.random.rand() < base_late_prob

        if is_late:
            # sometimes partial or no payment
            pay_factor = np.random.choice([0.0, 0.5, 1.0], p=[0.4, 0.3, 0.3])
            amount_paid = round(amount_due * pay_factor, 2)
            late_flag = 1
        else:
            amount_paid = amount_due
            late_flag = 0

        payment_rows.append([
            cid, bill_date.date(), amount_due, amount_paid, late_flag
        ])

        # next month
        if bill_date.month == 12:
            bill_date = bill_date.replace(year=bill_date.year + 1, month=1)
        else:
            bill_date = bill_date.replace(month=bill_date.month + 1)

payments_df = pd.DataFrame(payment_rows, columns=[
    "customer_id", "bill_month", "amount_due", "amount_paid", "late_payment"
])

print("Payments:", payments_df.shape)
payments_df.head()

# -----------------------------
# 4) SUPPORT TICKETS TABLE
# -----------------------------
ticket_rows = []
ticket_counter = 1

for cid in customers_df["customer_id"]:
    # base probability of having tickets
    # frustrated users (Basic) tend to open more
    mtype = customers_df.loc[customers_df["customer_id"] == cid, "membership_type"].iloc[0]
    base_prob = 0.1 if mtype == "Premium" else 0.18 if mtype == "Standard" else 0.25

    # draw number of tickets (0–5)
    n_tickets = np.random.poisson(lam=base_prob * 3)
    n_tickets = min(n_tickets, 5)

    for _ in range(n_tickets):
        created_date = random_date(START_DATE, END_DATE)
        issue_type = np.random.choice(["Billing", "Technical", "Content"], p=[0.3, 0.5, 0.2])
        severity = np.random.choice(["Low", "Medium", "High"], p=[0.5, 0.35, 0.15])

        # resolution time based on severity
        if severity == "Low":
            res_time = max(0.5, np.random.normal(8, 3))
        elif severity == "Medium":
            res_time = max(1, np.random.normal(24, 8))
        else:
            res_time = max(2, np.random.normal(48, 16))

        # some tickets unresolved
        resolved = int(np.random.rand() > 0.05)

        ticket_rows.append([
            f"TKT_{ticket_counter:07d}", cid, created_date.date(),
            issue_type, severity, round(res_time, 2), resolved
        ])
        ticket_counter += 1

support_df = pd.DataFrame(ticket_rows, columns=[
    "ticket_id", "customer_id", "created_date",
    "issue_type", "severity", "resolution_time_hrs", "resolved"
])

print("Support tickets:", support_df.shape)
support_df.head()

# -----------------------------
# 5) CHURN LABELS
# -----------------------------
# Observation window = last 60 days
obs_start = END_DATE - timedelta(days=60)

# aggregate usage in last 60 days
usage_recent = usage_logs_df[usage_logs_df["date"] >= obs_start.date()]

usage_agg = usage_recent.groupby("customer_id").agg(
    avg_watch_minutes=("watch_minutes", "mean"),
    avg_sessions=("sessions", "mean"),
    binge_days=("binge_flag", "sum"),
).fillna(0)

# aggregate payments in last 6 months
pay_start = END_DATE - timedelta(days=180)
payments_recent = payments_df[payments_df["bill_month"] >= pay_start.date()]

pay_agg = payments_recent.groupby("customer_id").agg(
    late_ratio=("late_payment", "mean")
).fillna(0)

# aggregate support tickets in last 6 months
tickets_recent = support_df[support_df["created_date"] >= pay_start.date()]
ticket_agg = tickets_recent.groupby("customer_id").agg(
    ticket_count=("ticket_id", "count"),
    avg_resolution=("resolution_time_hrs", "mean")
).fillna(0)

# join all aggregates with customers
features_df = customers_df.set_index("customer_id").join(
    [usage_agg, pay_agg, ticket_agg], how="left"
).fillna({
    "avg_watch_minutes": 0,
    "avg_sessions": 0,
    "binge_days": 0,
    "late_ratio": 0,
    "ticket_count": 0,
    "avg_resolution": 0
})

features_df["tenure_days"] = (END_DATE - features_df["join_date"]).dt.days

# churn score based on heuristics
scores = []

for idx, row in features_df.iterrows():
    score = 0.0

    # low usage
    if row["avg_watch_minutes"] < 40:
        score += np.random.uniform(0.1,0.4)
    elif row["avg_watch_minutes"] < 80:
        score += np.random.uniform(0.05,0.2)

    # low sessions
    if row["avg_sessions"] < 1.2:
        score += np.random.uniform(0.05,0.2)

    # few binge days (less engaged)
    if row["binge_days"] < 2:
        score += np.random.uniform(0.01,0.1)

    # late payments
    if row["late_ratio"] > 0.3:
        score += np.random.uniform(0.1,0.4)
    elif row["late_ratio"] > 0.1:
        score += np.random.uniform(0.05,0.2)

    # many tickets or slow resolution
    if row["ticket_count"] >= 3:
        score += np.random.uniform(0.01, 0.25)
    if row["avg_resolution"] > 36:
        score += np.random.uniform(0.01, 0.25)

    # membership type
    if row["membership_type"] == "Basic":
        score += np.random.uniform(0.05, 0.2)
    elif row["membership_type"] == "Premium":
        score -= np.random.uniform(0.01, 0.1)  # stickier

    if row["tenure_days"] < 60:
        score += np.random.uniform(0.05, 0.2)
    if row["tenure_days"] > 600:
        score += np.random.uniform(0.05, 0.15)

    # age – very young less loyal
    if row["age"] < 25:
        score += np.random.uniform(0.01,0.1)

    # some noise
    score += np.random.normal(0, 0.08)

    scores.append(score)

features_df["churn_score"] = scores

# convert score to probability
# anything above ~0.5 gets high probability
prob = 1 / (1 + np.exp(- (features_df["churn_score"] * 2)))
features_df["churn_prob"] = prob

# sample actual churn
features_df["churn"] = (np.random.rand(len(features_df)) < features_df["churn_prob"]).astype(int)

# set is_active = 0 for churned
customers_df = customers_df.set_index("customer_id")
customers_df = customers_df.reset_index()

churn_labels_df = features_df[["churn"]].reset_index()

print("Churn rate:", churn_labels_df["churn"].mean())
churn_labels_df.head()



customers_df.to_csv("data/customers.csv", index=False)
usage_logs_df.to_csv("data/usage_logs.csv", index=False)
payments_df.to_csv("data/payments.csv", index=False)
support_df.to_csv("data/support_tickets.csv", index=False)
churn_labels_df.to_csv("data/churn_labels.csv", index=False)

print("Saved files in ./data:")
os.listdir("data")
