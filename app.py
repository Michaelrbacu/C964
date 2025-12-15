# app.py
import io
import csv
import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

# -----------------------------
# App settings
# -----------------------------
APP_TITLE = "Sleep Pattern Anomaly Detection System"
MAX_UPLOAD_MB = 10
LOG_PATH = "app.log"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# Canonical column names used internally
REQUIRED_CANON = ["Start", "End", "Sleep quality", "Time in bed", "Heart rate", "Activity"]
OPTIONAL_CANON = ["Sleep Notes", "Wake up"]

# -----------------------------
# CSV loading
# -----------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """
    Attempts to read a CSV file with common encodings and delimiters.
    This helps support exports from different sleep tracking tools.
    """
    text = None
    for enc in ("utf-8", "latin1"):
        try:
            text = content.decode(enc)
            break
        except Exception:
            continue

    if text is not None:
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
            df = pd.read_csv(io.StringIO(text), sep=dialect.delimiter)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass

        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue

    return pd.read_csv(io.BytesIO(content), sep=None, engine="python", on_bad_lines="skip")


def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps common column name variations to the names used by the app.
    Designed around Sleep Cycle–style CSV exports.
    """
    df = df.copy()
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
    lower_map = {c.lower(): c for c in df.columns}
    col_map = {}

    if "start" in lower_map:
        col_map["Start"] = lower_map["start"]
    if "end" in lower_map:
        col_map["End"] = lower_map["end"]

    if "sleep quality" in lower_map:
        col_map["Sleep quality"] = lower_map["sleep quality"]

    if "time in bed" in lower_map:
        col_map["Time in bed"] = lower_map["time in bed"]

    if "heart rate" in lower_map:
        col_map["Heart rate"] = lower_map["heart rate"]

    if "activity (steps)" in lower_map:
        col_map["Activity"] = lower_map["activity (steps)"]
    elif "steps" in lower_map:
        col_map["Activity"] = lower_map["steps"]

    if "sleep notes" in lower_map:
        col_map["Sleep Notes"] = lower_map["sleep notes"]

    if "wake up" in lower_map:
        col_map["Wake up"] = lower_map["wake up"]

    for tgt, src in col_map.items():
        df[tgt] = df[src]

    return df


def parse_duration_to_minutes(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val)
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            if len(parts) == 3:
                return parts[0] * 60 + parts[1] + parts[2] / 60
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_and_map_columns(df)

    missing = [c for c in REQUIRED_CANON if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Start"] = pd.to_datetime(df["Start"], errors="coerce")
    df["End"] = pd.to_datetime(df["End"], errors="coerce")

    sq = df["Sleep quality"].astype(str).str.strip()
    sq = sq.str.replace("%", "", regex=False)
    sq = sq.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    df["Sleep_quality"] = pd.to_numeric(sq, errors="coerce")

    df["Heart_rate"] = pd.to_numeric(df["Heart rate"], errors="coerce")
    df["Activity"] = pd.to_numeric(df["Activity"], errors="coerce")

    df["Time_in_bed_min"] = df["Time in bed"].apply(parse_duration_to_minutes)
    if df["Time_in_bed_min"].median(skipna=True) > 1000:
        df["Time_in_bed_min"] /= 60

    df = df.dropna(subset=["Start", "End"]).reset_index(drop=True)

    mask = df["Time_in_bed_min"].isna()
    df.loc[mask, "Time_in_bed_min"] = (
        (df.loc[mask, "End"] - df.loc[mask, "Start"])
        .dt.total_seconds() / 60
    )

    df["Sleep_Notes"] = df.get("Sleep Notes", "").astype(str)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sleep_duration_min"] = (df["End"] - df["Start"]).dt.total_seconds() / 60
    df["sleep_efficiency"] = df["sleep_duration_min"] / df["Time_in_bed_min"]
    df["sleep_efficiency"] = df["sleep_efficiency"].clip(0, 1)

    df["date"] = df["Start"].dt.date
    df["week_start"] = df["Start"].dt.to_period("W").apply(lambda r: r.start_time.date())

    df["roll_sleep_dur_7"] = df["sleep_duration_min"].rolling(7, min_periods=1).mean()
    df["roll_sleep_eff_7"] = df["sleep_efficiency"].rolling(7, min_periods=1).mean()

    return df


def run_isolation_forest(df: pd.DataFrame, contamination: float):
    features = [
        "sleep_duration_min",
        "sleep_efficiency",
        "Heart_rate",
        "Activity",
        "Sleep_quality",
    ]
    X = df[features].fillna(-999).values

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    model.fit(X)

    df = df.copy()
    df["anomaly"] = model.predict(X) == -1
    df["anomaly_score"] = model.decision_function(X)
    return df, features


def create_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proxy_anom"] = (
        (df["sleep_duration_min"] < 180)
        | (df["sleep_duration_min"] > 720)
        | (df["sleep_efficiency"] < 0.6)
        | (df["Sleep_quality"] < 30)
        | (df["Heart_rate"] > df["Heart_rate"].median() + 15)
    )
    return df


def build_recommendations(df: pd.DataFrame):
    recs = []
    if df["anomaly"].sum() == 0:
        return ["No unusual sleep patterns detected in this range."]

    if (df["sleep_efficiency"] < 0.7).any():
        recs.append("Low sleep efficiency detected. Try maintaining a consistent bedtime.")
    if (df["sleep_duration_min"] < 360).any():
        recs.append("Short sleep duration detected. Consider extending your sleep window.")
    if (df["Heart_rate"] > df["Heart_rate"].median() + 15).any():
        recs.append("Elevated heart rate on some nights. Stress or late caffeine may be contributing.")

    return recs


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

uploaded = st.file_uploader("Upload your sleep CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error("Uploaded file is too large.")
    st.stop()

df_raw = read_csv_bytes(uploaded.read())

with st.expander("Raw data preview"):
    st.write(df_raw.head(20))

df_clean = preprocess(df_raw)
df_feat = engineer_features(df_clean)

st.metric("Records after cleaning", len(df_feat))

start_date, end_date = st.date_input(
    "Select date range",
    [df_feat["Start"].min().date(), df_feat["Start"].max().date()],
)

df_q = df_feat[
    (df_feat["Start"].dt.date >= start_date)
    & (df_feat["Start"].dt.date <= end_date)
]

contamination = st.slider("Anomaly sensitivity", 0.01, 0.2, 0.05, 0.01)
df_anom, feature_cols = run_isolation_forest(df_q, contamination)

# -----------------------------
# Charts
# -----------------------------
fig = px.line(
    df_anom,
    x="Start",
    y="sleep_duration_min",
    color=df_anom["anomaly"].map({True: "Anomaly", False: "Normal"}),
    title="Sleep Duration Over Time",
)
st.plotly_chart(fig, use_container_width=True)

weekly = df_anom.groupby("week_start").mean(numeric_only=True).reset_index()
fig2 = px.bar(
    weekly,
    x="week_start",
    y=["sleep_duration_min", "sleep_efficiency", "Sleep_quality"],
    title="Weekly Sleep Trends",
)
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(
    df_anom,
    x="sleep_duration_min",
    y="Sleep_quality",
    color=df_anom["anomaly"].map({True: "Anomaly", False: "Normal"}),
    title="Duration vs Sleep Quality",
)
st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# Results + Evaluation
# -----------------------------
st.subheader("Decision Support")
for rec in build_recommendations(df_anom):
    st.write("-", rec)

st.subheader("Model Evaluation (Heuristic Labels)")
df_eval = create_proxy_labels(df_anom)
prec = precision_score(df_eval["proxy_anom"], df_anom["anomaly"], zero_division=0)
rec = recall_score(df_eval["proxy_anom"], df_anom["anomaly"], zero_division=0)
f1 = f1_score(df_eval["proxy_anom"], df_anom["anomaly"], zero_division=0)

st.write(f"Precision: {prec:.3f}")
st.write(f"Recall: {rec:.3f}")
st.write(f"F1 Score: {f1:.3f}")

st.subheader("Downloads")
st.download_button("Download cleaned data", df_to_csv_bytes(df_feat), "cleaned_sleep.csv")
st.download_button(
    "Download anomalies only",
    df_to_csv_bytes(df_anom[df_anom["anomaly"]]),
    "sleep_anomalies.csv",
)

st.subheader("Flagged Nights")
st.dataframe(
    df_anom[df_anom["anomaly"]][
        ["Start", "End", "sleep_duration_min", "sleep_efficiency", "Sleep_quality", "Heart_rate", "Activity"]
    ]
)
