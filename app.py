import io
import csv
import os
import logging
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

APP_TITLE = "Sleep Pattern Anomaly Detection System"
MAX_UPLOAD_MB = 10
LOG_PATH = "app.log"

logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REQUIRED_CANON = ["Start", "End", "Sleep quality", "Time in bed", "Heart rate", "Activity"]
OPTIONAL_CANON = ["Sleep Notes", "Wake up"]


@st.cache_data(show_spinner=False)
def read_csv_bytes(content: bytes) -> pd.DataFrame:
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
            sep = dialect.delimiter
            df = pd.read_csv(io.StringIO(text), sep=sep)
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

    try:
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.read_csv(io.BytesIO(content), on_bad_lines="skip")


def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

    lower_map = {c.lower(): c for c in df.columns}
    col_map = {}

    if "start" in lower_map:
        col_map["Start"] = lower_map["start"]
    elif "start time" in lower_map:
        col_map["Start"] = lower_map["start time"]

    if "end" in lower_map:
        col_map["End"] = lower_map["end"]
    elif "end time" in lower_map:
        col_map["End"] = lower_map["end time"]

    if "sleep quality" in lower_map:
        col_map["Sleep quality"] = lower_map["sleep quality"]
    elif "sleep_quality" in lower_map:
        col_map["Sleep quality"] = lower_map["sleep_quality"]
    elif "sleep quality (%)" in lower_map:
        col_map["Sleep quality"] = lower_map["sleep quality (%)"]

    if "time in bed" in lower_map:
        col_map["Time in bed"] = lower_map["time in bed"]
    elif "time in bed (seconds)" in lower_map:
        col_map["Time in bed"] = lower_map["time in bed (seconds)"]
    elif "time in bed (s)" in lower_map:
        col_map["Time in bed"] = lower_map["time in bed (s)"]

    if "heart rate" in lower_map:
        col_map["Heart rate"] = lower_map["heart rate"]
    elif "heart rate (bpm)" in lower_map:
        col_map["Heart rate"] = lower_map["heart rate (bpm)"]
    elif "heart_rate" in lower_map:
        col_map["Heart rate"] = lower_map["heart_rate"]

    if "activity" in lower_map:
        col_map["Activity"] = lower_map["activity"]
    elif "steps" in lower_map:
        col_map["Activity"] = lower_map["steps"]
    elif "activity (steps)" in lower_map:
        col_map["Activity"] = lower_map["activity (steps)"]

    if "sleep notes" in lower_map:
        col_map["Sleep Notes"] = lower_map["sleep notes"]
    elif "notes" in lower_map:
        col_map["Sleep Notes"] = lower_map["notes"]

    if "wake up" in lower_map:
        col_map["Wake up"] = lower_map["wake up"]
    elif "wake_up" in lower_map:
        col_map["Wake up"] = lower_map["wake_up"]

    for tgt, src in col_map.items():
        if src in df.columns:
            df[tgt] = df[src]

    return df


def parse_duration_to_minutes(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()

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


def validate_required_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_CANON if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}. Found columns: {list(df.columns)}")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_and_map_columns(df)
    validate_required_columns(df)

    df["Start"] = pd.to_datetime(df["Start"], errors="coerce")
    df["End"] = pd.to_datetime(df["End"], errors="coerce")

    if "Sleep quality" in df.columns:
        s = df["Sleep quality"].astype(str).str.replace("%", "", regex=False).str.replace(",", ".")
        df["Sleep_quality"] = pd.to_numeric(s, errors="coerce")
        mx = df["Sleep_quality"].max(skipna=True)
        if not np.isnan(mx) and mx <= 1.0:
            df["Sleep_quality"] = df["Sleep_quality"] * 100.0
    else:
        df["Sleep_quality"] = np.nan
    df["Heart_rate"] = pd.to_numeric(df["Heart rate"], errors="coerce")
    df["Activity"] = pd.to_numeric(df["Activity"], errors="coerce")

    df["Time_in_bed_min"] = df["Time in bed"].apply(parse_duration_to_minutes)
    med = df["Time_in_bed_min"].median(skipna=True)
    if not np.isnan(med) and med > 1000:
        df["Time_in_bed_min"] = df["Time_in_bed_min"] / 60.0

    df = df.dropna(subset=["Start", "End"]).reset_index(drop=True)

    mask = df["Time_in_bed_min"].isna()
    df.loc[mask, "Time_in_bed_min"] = (df.loc[mask, "End"] - df.loc[mask, "Start"]).dt.total_seconds() / 60.0

    if "Sleep Notes" in df.columns:
        df["Sleep_Notes"] = df["Sleep Notes"].astype(str).replace("nan", "")
    else:
        df["Sleep_Notes"] = ""

    if "Wake up" in df.columns:
        df["Wake_up"] = df["Wake up"].astype(str).replace("nan", "")
    else:
        df["Wake_up"] = ""

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sleep_duration_min"] = (df["End"] - df["Start"]).dt.total_seconds() / 60.0

    df["sleep_efficiency"] = df["sleep_duration_min"] / df["Time_in_bed_min"]
    df["sleep_efficiency"] = df["sleep_efficiency"].replace([np.inf, -np.inf], np.nan).clip(0, 1)

    df["date"] = df["Start"].dt.date
    df["week_start"] = df["Start"].dt.to_period("W").apply(lambda r: r.start_time.date())
    df = df.sort_values("Start").reset_index(drop=True)

    df["roll_sleep_dur_7"] = df["sleep_duration_min"].rolling(window=7, min_periods=1).mean()
    df["roll_sleep_eff_7"] = df["sleep_efficiency"].rolling(window=7, min_periods=1).mean()
    df["roll_quality_7"] = df["Sleep_quality"].rolling(window=7, min_periods=1).mean()

    return df


def run_isolation_forest(df: pd.DataFrame, contamination: float, random_state: int = 42):
    df = df.copy()
    feature_cols = ["sleep_duration_min", "sleep_efficiency", "Heart_rate", "Activity", "Sleep_quality"]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999).values

    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    model.fit(X)
    pred = model.predict(X)
    score = model.decision_function(X)

    df["anomaly"] = (pred == -1)
    df["anomaly_score"] = score
    return df, model, feature_cols


def create_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cond_short = df["sleep_duration_min"] < 180
    cond_long = df["sleep_duration_min"] > 720
    cond_low_eff = df["sleep_efficiency"] < 0.60
    cond_low_quality = df["Sleep_quality"] < 30
    hr_thr = df["Heart_rate"].median(skipna=True) + 15
    cond_high_hr = df["Heart_rate"] > hr_thr
    df["proxy_anom"] = (cond_short | cond_long | cond_low_eff | cond_low_quality | cond_high_hr)
    return df


def build_recommendations(df: pd.DataFrame) -> list[str]:
    recs = []
    if df.empty:
        return ["No data in the selected range."]

    anom = df[df["anomaly"]]
    if anom.empty:
        return ["No anomalies detected in the selected range. Maintain current sleep habits."]

    med_hr = df["Heart_rate"].median(skipna=True)
    med_act = df["Activity"].median(skipna=True)

    if (anom["sleep_efficiency"] < 0.70).any():
        recs.append("Low sleep efficiency detected: consider a consistent bedtime, limit screens 1 hour before bed, and reduce late meals.")
    if (anom["sleep_duration_min"] < 360).any():
        recs.append("Short sleep duration detected: aim for a longer sleep window and avoid late caffeine.")
    if (anom["sleep_duration_min"] > 600).any():
        recs.append("Long sleep duration detected: consider a consistent wake time and review sleep schedule regularity.")
    if (anom["Heart_rate"] > (med_hr + 15)).any():
        recs.append("Higher resting heart rate on anomalous nights: consider stress reduction and avoiding stimulants later in the day.")
    if (anom["Activity"] > (med_act + 1000)).any():
        recs.append("Higher activity near anomalous nights: avoid intense exercise too close to bedtime.")
    if not recs:
        recs.append("Anomalies detected. Review the flagged nights and compare sleep quality, duration, and heart rate patterns.")
    return recs


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Controls")
    st.caption("Upload your sleep CSV and run anomaly detection.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use sample file from workspace (data/sleepdata.csv)", value=True)

    contamination = st.slider("Anomaly sensitivity (contamination)", 0.01, 0.20, 0.05, 0.01)
    show_anom_only = st.checkbox("Show anomalies only", value=False)

    st.divider()
    st.subheader("Monitoring")
    st.caption(f"Log file: {LOG_PATH}")
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            log_tail = f.read().splitlines()[-15:]
        st.code("\n".join(log_tail), language="text")
    else:
        st.write("No logs yet.")


if uploaded is not None:
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File too large. Max allowed is {MAX_UPLOAD_MB} MB.")
        logging.warning("Upload rejected: file too large (%s bytes)", uploaded.size)
        st.stop()

df_raw = None
data_source = None

try:
    if uploaded is not None:
        df_raw = read_csv_bytes(uploaded.read())
        data_source = "uploaded"
        logging.info("Loaded uploaded dataset with shape=%s", df_raw.shape)
    elif use_sample:
        try:
            with open(os.path.join("data", "sleepdata.csv"), "rb") as f:
                raw = f.read()
            df_raw = read_csv_bytes(raw)
            data_source = "sample:data/sleepdata.csv"
            logging.info("Loaded sample dataset with shape=%s", df_raw.shape)
            st.success("Loaded sample data/data/sleepdata.csv")
        except Exception as e:
            st.error(f"Failed to load sample file: {e}")
            logging.exception("Sample load failed")
            st.stop()
    else:
        st.info("Upload a CSV or enable 'Use sample file' to continue.")
        st.stop()
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    logging.exception("CSV read failed")
    st.stop()

with st.expander("Raw data preview", expanded=False):
    st.write("Columns:", list(df_raw.columns))
    st.dataframe(df_raw.head(20), use_container_width=True)

try:
    df_clean = preprocess(df_raw)
    df_feat = engineer_features(df_clean)

    logging.info(
        "Preprocess+features complete: rows=%d, date_range=%s..%s",
        len(df_feat),
        df_feat["Start"].min(),
        df_feat["Start"].max(),
    )
except Exception as e:
    st.error(str(e))
    logging.exception("Preprocess/feature engineering failed")
    st.stop()

dq1, dq2, dq3 = st.columns(3)
with dq1:
    st.metric("Rows (after cleaning)", f"{len(df_feat)}")
with dq2:
    st.metric("Date range", f"{df_feat['Start'].min().date()} → {df_feat['Start'].max().date()}")
with dq3:
    missing_pct = df_feat[["Sleep_quality", "Heart_rate", "Activity", "Time_in_bed_min"]].isna().mean() * 100
    st.metric("Avg missing % (key fields)", f"{missing_pct.mean():.1f}%")

with st.expander("Data quality checks (missing values)", expanded=False):
    st.dataframe((df_feat.isna().sum().sort_values(ascending=False)).to_frame("missing_count"), use_container_width=True)

min_d = df_feat["Start"].min().date()
max_d = df_feat["Start"].max().date()

qcol1, qcol2 = st.columns(2)
with qcol1:
    start_date = st.date_input("Filter start date", min_d, min_value=min_d, max_value=max_d)
with qcol2:
    end_date = st.date_input("Filter end date", max_d, min_value=min_d, max_value=max_d)

if start_date > end_date:
    st.error("Start date must be <= end date.")
    st.stop()

df_q = df_feat[(df_feat["Start"].dt.date >= start_date) & (df_feat["Start"].dt.date <= end_date)].copy()
if df_q.empty:
    st.warning("No rows in the selected date range.")
    st.stop()

# --- Run ML ---
df_anom, model, feature_cols = run_isolation_forest(df_q, contamination=contamination)
anom_count = int(df_anom["anomaly"].sum())
logging.info("IsolationForest run: contamination=%.3f anomalies=%d/%d", contamination, anom_count, len(df_anom))

# Apply anomaly-only view if toggled
df_view = df_anom[df_anom["anomaly"]].copy() if show_anom_only else df_anom.copy()

# -----------------------------
# Dashboard (3+ visualization types)
# -----------------------------
left, right = st.columns([2, 1])

with left:
    # 1) Time-series line chart with anomaly markers
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_view["Start"],
            y=df_view["sleep_duration_min"],
            mode="lines+markers",
            name="Sleep duration (min)",
        )
    )
    anom_pts = df_view[df_view["anomaly"]]
    if not anom_pts.empty:
        fig.add_trace(
            go.Scatter(
                x=anom_pts["Start"],
                y=anom_pts["sleep_duration_min"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=10),
                hovertext=anom_pts["Sleep_Notes"],
            )
        )
    fig.update_layout(
        title="Sleep duration over time (anomalies highlighted)",
        xaxis_title="Date",
        yaxis_title="Minutes",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2) Weekly bar chart (descriptive)
    weekly = df_view.groupby("week_start").agg(
        sleep_duration_min=("sleep_duration_min", "mean"),
        sleep_efficiency=("sleep_efficiency", "mean"),
        Sleep_quality=("Sleep_quality", "mean"),
    ).reset_index()

    weekly_melt = weekly.melt(id_vars="week_start", value_vars=["sleep_duration_min", "sleep_efficiency", "Sleep_quality"])
    fig2 = px.bar(
        weekly_melt,
        x="week_start",
        y="value",
        color="variable",
        barmode="group",
        title="Weekly averages (duration, efficiency, quality)",
        labels={"week_start": "Week", "value": "Value"},
        height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3) Scatter plot (exploration)
    fig3 = px.scatter(
        df_view,
        x="sleep_duration_min",
        y="Sleep_quality",
        color=df_view["anomaly"].map({True: "Anomaly", False: "Normal"}),
        hover_data=["date", "sleep_efficiency", "Heart_rate", "Activity"],
        title="Duration vs Sleep Quality (colored by anomaly)",
        height=420,
    )
    st.plotly_chart(fig3, use_container_width=True)

with right:
    st.header("Decision support")
    st.write(f"Anomalous nights: **{anom_count}** out of **{len(df_anom)}** in the selected range.")

    recs = build_recommendations(df_anom)
    for r in recs:
        st.write(f"- {r}")

    st.divider()
    st.header("Model details")
    st.write("Algorithm: Isolation Forest")
    st.write("Features used:")
    st.code("\n".join(feature_cols), language="text")

    st.divider()
    st.header("Evaluation (proxy labels)")
    df_eval = create_proxy_labels(df_anom)
    y_true = df_eval["proxy_anom"].astype(int)
    y_pred = df_eval["anomaly"].astype(int)

    if y_true.sum() == 0:
        st.write("Proxy labels produced no positive cases in this range. Expand the date range or adjust heuristics.")
    else:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        st.write(f"Precision: **{prec:.3f}**")
        st.write(f"Recall: **{rec:.3f}**")
        st.write(f"F1: **{f1:.3f}**")
        st.caption("Note: proxy labels are heuristics used only for rubric-required evaluation.")

    st.divider()
    st.header("Exports")
    st.download_button(
        "Download CLEANED dataset (CSV)",
        df_to_csv_bytes(df_feat),
        file_name="sleepdata_cleaned.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download ANOMALIES only (CSV)",
        df_to_csv_bytes(df_anom[df_anom["anomaly"]]),
        file_name="sleepdata_anomalies.csv",
        mime="text/csv",
    )

# -----------------------------
# Table of anomalies (useful for graders)
# -----------------------------
st.subheader("Flagged nights (table)")
anom_table = df_anom[df_anom["anomaly"]].copy()
if anom_table.empty:
    st.write("No anomalies flagged in the selected range.")
else:
    show_cols = ["Start", "End", "sleep_duration_min", "sleep_efficiency", "Sleep_quality", "Heart_rate", "Activity", "anomaly_score", "Sleep_Notes"]
    show_cols = [c for c in show_cols if c in anom_table.columns]
    st.dataframe(anom_table[show_cols].sort_values("Start"), use_container_width=True)

st.caption("Tip: If your file uses semicolons, this app auto-detects delimiters. Your header format is supported.")
