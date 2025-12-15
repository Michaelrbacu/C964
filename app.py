import io
import math
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Helpers ---
REQUIRED_COLS = ['Start', 'End', 'Sleep quality', 'Time in bed', 'Heart rate', 'Activity']
OPTIONAL_COLS = ['Sleep Notes']

@st.cache_data
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    # try common delimiters
    for sep in [',', ';', '\t']:
        try:
            return pd.read_csv(io.BytesIO(content), sep=sep)
        except Exception:
            continue
    # fallback
    return pd.read_csv(io.BytesIO(content), on_bad_lines='skip')


def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize names
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    lower_map = {c.lower(): c for c in df.columns}
    col_map = {}
    # Start/End
    if 'start' in lower_map:
        col_map['Start'] = lower_map['start']
    elif 'start time' in lower_map:
        col_map['Start'] = lower_map['start time']
    if 'end' in lower_map:
        col_map['End'] = lower_map['end']
    elif 'end time' in lower_map:
        col_map['End'] = lower_map['end time']
    # Sleep quality
    if 'sleep quality' in lower_map:
        col_map['Sleep quality'] = lower_map['sleep quality']
    elif 'sleep_quality' in lower_map:
        col_map['Sleep quality'] = lower_map['sleep_quality']
    elif 'sleep quality (%)' in lower_map:
        col_map['Sleep quality'] = lower_map['sleep quality (%)']
    # Time in bed
    if 'time in bed' in lower_map:
        col_map['Time in bed'] = lower_map['time in bed']
    elif 'time in bed (seconds)' in lower_map:
        col_map['Time in bed'] = lower_map['time in bed (seconds)']
    elif 'time in bed (s)' in lower_map:
        col_map['Time in bed'] = lower_map['time in bed (s)']
    # Heart rate
    if 'heart rate' in lower_map:
        col_map['Heart rate'] = lower_map['heart rate']
    elif 'heart rate (bpm)' in lower_map:
        col_map['Heart rate'] = lower_map['heart rate (bpm)']
    elif 'heart_rate' in lower_map:
        col_map['Heart rate'] = lower_map['heart_rate']
    # Activity / Steps
    if 'activity' in lower_map:
        col_map['Activity'] = lower_map['activity']
    elif 'steps' in lower_map:
        col_map['Activity'] = lower_map['steps']
    # Notes
    if 'sleep notes' in lower_map:
        col_map['Sleep Notes'] = lower_map['sleep notes']
    elif 'notes' in lower_map:
        col_map['Sleep Notes'] = lower_map['notes']

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
    # HH:MM or H:MM:SS
    if ':' in s:
        parts = s.split(':')
        try:
            parts = [float(p) for p in parts]
            if len(parts) == 2:
                return parts[0]*60 + parts[1]
            if len(parts) == 3:
                return parts[0]*60 + parts[1] + parts[2]/60
        except Exception:
            pass
    # numeric string
    try:
        return float(s)
    except Exception:
        return np.nan


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_and_map_columns(df)
    # Parse datetimes
    if 'Start' in df.columns:
        df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    if 'End' in df.columns:
        df['End'] = pd.to_datetime(df['End'], errors='coerce')
    # Time in bed
    if 'Time in bed' in df.columns:
        df['Time_in_bed_min'] = df['Time in bed'].apply(parse_duration_to_minutes)
        # if values appear to be seconds (large median), convert to minutes
        med = df['Time_in_bed_min'].median(skipna=True)
        if not np.isnan(med) and med > 1000:
            df['Time_in_bed_min'] = df['Time_in_bed_min'] / 60.0
    else:
        df['Time_in_bed_min'] = np.nan
    # Numeric fields
    if 'Sleep quality' in df.columns:
        df['Sleep_quality'] = pd.to_numeric(df['Sleep quality'], errors='coerce')
    else:
        df['Sleep_quality'] = np.nan
    if 'Heart rate' in df.columns:
        df['Heart_rate'] = pd.to_numeric(df['Heart rate'], errors='coerce')
    else:
        df['Heart_rate'] = np.nan
    if 'Activity' in df.columns:
        df['Activity'] = pd.to_numeric(df['Activity'], errors='coerce')
    else:
        df['Activity'] = np.nan
    # Drop rows without Start/End
    df = df.dropna(subset=['Start', 'End']).reset_index(drop=True)
    mask = df['Time_in_bed_min'].isna() & df['Start'].notna() & df['End'].notna()
    df.loc[mask, 'Time_in_bed_min'] = (df.loc[mask, 'End'] - df.loc[mask, 'Start']).dt.total_seconds() / 60.0
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sleep_duration_min'] = (df['End'] - df['Start']).dt.total_seconds() / 60.0
    df['sleep_efficiency'] = df['sleep_duration_min'] / df['Time_in_bed_min']
    df['sleep_efficiency'] = df['sleep_efficiency'].replace([np.inf, -np.inf], np.nan).clip(0, 1)
    df['heart_rate_avg'] = df['Heart_rate']
    df['activity_level'] = df['Activity']
    df['date'] = df['Start'].dt.date
    df['week'] = df['Start'].dt.to_period('W').apply(lambda r: r.start_time.date())
    df = df.sort_values('Start')
    df['roll_sleep_dur_7'] = df['sleep_duration_min'].rolling(window=7, min_periods=1).mean()
    df['roll_sleep_eff_7'] = df['sleep_efficiency'].rolling(window=7, min_periods=1).mean()
    return df


def run_isolation(df: pd.DataFrame, contamination: float=0.05, random_state=42):
    feats = ['sleep_duration_min', 'sleep_efficiency', 'heart_rate_avg', 'activity_level']
    X = df[feats].fillna(-999).values
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X)
    preds = iso.predict(X)
    df['anomaly'] = (preds == -1)
    return df, iso


def create_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cond_short = df['sleep_duration_min'] < 180
    cond_long = df['sleep_duration_min'] > 720
    cond_low_eff = df['sleep_efficiency'] < 0.6
    cond_hr = df['heart_rate_avg'] > (df['heart_rate_avg'].median() + 20)
    df['proxy_anom'] = (cond_short | cond_long | cond_low_eff | cond_hr)
    return df


# --- Streamlit UI ---
st.set_page_config(page_title='Sleep Anomaly Detection', layout='wide')
st.title('Sleep Pattern Anomaly Detection System')

uploaded = st.file_uploader('Upload a CSV file', type=['csv'])
# default to using the provided sample in the workspace
use_sample = st.checkbox('Use sample file from workspace (data/sleepdata.csv)', value=True)

df = None
if use_sample and uploaded is None:
    try:
        with open('data/sleepdata.csv', 'rb') as f:
            raw = f.read()
        df = read_csv_bytes(raw)
        st.success('Sample file loaded')
    except Exception as e:
        st.error('Could not load sample file from workspace: ' + str(e))

if uploaded is not None:
    raw = uploaded.read()
    df = read_csv_bytes(raw)
    st.success('File uploaded')

if df is None:
    st.info('Upload a CSV or check "Use sample file"')
    st.stop()

st.write('Raw columns:', list(df.columns))

# Preprocess
if st.button('Run Preprocessing'):
    df = preprocess(df)
    st.session_state['df'] = df
    st.success(f'Preprocessing complete. Rows: {len(df)}')

if 'df' not in st.session_state:
    st.warning('Run preprocessing to continue')
    st.stop()

# Feature engineering
if st.button('Engineer Features'):
    df_feat = engineer_features(st.session_state['df'])
    st.session_state['df_feat'] = df_feat
    st.success('Features engineered')

if 'df_feat' not in st.session_state:
    st.warning('Engineer features to continue')
    st.stop()

# Anomaly detection controls
contamination = st.slider('Anomaly contamination (sensitivity)', 0.001, 0.2, 0.05, 0.001)
if st.button('Run Isolation Forest'):
    df_anom, iso = run_isolation(st.session_state['df_feat'], contamination=contamination)
    st.session_state['df_anom'] = df_anom
    st.session_state['iso'] = iso
    st.success(f'Isolation Forest complete. Found {int(df_anom.anomaly.sum())} anomalies')

if 'df_anom' not in st.session_state:
    st.warning('Run Isolation Forest to continue')
    st.stop()

# Visuals
df_anom = st.session_state['df_anom']
col1, col2 = st.columns([2,1])
with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_anom['Start'], y=df_anom['sleep_duration_min'], mode='lines+markers', name='Duration'))
    anom = df_anom[df_anom['anomaly']]
    if len(anom):
        fig.add_trace(go.Scatter(x=anom['Start'], y=anom['sleep_duration_min'], mode='markers', marker=dict(color='red', size=10), name='Anomaly'))
    fig.update_layout(title='Sleep duration over time', xaxis_title='Date', yaxis_title='Duration (min)', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Weekly bar
    weekly = df_anom.groupby('week').agg({'sleep_duration_min':'mean','sleep_efficiency':'mean'}).reset_index()
    weekly_melt = weekly.melt(id_vars='week', value_vars=['sleep_duration_min','sleep_efficiency'])
    fig2 = px.bar(weekly_melt, x='week', y='value', color='variable', barmode='group', labels={'value':'Metric','week':'Week'})
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

    # Scatter
    fig3 = px.scatter(df_anom, x='sleep_duration_min', y='Sleep_quality', color=df_anom['anomaly'].map({True:'Anomaly', False:'Normal'}), hover_data=['date'])
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.header('Decision support')
    n_anom = int(df_anom['anomaly'].sum())
    total = len(df_anom)
    if n_anom == 0:
        st.write('No anomalies detected. Maintain current sleep habits.')
    else:
        st.write(f'Found {n_anom} anomalous nights out of {total}.')
        low_eff = df_anom[df_anom['anomaly'] & (df_anom['sleep_efficiency'] < 0.7)]
        if len(low_eff):
            st.write('- Low sleep efficiency on anomalies: consider improving sleep hygiene')
        high_hr = df_anom[df_anom['anomaly'] & (df_anom['heart_rate_avg'] > df_anom['heart_rate_avg'].median() + 15)]
        if len(high_hr):
            st.write('- High resting heart rate on anomalies: consider stress or caffeine causes')
        high_act = df_anom[df_anom['anomaly'] & (df_anom['activity_level'] > df_anom['activity_level'].median() + 1000)]
        if len(high_act):
            st.write('- High activity before sleep on anomalies: avoid intense late exercise')

    st.header('Evaluation')
    df_eval = create_proxy_labels(df_anom)
    y_true = df_eval['proxy_anom'].astype(int)
    y_pred = df_eval['anomaly'].astype(int)
    if y_true.sum() == 0:
        st.write('Proxy labels contain no positives; adjust heuristics or inspect data.')
    else:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        st.write(f'Precision: {prec:.3f}, Recall: {rec:.3f} (against proxy labels)')

    st.header('Export')
    if st.button('Download anomalies CSV'):
        out = df_anom[df_anom['anomaly']].to_csv(index=False).encode('utf-8')
        st.download_button('Download', out, file_name='anomalies.csv', mime='text/csv')

st.sidebar.header('About')
st.sidebar.write('Streamlit Sleep Pattern Anomaly Detection')

