"""Accident Risk Detector - dashboard (Streamlit).

Sensor node readings (speed, shock, gyro, panic button) -> risk level: low / medium / high.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

_csvs = sorted(Path(__file__).parent.rglob("*.csv"))
_pick = [p for p in _csvs if p.name.lower() == "dataset.csv"] or _csvs
DATA_PATH = _pick[0] if _pick else Path(__file__).parent / "data" / "Dataset.csv"
FEATURES = ["speed_kmh", "shock_mps2", "gyro_dps", "button"]
TARGET = "risk_label"
LEVELS = ["low", "medium", "high"]

st.set_page_config(page_title="Accident Risk Detector", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        st.error(f"Dataset.csv is missing columns: {missing}")
        st.stop()
    return df


def simple_rule(r):
    """A hand-written 3-line rule, used as a baseline to compare with the model."""
    if r["button"] == 1 or r["shock_mps2"] >= 8:
        return "high"
    if r["speed_kmh"] < 70 and r["shock_mps2"] < 4 and r["gyro_dps"] < 150:
        return "low"
    return "medium"


@st.cache_resource
def train(df):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    report = pd.DataFrame(
        classification_report(y_te, pred, labels=LEVELS, output_dict=True,
                              zero_division=0)).T.loc[LEVELS + ["macro avg"]]
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=LEVELS),
                      index=[f"actual {l}" for l in LEVELS],
                      columns=[f"predicted {l}" for l in LEVELS])
    rule_pred = X_te.apply(simple_rule, axis=1)
    return model, {
        "accuracy": accuracy_score(y_te, pred),
        "rule_accuracy": accuracy_score(y_te, rule_pred),
        "report": report,
        "cm": cm,
        "n_test": len(y_te),
    }


def predict(model, row):
    probs = model.predict_proba(pd.DataFrame([row])[FEATURES])[0]
    probs = pd.Series(probs, index=model.classes_).reindex(LEVELS).fillna(0)
    return probs.idxmax(), probs


df = load_data()
model, m = train(df)

st.title("Accident Risk Detector")
st.caption("Reads vehicle sensor data (speed, shock, rotation, panic button) "
           "and classifies the risk as low, medium or high.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Data overview", "Model quality", "Predict risk", "Live sensor feed (simulated)"])

with tab1:
    counts = df[TARGET].value_counts().reindex(LEVELS)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Low", f"{counts['low']:,}")
    c3.metric("Medium", f"{counts['medium']:,}")
    c4.metric("High", f"{counts['high']:,}")
    st.subheader("Risk level counts")
    st.bar_chart(counts)
    st.subheader("Average sensor values per risk level")
    st.dataframe(df.groupby(TARGET)[FEATURES].mean().reindex(LEVELS).round(2))
    st.subheader("Sample rows")
    st.dataframe(df.head(50))
    st.info("Class imbalance: 'low' is only about 5% of the data, so the model "
            "sees few low-risk examples.")

with tab2:
    st.write(f"Tested on {m['n_test']} rows the model never saw during training.")
    c1, c2 = st.columns(2)
    c1.metric("Model accuracy", f"{m['accuracy']:.1%}")
    c2.metric("Simple 3-line rule accuracy", f"{m['rule_accuracy']:.1%}")
    st.warning("The model is only slightly better than a simple hand-written rule. "
               "This suggests the labels in this dataset were created by rules "
               "(simulated data). Results here do not prove real-world accuracy.")
    st.subheader("Per-level results")
    st.dataframe(m["report"].round(2))
    st.subheader("Confusion matrix")
    st.dataframe(m["cm"])
    st.subheader("Which sensors matter most")
    st.bar_chart(pd.Series(model.feature_importances_, index=FEATURES).sort_values())

with tab3:
    c1, c2 = st.columns(2)
    speed = c1.slider("Speed (km/h)", 0, 140, 60)
    shock = c1.slider("Shock (m/s²)", 0.0, 12.0, 2.0, 0.1)
    gyro = c2.slider("Rotation rate (deg/s)", 0, 250, 80)
    button = c2.checkbox("Panic button pressed")
    level, probs = predict(model, {"speed_kmh": speed, "shock_mps2": shock,
                                   "gyro_dps": gyro, "button": int(button)})
    st.metric("Predicted risk level", level.upper())
    st.bar_chart(probs)
    if level == "high":
        st.error("HIGH RISK: alert the driver / emergency contact.")
    elif level == "medium":
        st.warning("Medium risk: caution.")
    else:
        st.success("Low risk.")

with tab4:
    st.write("Simulates a sensor node sending readings. Replace `read_sensor()` "
             "with real data from an ESP32/Arduino later.")

    def read_sensor():
        rng = np.random.default_rng()
        return {"speed_kmh": int(rng.integers(0, 141)),
                "shock_mps2": round(float(rng.uniform(0, 12)), 2),
                "gyro_dps": round(float(rng.uniform(0, 250)), 2),
                "button": int(rng.random() < 0.1)}

    if "feed" not in st.session_state:
        st.session_state.feed = []
    if st.button("Get new sensor reading"):
        reading = read_sensor()
        level, probs = predict(model, reading)
        reading["risk"] = level
        reading["confidence"] = round(float(probs.max()), 2)
        st.session_state.feed.insert(0, reading)
    if st.session_state.feed:
        st.dataframe(pd.DataFrame(st.session_state.feed).head(15))
        if st.session_state.feed[0]["risk"] == "high":
            st.error("ALERT: high accident risk detected.")
