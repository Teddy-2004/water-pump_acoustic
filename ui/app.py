"""
ui/app.py
---------
Streamlit dashboard for the Acoustic Pump Diagnostics project.

Covers the three required UI functionalities:
  1. Model uptime
  2. Data visualizations
  3. Access to predict / bulk-upload / trigger-retraining

Run:
    streamlit run ui/app.py

Set API_BASE_URL to your deployed Render URL (or http://localhost:8080 for
the local docker-compose / nginx setup) via an environment variable or the
sidebar input.
"""

import os
import io
import time
import zipfile

import requests
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Acoustic Pump Diagnostics", layout="wide")

DEFAULT_API_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.sidebar.title("⚙️ Settings")
api_url = st.sidebar.text_input("API base URL", value=DEFAULT_API_URL)

st.title("💧 Acoustic Water Pump Diagnostics")
st.caption(
    "Extension of *Predicting Rural Water Point Functionality in Tanzania* — "
    "diagnosing pump status from a 10-second audio recording instead of administrative metadata."
)

tab_overview, tab_predict, tab_retrain, tab_viz = st.tabs(
    ["📈 Model Status", "🔍 Predict", "🔁 Retrain", "📊 Data Visualizations"]
)

# ------------------------------------------------------------------
with tab_overview:
    st.subheader("Model Uptime & Status")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Refresh uptime"):
            st.session_state["_refresh"] = time.time()
        try:
            r = requests.get(f"{api_url}/uptime", timeout=5)
            r.raise_for_status()
            data = r.json()
            mins = data["uptime_seconds"] / 60
            st.metric("API Uptime", f"{mins:.1f} min")
            st.caption(f"Started at: {data['started_at']}")
        except Exception as e:
            st.error(f"Could not reach API at {api_url}: {e}")

    with col2:
        try:
            r = requests.get(f"{api_url}/model-info", timeout=5)
            r.raise_for_status()
            info = r.json()
            st.metric("Last Trained", info["last_trained"][:19].replace("T", " "))
            st.write("**Retrain status:**", info["retrain_status"]["state"])
            if info["retrain_status"].get("detail"):
                st.caption(info["retrain_status"]["detail"])
        except Exception as e:
            st.error(f"Could not fetch model info: {e}")

    st.subheader("Latest Evaluation Metrics")
    try:
        r = requests.get(f"{api_url}/metrics", timeout=5)
        r.raise_for_status()
        m = r.json()
        if "detail" in m:
            st.info(m["detail"])
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{m['accuracy']:.3f}")
            c2.metric("F1", f"{m['f1']:.3f}")
            c3.metric("Precision", f"{m['precision']:.3f}")
            c4.metric("Recall", f"{m['recall']:.3f}")
            c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    except Exception as e:
        st.error(f"Could not fetch metrics: {e}")

# ------------------------------------------------------------------
with tab_predict:
    st.subheader("Upload one pump audio clip (.wav) for diagnosis")
    uploaded = st.file_uploader("Choose a .wav file", type=["wav"])

    if uploaded is not None:
        st.audio(uploaded)
        if st.button("Run Prediction"):
            with st.spinner("Classifying..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), "audio/wav")}
                    r = requests.post(f"{api_url}/predict", files=files, timeout=30)
                    r.raise_for_status()
                    result = r.json()

                    if result["label"] == "abnormal":
                        st.error(f"⚠️ Prediction: **ABNORMAL** (confidence {result['confidence']:.1%})")
                    else:
                        st.success(f"✅ Prediction: **NORMAL** (confidence {result['confidence']:.1%})")

                    st.caption(f"P(abnormal) = {result['probability_abnormal']:.4f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------------
with tab_retrain:
    st.subheader("Bulk upload new labeled clips to trigger retraining")
    st.markdown(
        "Upload a **.zip** containing `normal/` and/or `abnormal/` folders of `.wav` files. "
        "The model will be fine-tuned on this data in the background and only promoted to "
        "production if it doesn't underperform the current model on the held-out test set."
    )

    zip_file = st.file_uploader("Choose a .zip file", type=["zip"], key="retrain_zip")

    if zip_file is not None:
        try:
            with zipfile.ZipFile(io.BytesIO(zip_file.getvalue())) as z:
                names = z.namelist()
            n_wav = sum(1 for n in names if n.lower().endswith(".wav"))
            st.write(f"Found **{n_wav}** .wav files in the uploaded zip.")
        except Exception as e:
            st.error(f"Could not read zip: {e}")

        if st.button("🚀 Trigger Retraining"):
            with st.spinner("Uploading and starting retraining job..."):
                try:
                    files = {"file": (zip_file.name, zip_file.getvalue(), "application/zip")}
                    r = requests.post(f"{api_url}/retrain", files=files, timeout=60)
                    r.raise_for_status()
                    st.success(r.json()["message"])
                    st.info("Poll the **Model Status** tab to see when retraining finishes.")
                except Exception as e:
                    st.error(f"Retraining request failed: {e}")

# ------------------------------------------------------------------
with tab_viz:
    st.subheader("Dataset Insights")
    st.markdown(
        "These charts summarize the training data characteristics discussed in the notebook's "
        "EDA section (class balance, spectral differences, and clip duration)."
    )

    try:
        r = requests.get(f"{api_url}/metrics", timeout=5)
        m = r.json()
        if "confusion_matrix" in m:
            import numpy as np
            cm = np.array(m["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "abnormal"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "abnormal"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title("Latest Test-Set Confusion Matrix")
            st.pyplot(fig)
        else:
            st.info("No confusion matrix available yet -- run the notebook or a retrain first.")
    except Exception as e:
        st.error(f"Could not load visualizations: {e}")

    st.markdown(
        "See `notebook/acoustic_pump_diagnostics.ipynb` Section 2 for the full EDA: "
        "class distribution, normal-vs-abnormal waveform/spectrogram comparison, "
        "and clip duration distribution."
    )
