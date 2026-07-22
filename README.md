# Acoustic Water Pump Diagnostics

**Predicting rural water pump functionality from sound — an extension of *Predicting Rural Water Point Functionality in Tanzania* (Intro to ML Summative) into a non-tabular (audio) modality.**

- 🎥 **Video Demo:** _[add your YouTube link here]_
- 🌐 **Live API URL:** _[add your Render deployment URL here]_
- 🖥️ **Live UI URL:** _[add your Render/Streamlit Cloud URL here]_
- 📓 **Original tabular summative:** https://github.com/Teddy-2004/water-pump_project

---

## 1. Project Description

The Introduction to Machine Learning summative asked whether a rural water point's functional
status (functional / needs repair / non-functional) could be predicted from **administrative
metadata**, without a site visit — useful for prioritising limited inspection resources across
tens of thousands of water points in Tanzania.

This project extends the same underlying question — *is the pump working?* — to a **non-tabular**
signal: a **10-second audio recording** of the pump in operation. Instead of relying on
administrative records (which the original report found were often missing or unreliable —
`scheme_name` was missing in 48.5% of rows), a field agent or even an ordinary phone microphone
could capture a short clip and get an instant diagnosis.

**Dataset:** [MIMII Dataset](https://zenodo.org/record/3384388) (Purohit et al., 2019) — pump
subset, real recordings of industrial pumps (used for water intake/discharge) labeled
`normal` / `abnormal`, at 0dB SNR (moderate background factory noise).

**Model:** A compact CNN trained on log-mel spectrograms of each clip, with class weighting
(abnormal is the minority class — the same imbalance pattern as the "needs repair" class in the
original tabular study), batch normalization, dropout, early stopping, and LR scheduling.

**Pipeline covered:**
1. Data acquisition (MIMII pump subset, downloaded from Zenodo)
2. Data preprocessing (waveform → log-mel spectrogram, shared by notebook/API/retraining)
3. Model creation (CNN) with a baseline comparison model
4. Model evaluation (accuracy, F1, precision, recall, ROC-AUC, confusion matrix)
5. Model retraining, triggerable via API/UI on newly uploaded labeled clips
6. FastAPI service (`/predict`, `/retrain`, `/uptime`, `/metrics`, `/model-info`)
7. Streamlit UI (uptime, visualizations, predict, retrain)
8. Dockerized deployment + Locust flood-request load testing at 1 vs. N containers

---

## 2. Repository Structure

```
water-pump-acoustic/
│
├── README.md
├── requirements.txt
├── docker-compose.yml
├── nginx.conf
│
├── notebook/
│   └── acoustic_pump_diagnostics.ipynb    # data acquisition -> preprocessing -> model -> eval
│
├── src/
│   ├── preprocessing.py                    # audio -> log-mel spectrogram (shared everywhere)
│   ├── model.py                            # CNN architecture, train/evaluate/save/load
│   └── prediction.py                       # single-clip inference wrapper
│
├── api/
│   ├── main.py                             # FastAPI app: /predict /retrain /uptime /metrics
│   ├── retrain.py                          # background retraining job
│   └── Dockerfile
│
├── ui/
│   ├── app.py                              # Streamlit dashboard
│   ├── Dockerfile
│   └── requirements.txt
│
├── locust/
│   └── locustfile.py                       # flood-request simulation
│
├── data/
│   ├── train/{normal,abnormal}/            # populated by notebook Section 1
│   └── test/{normal,abnormal}/
│
└── models/
    └── pump_cnn_v1.h5                      # produced by the notebook
```

---

## 3. Setup Instructions

### 3.1 Train the model (Google Colab / Kaggle, GPU runtime recommended)

1. Open `notebook/acoustic_pump_diagnostics.ipynb` in Colab.
2. Run all cells top to bottom. Section 1 downloads the MIMII pump subset directly from Zenodo
   (~7.9 GB) and organizes it into `data/train/` and `data/test/`.
3. The training/evaluation cells save the trained model to `models/pump_cnn_v1.h5` and print
   your accuracy/F1/precision/recall/ROC-AUC on the held-out test set.
4. **Download the model immediately** after it saves (Colab wipes `/content` if the runtime
   disconnects) — either via the Files panel or `google.colab.files.download(...)`.
5. **Don't try to download the full ~8GB dataset.** Section 9 of the notebook builds and
   downloads a small stratified sample instead (200 train + 100 test clips per class,
   a few hundred MB) — plenty for local API testing, Docker builds, and the retraining
   pipeline's existing-data sampling step.
6. Place both downloads in your local repo clone, matching the structure above.

> Model files are excluded from git via `.gitignore` (they can be large). Use Git LFS, or attach
> the `.h5` file as a GitHub Release asset, and note that in your submission.

### 3.2 Run the API locally

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger UI.

### 3.3 Run the UI locally

```bash
pip install -r ui/requirements.txt
export API_BASE_URL=http://localhost:8000
streamlit run ui/app.py
```

### 3.4 Run with Docker Compose (API + nginx load balancer)

```bash
docker compose up --build --scale api=1
```

Point the UI's `API_BASE_URL` at `http://localhost:8080` (the nginx port) when using Compose.

### 3.5 Deploy to Render

1. Push this repo to GitHub (model file via Release asset or Git LFS).
2. On Render: **New Web Service** → connect the repo → set **Dockerfile path** to `api/Dockerfile`.
3. Repeat for the UI: **New Web Service** → **Dockerfile path** `ui/Dockerfile` → set the
   `API_BASE_URL` environment variable to your deployed API's Render URL.
4. _[Fill in your live URLs at the top of this README once deployed.]_

### 3.6 Flood-request simulation (Locust)

```bash
pip install locust
# Against local docker-compose (nginx on :8080):
locust -f locust/locustfile.py --host http://localhost:8080
# Against your live Render deployment:
locust -f locust/locustfile.py --host https://<your-render-url>
```

Open `http://localhost:8089`, set the number of users and spawn rate, and start the test.
Repeat the same test for `--scale api=1`, `--scale api=3`, `--scale api=5` and record the
results below.

---

## 4. Results from Flood Request Simulation

_[Fill in after running the Locust tests at each container count. Example table below.]_

| Containers | Users | Requests/sec | Median Latency (ms) | p95 Latency (ms) | Failures |
|---|---|---|---|---|---|
| 1 | 50 | | | | |
| 3 | 50 | | | | |
| 5 | 50 | | | | |

**Observations:** _[e.g. "Median latency dropped from Xms to Yms going from 1 to 3 containers,
with diminishing returns at 5 containers because the nginx VM itself became the bottleneck..."]_

---

## 5. Model Evaluation Summary

_[Fill in with your actual notebook results once trained on real MIMII data — accuracy, F1,
precision, recall, ROC-AUC on the held-out test set, plus a one-paragraph interpretation,
mirroring Section 5.3 of the original water pump report.]_

---

## 6. Retraining

New labeled clips can be added via:
- **UI:** "Retrain" tab → upload a `.zip` with `normal/` and/or `abnormal/` folders.
- **API directly:** `POST /retrain` with a multipart `.zip` file.

The retraining job (`api/retrain.py`) fine-tunes the currently deployed model on the new data
mixed with a sample of existing training data (to avoid catastrophic forgetting on small
uploads), evaluates the result on the held-out test set, and **only promotes the new model to
production if it does not underperform the current one** — retraining should never silently
make production worse.

---

## References

- Purohit, H., Tanabe, R., Ichige, K., Endo, T., Nikaido, Y., Suefusa, K., & Kawaguchi, Y. (2019).
  *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and
  Inspection.* arXiv:1909.09347.
- Godebo, T. T. (2026). *Predicting Rural Water Point Functionality in Tanzania: A Comparative
  Machine Learning and Deep Learning Approach.* African Leadership University.
