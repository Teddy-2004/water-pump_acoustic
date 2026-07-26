# Acoustic Water Pump Diagnostics

**Predicting rural water pump functionality from sound — an extension of *Predicting Rural Water Point Functionality in Tanzania* (Intro to ML Summative) into a non-tabular (audio) modality.**

- 🎥 **Video Demo:** [https://youtu.be/TWrUYylLyXI]
- 🌐 **Live API URL:** https://water-pump-acoustic.onrender.com (Swagger docs: https://water-pump-acoustic.onrender.com/docs)
- 🖥️ **Live UI URL:** https://water-pump-acoustic-1.onrender.com
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

### 3.1 Training data and model

This repo already includes a sample of training/test data (`data/train/`, `data/test/`) and the
trained model (`models/pump_cnn_v1.h5`, `models/latest_metrics.json`) committed directly — no
download required to run the API or UI locally.

If you want to retrain from scratch on the full dataset:

1. Open `notebook/acoustic_pump_diagnostics.ipynb` in Colab (GPU runtime recommended).
2. Run all cells top to bottom. Section 1 downloads the full MIMII pump subset directly from
   Zenodo (~7.9 GB) and organizes it into `data/train/` and `data/test/`.
3. The training/evaluation cells save the trained model to `models/pump_cnn_v1.h5` and print
   accuracy/F1/precision/recall/ROC-AUC on the held-out test set. **Download the model
   immediately after it saves** — Colab wipes `/content` if the runtime disconnects.
4. Section 9 builds and downloads a small stratified sample (200 train + 100 test clips per
   class) if you want to refresh the committed sample data instead of the full dataset.

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

1. Push this repo to GitHub (model and sample data are committed directly, no Release asset or
   Git LFS needed).
2. On Render: **New Web Service** → connect the repo → Language: **Docker** → **Dockerfile
   Path**: `api/Dockerfile` → **Root Directory**: leave blank (build context must stay at repo
   root since the Dockerfile's `COPY` paths are relative to it).
3. Repeat for the UI: **Dockerfile Path**: `ui/Dockerfile` → add environment variable
   `API_BASE_URL` = your deployed API's Render URL.
4. Live URLs are linked at the top of this README.

> **Note on Render's free tier:** both services spin down after a period of inactivity. The
> first request after idling can take 30-90+ seconds to respond while the container cold-starts
> (importing TensorFlow and loading the model). This is expected, not a bug.

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

Tested locally via `docker compose up --build --scale api=N`, with nginx load-balancing across
however many API containers were running. Locust was run with **20 simulated users, spawn rate 5**,
for 60 seconds per configuration, against `http://localhost:8080` (the nginx endpoint) in every case.

| Containers | Users | Requests/sec | Median Latency (ms) | 95%ile Latency (ms) | Failures |
|---|---|---|---|---|---|
| 1 | 20 | 4.9  | 1900 | 4700 | 0 |
| 3 | 20 | 13.6 | 220  | 440  | 0 |
| 5 | 20 | 14.6 | 190  | 250  | 0 |

**Observations:**

Going from 1 to 3 containers gave a dramatic improvement: throughput nearly tripled (4.9 → 13.6
req/s) and median latency dropped by ~8.6x (1900ms → 220ms). Each `/predict` request runs a full
TensorFlow inference (spectrogram computation + CNN forward pass) on a container capped at 1 CPU /
1GB RAM, so with only one container, requests were queuing up behind each other -- adding
containers let nginx distribute that inference load in parallel instead of serializing it.

Going from 3 to 5 containers gave only a small further improvement (13.6 → 14.6 req/s, ~7%), with
the clearest gain being a much tighter 95%ile (440ms → 250ms) rather than a big throughput jump.
This is a diminishing-returns pattern rather than a linear continuation, most likely because the
test load (20 concurrent simulated users) wasn't generating enough simultaneous demand to fully
saturate 5 parallel containers -- 3 containers were already close to satisfying what 20 users could
throw at the API, so the two extra containers mostly reduced tail latency rather than adding raw
throughput. A test with more simulated users would likely show 5 containers pulling further ahead.

All three configurations had **0 failures**, confirming the API remains correct under concurrent
load (this was not always true during development -- an earlier version had a Keras version
mismatch between the local training environment and the Docker image that caused every `/predict`
request to fail under load with a 500 error; fixing that version pin was a precondition for these
results being meaningful).

---

## 5. Model Evaluation Summary

The final CNN (trained on log-mel spectrograms of the MIMII pump subset, 0dB SNR) achieved on the
held-out test set (842 clips, stratified 750 normal / 92 abnormal):

| Metric | Value |
|---|---|
| Accuracy | 0.9359 |
| F1 (abnormal class) | 0.7589 |
| Precision (abnormal) | 0.6439 |
| Recall (abnormal) | 0.9239 |
| ROC-AUC | 0.969 |

```
              precision    recall  f1-score   support

      normal       0.99      0.94      0.96       750
    abnormal       0.64      0.92      0.76        92

    accuracy                           0.94       842
   macro avg       0.82      0.93      0.86       842
weighted avg       0.95      0.94      0.94       842
```

Train-validation gap was small (accuracy gap 0.005, F1 gap 0.0105), indicating the regularisation
(dropout, batch normalization, early stopping) controlled overfitting effectively rather than the
model simply memorising the training set.

The headline finding: the model catches 92% of genuinely faulty pumps (recall), at the cost of a
lower precision (64%) -- i.e. some normal pumps get flagged as needing inspection. For a
maintenance-prioritisation tool this is the right tradeoff direction, since a false alarm only
costs an unnecessary inspection, while a missed fault leaves a pump broken. This is also a
noticeably cleaner separation than the original tabular "needs repair" class from the summative
(F1 ≈0.45 there), suggesting acoustic signal carries more predictive signal for this particular
question than the administrative metadata did.

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
