"""
locust/locustfile.py
---------------------
Flood-request simulation against the deployed (or docker-compose'd) API.

Usage:
    locust -f locust/locustfile.py --host http://localhost:8080

Then open http://localhost:8089, set number of users / spawn rate, and start
the test. Run this once per container-count configuration (1, 3, 5 containers,
via `docker compose up --scale api=N`) and record the requests/sec, median
latency, and p95 latency Locust reports for each -- these are the numbers
that go in the README's "Results from Flood Request Simulation" section.

A short sample .wav (~1s of silence/noise) is generated once and reused for
every request, so the test measures API/model throughput, not file-transfer
size effects.
"""

import io
import os
import numpy as np
import soundfile as sf
from locust import HttpUser, task, between

# Generate a small reusable test clip once at import time.
_SAMPLE_RATE = 16000
_DURATION_S = 10  # match MIMII clip length so preprocessing behaves realistically
_rng = np.random.default_rng(42)
_dummy_audio = (_rng.standard_normal(_SAMPLE_RATE * _DURATION_S) * 0.05).astype(np.float32)

_buf = io.BytesIO()
sf.write(_buf, _dummy_audio, _SAMPLE_RATE, format="WAV")
DUMMY_WAV_BYTES = _buf.getvalue()


class PumpDiagnosticsUser(HttpUser):
    wait_time = between(0.5, 2.0)  # simulate realistic user pacing between requests

    @task(9)
    def predict(self):
        files = {"file": ("test_clip.wav", DUMMY_WAV_BYTES, "audio/wav")}
        self.client.post("/predict", files=files, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
