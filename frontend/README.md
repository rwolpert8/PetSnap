# PetSnap web demo

The responsive interface is plain HTML, CSS, and JavaScript. FastAPI serves it
from the same origin as inference; no Node build, frontend API key, database,
or third-party image upload service is needed.

## Run locally

From the repository root, with Python 3.10 or newer:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m uvicorn api.api_server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Keep both `models/best_model.pth` and
`models/classes.json` available. No `Images/` folder is needed at runtime.
Weights are intentionally excluded from Git; copy your trained checkpoint when
setting up a fresh checkout. Do not reorder labels or pair them with a model
trained using another class ordering.

The local demo exposes `POST /api/demo/predict` without a key. Legacy inference
routes require `DOG_CLASSIFIER_API_KEY` to be explicitly set on the server,
and the caller supplies `X-API-Key`. The web page never receives this key.
`GET /health` is the readiness endpoint; `/` now serves the interface.

Upload processing supports JPEG, PNG, and WebP, at most 10 MiB and 20 million
pixels. EXIF rotation is honored. Photos are not retained by the application;
the multipart parser can temporarily spool uploads to disk and closes them
after reading. Inference is serialized per process and returns a retryable 503
while busy. Use one worker initially to avoid duplicating model memory.

## Checks

```sh
python -m pip install httpx
python -m unittest discover -s tests -v
node --check frontend/app.js
```

Integration tests load the real local checkpoint and verify three sample
predictions, malformed uploads, limits, authentication, and busy responses.
They do not require the training dataset or retrieve any external breed data.

## Configuration

- `PETSNAP_MODEL_PATH`: optional absolute checkpoint path.
- `PETSNAP_CLASSES_PATH`: optional absolute ordered labels path.
- `PETSNAP_CORS_ORIGINS`: comma-separated allowed origins for separate clients.
  The same-origin demo needs no CORS configuration.
- `PORT`: used when launching `python api/api_server.py` directly.

Before public hosting, configure request-body limits at the proxy (including
multipart overhead), per-client rate limits, HTTPS, and resource monitoring.
The application checks image bytes after multipart parsing; it is not a
replacement for an ingress request-body limit. This change builds and tests
the local demo; it does not publish a service.

## Photo credits and limitations

The three bundled examples are from the Stanford Dogs dataset already present
in this project. Original dataset filenames:

- `golden.jpg`: `Golden Retriever/n02099601_1010.jpg`
- `beagle.jpg`: `Beagle/n02088364_10108.jpg`
- `husky.jpg`: `Siberian Husky/n02110185_10047.jpg`

Dataset: https://vision.stanford.edu/aditya86/ImageNetDogs/

These examples are for demonstration, not an independent accuracy evaluation.
Predictions describe visual similarity among 120 classes, not mixed-breed
ancestry. There is no non-dog rejection model. Fonts load from Google Fonts
with system-font fallbacks; images, icons, scripts, and inference are local.
