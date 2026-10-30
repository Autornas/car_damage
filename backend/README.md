# Car Damage Detector — local web app

A local-only FastAPI backend + plain HTML/JS frontend for trying out the
phase 2 detector interactively. **Local development and testing only** —
no auth, no Docker, no deployment config. Two commands, two terminals.

## 1. Backend

From `car_damage/` (the project root, not `backend/`), make sure the
`car_damage` package itself is installed once — only needed the first
time, or after a fresh clone/venv:

```bash
pip install -e .
```

Then, from `car_damage/backend/`:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check it started correctly:

```bash
curl http://localhost:8000/health
```

- `{"status":"ok","model_loaded":true}` — checkpoint loaded, `/predict` is ready.
- `{"status":"ok","model_loaded":false}` — the server is up, but no checkpoint
  was found/loadable. `/predict` returns a clear 503 with the reason instead
  of a stack trace until this is fixed.

By default the backend looks for the pilot smoke-test checkpoint at
`checkpoints/phase2_detector/detector_smoke_test.pt` relative to the
project root (resolved from `backend/`'s own location, so it works
regardless of your current directory). Point it at a different checkpoint
with an environment variable:

```bash
CAR_DAMAGE_CHECKPOINT=/path/to/your_checkpoint.pt uvicorn main:app --reload --port 8000
```

When the manual `/retrain` endpoint is triggered, training merges every
registered dataset under `data/raw/`, including imported sources and user
corrections. New COCO-format CarDD archives can be converted, deduplicated,
validated, and registered from the project root with:

```powershell
python scripts/import_cardd.py
```

The importer reads the archives in `new_data/`, keeps CarDD's original
annotations when the Roboflow v3 export contains a duplicate photo, and
adds only non-duplicate photos. It does not start training automatically;
review the dataset summary before triggering `/retrain`. Check the
upstream terms for both archives before redistributing data or using it
commercially.

## 2. Frontend

In a second terminal, from `car_damage/frontend/`:

```bash
cd frontend
python -m http.server 5500
```

Open http://localhost:5500 in a browser, upload a photo, adjust the
confidence threshold if you like, and click "Detect damage". The backend
must already be running on port 8000 (step 1) — the page calls
`http://localhost:8000/predict` directly; nothing is proxied.

## Notes

- The shipped checkpoint was trained on the 60-image pilot batch as a
  pipeline smoke test, not a real training run (see
  `data/raw/phase2_pilot/README.md`). Expect noisy/low-quality detections
  — that's expected, not a bug in this app.
- CORS is open for `http://localhost` and `http://127.0.0.1` on any port,
  so serving the frontend on a port other than 5500 also works with no
  extra setup.
- No auth, no Docker/deployment config, no build tooling on the frontend —
  by design. This is a local single-user dev tool.
