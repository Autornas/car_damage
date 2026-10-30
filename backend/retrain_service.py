"""
retrain_service.py

Manual retrain trigger: merges every registered source under data/raw
(including data/raw/corrections/ via dataset_merge.merge_datasets) and
launches the existing phase 2 training entry point,
car_damage.training.train_detector, as a detached
background subprocess -- never imported directly, so a training crash
can't take the API process down with it, and never awaited, so POST
/retrain returns immediately.

Status is answered purely by re-reading the run's log file (see
get_retrain_status), not from in-memory process state -- that stays
correct even across an API process restart (e.g. --reload picking up a
code change) mid-run, since only the log file's content, not this
module's globals, is required to know whether a run finished or failed.

Allowed imports: standard library, pyyaml (read/write the training
config). train_detector itself is invoked as a subprocess, never imported.
"""

import re
import subprocess
import sys
import threading
from pathlib import Path

import yaml

from dataset_merge import merge_datasets

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent

BASE_CONFIG_PATH = _PROJECT_ROOT / "configs" / "phase2_detector.yaml"
MERGED_DATA_DIR = _PROJECT_ROOT / "data" / "raw" / "_merged_for_retrain"
RETRAIN_DIR = _PROJECT_ROOT / "experiments" / "retrain_runs"
LOG_PATH = RETRAIN_DIR / "latest.log"
GENERATED_CONFIG_PATH = RETRAIN_DIR / "retrain_config.yaml"

_START_MARKER = "=== retrain started ==="
_END_MARKER_RE = re.compile(r"=== retrain (finished|failed) \(exit code (-?\d+)\) ===")
_LOSS_RE = re.compile(r"train_loss=([0-9.eE+-]+)")

_lock = threading.Lock()
_process = None  # subprocess.Popen handle for the currently/most-recently started run


def is_retrain_running():
    with _lock:
        return _process is not None and _process.poll() is None


def start_retrain():
    """
    Merges the current dataset, writes a training config pointing at the
    merge, and launches train_detector as a background subprocess.
    Returns the log path immediately -- does not wait for training.

    Raises RuntimeError if a retrain is already in progress.
    """
    global _process

    with _lock:
        if _process is not None and _process.poll() is None:
            raise RuntimeError("A retrain is already in progress")

        RETRAIN_DIR.mkdir(parents=True, exist_ok=True)
        merge_datasets(MERGED_DATA_DIR)

        with open(BASE_CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["data_dir"] = str(MERGED_DATA_DIR)
        with open(GENERATED_CONFIG_PATH, "w") as f:
            yaml.safe_dump(cfg, f)

        log_file = open(LOG_PATH, "w")
        log_file.write(_START_MARKER + "\n")
        log_file.flush()

        process = subprocess.Popen(
            [
                sys.executable, "-m", "car_damage.training.train_detector",
                "--config", str(GENERATED_CONFIG_PATH),
            ],
            cwd=str(_PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        _process = process

    def _wait_and_mark():
        returncode = process.wait()
        log_file.close()
        status_word = "finished" if returncode == 0 else "failed"
        with open(LOG_PATH, "a") as f:
            f.write(f"\n=== retrain {status_word} (exit code {returncode}) ===\n")

    threading.Thread(target=_wait_and_mark, daemon=True).start()
    return str(LOG_PATH)


def get_retrain_status():
    """
    Reads LOG_PATH and reports one of:
      - {"status": "idle", ...}     -- no retrain has ever been started
      - {"status": "running", ...}  -- started, no end marker yet
      - {"status": "finished", "final_loss": <float or None>, ...}
      - {"status": "failed", "error": <str>, ...}
    """
    if not LOG_PATH.is_file():
        return {"status": "idle", "final_loss": None, "error": None, "log_path": None}

    text = LOG_PATH.read_text(errors="replace")

    match = _END_MARKER_RE.search(text)
    if match is None:
        return {"status": "running", "final_loss": None, "error": None, "log_path": str(LOG_PATH)}

    word, exit_code = match.group(1), int(match.group(2))
    if word == "finished":
        losses = _LOSS_RE.findall(text)
        final_loss = float(losses[-1]) if losses else None
        return {"status": "finished", "final_loss": final_loss, "error": None, "log_path": str(LOG_PATH)}

    body = text[: match.start()]
    error_lines = [line for line in body.strip().splitlines() if line.strip()][-5:]
    error = "\n".join(error_lines) or f"Process exited with code {exit_code}"
    return {"status": "failed", "final_loss": None, "error": error, "log_path": str(LOG_PATH)}
