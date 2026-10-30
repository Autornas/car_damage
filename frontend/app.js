// Plain JS, no build step. Talks to the FastAPI backend at API_BASE.
const API_BASE = "http://localhost:8000";
const API_URL = `${API_BASE}/predict`;
const CORRECT_URL = `${API_BASE}/correct`;
const CORRECTIONS_COUNT_URL = `${API_BASE}/corrections/count`;
const RETRAIN_URL = `${API_BASE}/retrain`;
const RETRAIN_STATUS_URL = `${API_BASE}/retrain/status`;
const QUEUE_NEXT_URL = `${API_BASE}/queue/next`;
const QUEUE_STATS_URL = `${API_BASE}/queue/stats`;

const MAX_CANVAS_WIDTH = 800;
const HANDLE_SIZE = 8; // resize handle square, canvas px
const DELETE_RADIUS = 8; // delete "x" control circle, canvas px
const MIN_BOX_SIZE = 6; // smallest box edge allowed while drawing/resizing, canvas px
const RETRAIN_POLL_MS = 4000;

const imageInput = document.getElementById("imageInput");
const detectBtn = document.getElementById("detectBtn");
const thresholdInput = document.getElementById("thresholdInput");
const thresholdValue = document.getElementById("thresholdValue");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const boxListEl = document.getElementById("boxList");

const drawModeBtn = document.getElementById("drawModeBtn");
const saveCorrectionBtn = document.getElementById("saveCorrectionBtn");
const saveAndNextBtn = document.getElementById("saveAndNextBtn");
const correctionStatusEl = document.getElementById("correctionStatus");
const correctionCountEl = document.getElementById("correctionCount");
const retrainBtn = document.getElementById("retrainBtn");
const retrainStatusEl = document.getElementById("retrainStatus");

const tabSingleBtn = document.getElementById("tabSingleBtn");
const tabQueueBtn = document.getElementById("tabQueueBtn");
const singleUploadPanel = document.getElementById("singleUploadPanel");
const reviewQueuePanel = document.getElementById("reviewQueuePanel");
const queueProgressEl = document.getElementById("queueProgress");
const queueProgressFillEl = document.getElementById("queueProgressFill");
const queueUncertaintyEl = document.getElementById("queueUncertainty");
const queueMessageEl = document.getElementById("queueMessage");

let currentImage = null; // HTMLImageElement for the currently loaded file
let currentFile = null; // File (single-upload) or Blob (review queue) to POST to /correct
let currentFilename = null; // filename to send alongside currentFile -- File.name doesn't apply to a Blob
let currentObjectUrl = null; // revoked/replaced each time a queue image is loaded, to avoid leaking blob URLs
let imgWidth = 0; // natural pixel dimensions of currentImage
let imgHeight = 0;

// The live, editable box state -- starts as the model's predictions once
// a detection comes back, mutates as boxes are moved/resized/deleted/
// drawn. This (not the original /predict response) is what /correct saves.
// Each entry: {x_min, y_min, x_max, y_max, score?} in ORIGINAL image
// pixel coordinates. `score` is present only for untouched model
// predictions and is purely informational -- it's stripped before saving.
let currentBoxes = [];
let editingEnabled = false; // true once a prediction has come back for the current image
let drawMode = false;
let interaction = null; // in-progress mouse interaction, see canvas handlers below

let retrainPollHandle = null;

// ---- File selection ----

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  currentFile = file;
  currentFilename = file.name;

  resetEditingState();
  setStatus("", "");
  boxListEl.innerHTML = "";
  correctionStatusEl.textContent = "";

  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      currentImage = img;
      imgWidth = img.naturalWidth;
      imgHeight = img.naturalHeight;
      drawImageOnly();
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
});

function resetEditingState() {
  currentBoxes = [];
  editingEnabled = false;
  drawMode = false;
  interaction = null;
  drawModeBtn.textContent = "Draw new box";
  drawModeBtn.classList.remove("active");
  canvas.classList.remove("draw-mode");
  saveCorrectionBtn.disabled = true;
  saveAndNextBtn.disabled = true;
}

// ---- Threshold + detect (unchanged flow) ----

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
});

thresholdInput.addEventListener("change", () => {
  // Only re-run if we already have an image and a prior/current result to refresh.
  if (currentFile) {
    runDetection();
  }
});

detectBtn.addEventListener("click", () => {
  if (!currentFile) {
    setStatus("Choose an image first.", "error");
    return;
  }
  runDetection();
});

function drawImageOnly() {
  const scale = Math.min(1, MAX_CANVAS_WIDTH / currentImage.naturalWidth);
  canvas.width = currentImage.naturalWidth * scale;
  canvas.height = currentImage.naturalHeight * scale;
  ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
}

function setStatus(message, kind) {
  statusEl.textContent = message;
  statusEl.className = kind ? `status ${kind}` : "status";
}

async function runDetection() {
  setStatus("Running detection...", "loading");
  detectBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", currentFile);

  const threshold = thresholdInput.value;
  const url = `${API_URL}?confidence_threshold=${encodeURIComponent(threshold)}`;

  let response;
  try {
    response = await fetch(url, { method: "POST", body: formData });
  } catch (err) {
    // fetch() only throws on network-level failure (backend not running,
    // CORS block, DNS, etc.) -- distinct from a non-2xx HTTP response.
    setStatus(
      "Could not reach the backend. Is it running on http://localhost:8000?",
      "error"
    );
    detectBtn.disabled = false;
    return;
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errBody = await response.json();
      detail = errBody.detail || detail;
    } catch (_) {
      // response body wasn't JSON; fall back to statusText already set above
    }
    setStatus(`Request failed (${response.status}): ${detail}`, "error");
    detectBtn.disabled = false;
    return;
  }

  const data = await response.json();
  detectBtn.disabled = false;

  imgWidth = data.image_width;
  imgHeight = data.image_height;
  currentBoxes = data.boxes.map((b) => ({
    x_min: b.x_min,
    y_min: b.y_min,
    x_max: b.x_max,
    y_max: b.y_max,
    score: b.score,
  }));
  editingEnabled = true;

  render();
  renderBoxList(currentBoxes);
  updateSaveButtonState();

  if (currentBoxes.length === 0) {
    setStatus("No damage detected above the current confidence threshold.", "empty");
  } else {
    setStatus(`Found ${currentBoxes.length} detection(s).`, "success");
  }
}

// ---- Coordinate conversion between original-image px and canvas px ----

function imgToCanvas(pt) {
  return {
    x: (pt.x * canvas.width) / imgWidth,
    y: (pt.y * canvas.height) / imgHeight,
  };
}

function canvasToImg(pt) {
  return {
    x: (pt.x * imgWidth) / canvas.width,
    y: (pt.y * imgHeight) / canvas.height,
  };
}

function getCanvasPos(evt) {
  const rect = canvas.getBoundingClientRect();
  // Accounts for the canvas's CSS display size differing from its
  // backing pixel size (e.g. max-width:100% shrinking it on a narrow screen).
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (evt.clientX - rect.left) * scaleX,
    y: (evt.clientY - rect.top) * scaleY,
  };
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

// ---- Rendering ----

function render() {
  if (!currentImage) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

  currentBoxes.forEach((box) => drawBox(box));

  if (interaction && interaction.type === "draw") {
    const x = Math.min(interaction.startX, interaction.currentX);
    const y = Math.min(interaction.startY, interaction.currentY);
    const w = Math.abs(interaction.currentX - interaction.startX);
    const h = Math.abs(interaction.currentY - interaction.startY);
    ctx.save();
    ctx.setLineDash([5, 3]);
    ctx.strokeStyle = "#1a7f37";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }
}

function drawBox(box) {
  const p1 = imgToCanvas({ x: box.x_min, y: box.y_min });
  const p2 = imgToCanvas({ x: box.x_max, y: box.y_max });
  const x = p1.x;
  const y = p1.y;
  const w = p2.x - p1.x;
  const h = p2.y - p1.y;

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ff3b30";
  ctx.strokeRect(x, y, w, h);

  const label = box.score != null ? `${(box.score * 100).toFixed(0)}%` : "manual";
  ctx.font = "12px sans-serif";
  ctx.textBaseline = "top";
  const labelWidth = ctx.measureText(label).width + 6;
  const labelY = Math.max(0, y - 16);
  ctx.fillStyle = "#ff3b30";
  ctx.fillRect(x, labelY, labelWidth, 16);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, x + 3, labelY + 1);

  if (!editingEnabled) return;

  // Resize handles at each corner.
  ctx.fillStyle = "#ff3b30";
  [
    [x, y],
    [x + w, y],
    [x, y + h],
    [x + w, y + h],
  ].forEach(([hx, hy]) => {
    ctx.fillRect(hx - HANDLE_SIZE / 2, hy - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE);
  });

  // Delete "x" control, just outside the top-right corner.
  const dx = x + w + 2;
  const dy = y - 2;
  ctx.beginPath();
  ctx.arc(dx, dy, DELETE_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = "#333";
  ctx.fill();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(dx - 3, dy - 3);
  ctx.lineTo(dx + 3, dy + 3);
  ctx.moveTo(dx + 3, dy - 3);
  ctx.lineTo(dx - 3, dy + 3);
  ctx.stroke();
}

function renderBoxList(boxes) {
  boxListEl.innerHTML = "";

  if (boxes.length === 0) {
    const li = document.createElement("li");
    li.textContent = editingEnabled
      ? "No boxes -- will be saved as a negative (no damage) example."
      : "No detections above threshold.";
    boxListEl.appendChild(li);
    return;
  }

  boxes.forEach((box, i) => {
    const li = document.createElement("li");
    const coords = [box.x_min, box.y_min, box.x_max, box.y_max]
      .map((v) => Math.round(v))
      .join(", ");
    const label =
      box.score != null ? `confidence ${(box.score * 100).toFixed(1)}%` : "manually added";
    li.textContent = `#${i + 1} — ${label} — box [${coords}]`;
    boxListEl.appendChild(li);
  });
}

function isValidBox(box) {
  return box.x_min < box.x_max && box.y_min < box.y_max;
}

function updateSaveButtonState() {
  const valid = editingEnabled && currentBoxes.every(isValidBox);
  saveCorrectionBtn.disabled = !valid;
  saveAndNextBtn.disabled = !valid;
}

// ---- Hit testing for editing ----

function hitDeleteControl(box, pos) {
  const p1 = imgToCanvas({ x: box.x_min, y: box.y_min });
  const p2 = imgToCanvas({ x: box.x_max, y: box.y_max });
  const dx = p2.x + 2;
  const dy = p1.y - 2;
  return Math.hypot(pos.x - dx, pos.y - dy) <= DELETE_RADIUS + 3;
}

function hitCorner(box, pos) {
  const p1 = imgToCanvas({ x: box.x_min, y: box.y_min });
  const p2 = imgToCanvas({ x: box.x_max, y: box.y_max });
  const corners = {
    nw: { x: p1.x, y: p1.y },
    ne: { x: p2.x, y: p1.y },
    sw: { x: p1.x, y: p2.y },
    se: { x: p2.x, y: p2.y },
  };
  for (const [name, c] of Object.entries(corners)) {
    if (Math.hypot(pos.x - c.x, pos.y - c.y) <= HANDLE_SIZE) return name;
  }
  return null;
}

function hitBody(box, pos) {
  const p1 = imgToCanvas({ x: box.x_min, y: box.y_min });
  const p2 = imgToCanvas({ x: box.x_max, y: box.y_max });
  return pos.x >= p1.x && pos.x <= p2.x && pos.y >= p1.y && pos.y <= p2.y;
}

// ---- Draw-new-box mode toggle ----

drawModeBtn.addEventListener("click", () => {
  if (!editingEnabled) return;
  drawMode = !drawMode;
  drawModeBtn.textContent = drawMode ? "Drawing... (click again to stop)" : "Draw new box";
  drawModeBtn.classList.toggle("active", drawMode);
  canvas.classList.toggle("draw-mode", drawMode);
});

// ---- Mouse-driven editing: move, resize, delete, draw ----

canvas.addEventListener("mousedown", (evt) => {
  if (!editingEnabled) return;
  const pos = getCanvasPos(evt);

  if (drawMode) {
    interaction = { type: "draw", startX: pos.x, startY: pos.y, currentX: pos.x, currentY: pos.y };
    return;
  }

  // Iterate topmost (last-drawn) box first so overlapping boxes resolve predictably.
  for (let i = currentBoxes.length - 1; i >= 0; i--) {
    if (hitDeleteControl(currentBoxes[i], pos)) {
      currentBoxes.splice(i, 1);
      render();
      renderBoxList(currentBoxes);
      updateSaveButtonState();
      return;
    }
  }

  for (let i = currentBoxes.length - 1; i >= 0; i--) {
    const corner = hitCorner(currentBoxes[i], pos);
    if (corner) {
      interaction = { type: "resize", index: i, corner };
      return;
    }
  }

  for (let i = currentBoxes.length - 1; i >= 0; i--) {
    if (hitBody(currentBoxes[i], pos)) {
      const p1 = imgToCanvas({ x: currentBoxes[i].x_min, y: currentBoxes[i].y_min });
      interaction = { type: "move", index: i, grabDx: pos.x - p1.x, grabDy: pos.y - p1.y };
      return;
    }
  }
});

canvas.addEventListener("mousemove", (evt) => {
  if (!interaction) return;
  const pos = getCanvasPos(evt);
  pos.x = clamp(pos.x, 0, canvas.width);
  pos.y = clamp(pos.y, 0, canvas.height);

  if (interaction.type === "draw") {
    interaction.currentX = pos.x;
    interaction.currentY = pos.y;
    render();
    return;
  }

  const box = currentBoxes[interaction.index];
  const p1 = imgToCanvas({ x: box.x_min, y: box.y_min });
  const p2 = imgToCanvas({ x: box.x_max, y: box.y_max });

  if (interaction.type === "move") {
    const w = p2.x - p1.x;
    const h = p2.y - p1.y;
    const newX = clamp(pos.x - interaction.grabDx, 0, canvas.width - w);
    const newY = clamp(pos.y - interaction.grabDy, 0, canvas.height - h);
    const img1 = canvasToImg({ x: newX, y: newY });
    const img2 = canvasToImg({ x: newX + w, y: newY + h });
    box.x_min = img1.x;
    box.y_min = img1.y;
    box.x_max = img2.x;
    box.y_max = img2.y;
  } else if (interaction.type === "resize") {
    let nx1 = p1.x;
    let ny1 = p1.y;
    let nx2 = p2.x;
    let ny2 = p2.y;
    if (interaction.corner === "nw") {
      nx1 = pos.x;
      ny1 = pos.y;
    } else if (interaction.corner === "ne") {
      nx2 = pos.x;
      ny1 = pos.y;
    } else if (interaction.corner === "sw") {
      nx1 = pos.x;
      ny2 = pos.y;
    } else if (interaction.corner === "se") {
      nx2 = pos.x;
      ny2 = pos.y;
    }

    if (nx2 - nx1 >= MIN_BOX_SIZE && ny2 - ny1 >= MIN_BOX_SIZE) {
      const img1 = canvasToImg({ x: nx1, y: ny1 });
      const img2 = canvasToImg({ x: nx2, y: ny2 });
      box.x_min = img1.x;
      box.y_min = img1.y;
      box.x_max = img2.x;
      box.y_max = img2.y;
    }
  }

  // Editing a prediction supersedes its confidence score -- it's now a
  // user-provided label, not a model guess.
  delete box.score;

  render();
  renderBoxList(currentBoxes);
  updateSaveButtonState();
});

window.addEventListener("mouseup", () => {
  if (!interaction) return;

  if (interaction.type === "draw") {
    const x1 = Math.min(interaction.startX, interaction.currentX);
    const y1 = Math.min(interaction.startY, interaction.currentY);
    const x2 = Math.max(interaction.startX, interaction.currentX);
    const y2 = Math.max(interaction.startY, interaction.currentY);
    if (x2 - x1 >= MIN_BOX_SIZE && y2 - y1 >= MIN_BOX_SIZE) {
      const img1 = canvasToImg({ x: x1, y: y1 });
      const img2 = canvasToImg({ x: x2, y: y2 });
      currentBoxes.push({ x_min: img1.x, y_min: img1.y, x_max: img2.x, y_max: img2.y });
    }
  }

  interaction = null;
  render();
  renderBoxList(currentBoxes);
  updateSaveButtonState();
});

// ---- Save correction (shared by the single-upload and review-queue flows) ----

// POSTs the current image + currentBoxes to /correct. Returns the parsed
// response body on success, or null on failure (with correctionStatusEl
// already updated to explain why) -- callers decide what "success" means
// for their flow (just report it, vs. also advancing the queue).
async function submitCorrection() {
  const formData = new FormData();
  formData.append("file", currentFile, currentFilename);
  formData.append(
    "boxes",
    JSON.stringify(
      currentBoxes.map((b) => ({
        x_min: b.x_min,
        y_min: b.y_min,
        x_max: b.x_max,
        y_max: b.y_max,
      }))
    )
  );

  let response;
  try {
    response = await fetch(CORRECT_URL, { method: "POST", body: formData });
  } catch (err) {
    correctionStatusEl.textContent = "Could not reach the backend.";
    correctionStatusEl.className = "status error";
    return null;
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errBody = await response.json();
      detail = errBody.detail || detail;
    } catch (_) {
      // response body wasn't JSON
    }
    correctionStatusEl.textContent = `Save failed (${response.status}): ${detail}`;
    correctionStatusEl.className = "status error";
    return null;
  }

  return response.json();
}

saveCorrectionBtn.addEventListener("click", async () => {
  if (!currentFile || !editingEnabled) return;

  saveCorrectionBtn.disabled = true;
  correctionStatusEl.textContent = "Saving correction...";
  correctionStatusEl.className = "status loading";

  const data = await submitCorrection();
  if (data) {
    correctionStatusEl.textContent = `Correction saved. Total corrections: ${data.total_corrections}.`;
    correctionStatusEl.className = "status success";
    correctionCountEl.textContent = data.total_corrections;
  }
  updateSaveButtonState();
});

saveAndNextBtn.addEventListener("click", async () => {
  if (!currentFile || !editingEnabled) return;

  saveAndNextBtn.disabled = true;
  correctionStatusEl.textContent = "Saving correction...";
  correctionStatusEl.className = "status loading";

  const data = await submitCorrection();
  if (!data) {
    updateSaveButtonState();
    return;
  }

  correctionCountEl.textContent = data.total_corrections;
  correctionStatusEl.textContent = "Correction saved.";
  correctionStatusEl.className = "status success";
  await loadNextInQueue();
});

// ---- Corrections count (loaded once on page load, refreshed after each save) ----

async function refreshCorrectionCount() {
  try {
    const res = await fetch(CORRECTIONS_COUNT_URL);
    if (res.ok) {
      const data = await res.json();
      correctionCountEl.textContent = data.count;
    }
  } catch (_) {
    // Backend not reachable yet -- leave the placeholder, /health-style
    // errors are already surfaced via the main detect flow.
  }
}

// ---- Retrain panel ----

function applyRetrainStatus(data) {
  if (data.status === "running") {
    retrainStatusEl.textContent = "Training...";
    retrainStatusEl.className = "status loading";
    retrainBtn.disabled = true;
    if (!retrainPollHandle) {
      retrainPollHandle = setInterval(refreshRetrainStatus, RETRAIN_POLL_MS);
    }
    return;
  }

  if (retrainPollHandle) {
    clearInterval(retrainPollHandle);
    retrainPollHandle = null;
  }
  retrainBtn.disabled = false;

  if (data.status === "finished") {
    const lossText = data.final_loss != null ? data.final_loss.toFixed(4) : "unknown";
    retrainStatusEl.textContent = `Finished. Final loss: ${lossText}`;
    retrainStatusEl.className = "status success";
  } else if (data.status === "failed") {
    retrainStatusEl.textContent = `Failed: ${data.error || "unknown error"}`;
    retrainStatusEl.className = "status error";
  } else {
    retrainStatusEl.textContent = "No retrain run yet.";
    retrainStatusEl.className = "status";
  }
}

async function refreshRetrainStatus() {
  let res;
  try {
    res = await fetch(RETRAIN_STATUS_URL);
  } catch (_) {
    return;
  }
  if (!res.ok) return;
  applyRetrainStatus(await res.json());
}

retrainBtn.addEventListener("click", async () => {
  retrainBtn.disabled = true;
  retrainStatusEl.textContent = "Starting retrain...";
  retrainStatusEl.className = "status loading";

  let response;
  try {
    response = await fetch(RETRAIN_URL, { method: "POST" });
  } catch (err) {
    retrainStatusEl.textContent = "Could not reach the backend.";
    retrainStatusEl.className = "status error";
    retrainBtn.disabled = false;
    return;
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errBody = await response.json();
      detail = errBody.detail || detail;
    } catch (_) {
      // response body wasn't JSON
    }
    retrainStatusEl.textContent = `Could not start retrain (${response.status}): ${detail}`;
    retrainStatusEl.className = "status error";
    retrainBtn.disabled = false;
    return;
  }

  applyRetrainStatus({ status: "running" });
});

// ---- Review queue tab ----

let activeTab = "single";
let queueLoadedOnce = false; // load the first queue item lazily, the first time the tab is opened

tabSingleBtn.addEventListener("click", () => switchTab("single"));
tabQueueBtn.addEventListener("click", () => switchTab("queue"));

function switchTab(tab) {
  activeTab = tab;
  tabSingleBtn.classList.toggle("active", tab === "single");
  tabQueueBtn.classList.toggle("active", tab === "queue");
  singleUploadPanel.hidden = tab !== "single";
  reviewQueuePanel.hidden = tab !== "queue";
  saveCorrectionBtn.hidden = tab !== "single";
  saveAndNextBtn.hidden = tab !== "queue";

  if (tab === "queue") {
    refreshQueueStats();
    if (!queueLoadedOnce) {
      queueLoadedOnce = true;
      loadNextInQueue();
    }
  }
}

// Buckets the "uncertainty" field (0 == least confident, 0.5 == most
// confident -- see backend/batch_prelabel.py's compute_uncertainty) into
// a plain-language confidence level for display.
function describeConfidence(uncertainty) {
  const normalized = clamp(uncertainty / 0.5, 0, 1);
  if (normalized < 0.33) return "low";
  if (normalized < 0.66) return "medium";
  return "high";
}

async function refreshQueueStats() {
  try {
    const res = await fetch(QUEUE_STATS_URL);
    if (!res.ok) return;
    const data = await res.json();
    if (data.total === 0) {
      queueProgressEl.textContent = "No review queue yet.";
      queueProgressFillEl.style.width = "0%";
      return;
    }
    queueProgressEl.textContent = `${data.reviewed} of ${data.total} reviewed`;
    queueProgressFillEl.style.width = `${Math.round((data.reviewed / data.total) * 100)}%`;
  } catch (_) {
    // Backend not reachable -- leave whatever was last shown.
  }
}

async function loadNextInQueue() {
  saveAndNextBtn.disabled = true;
  queueUncertaintyEl.textContent = "";
  correctionStatusEl.textContent = "";
  queueMessageEl.textContent = "Loading next image...";
  queueMessageEl.className = "status loading";
  resetEditingState();

  let response;
  try {
    response = await fetch(QUEUE_NEXT_URL);
  } catch (err) {
    queueMessageEl.textContent = "Could not reach the backend.";
    queueMessageEl.className = "status error";
    return;
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errBody = await response.json();
      detail = errBody.detail || detail;
    } catch (_) {
      // response body wasn't JSON
    }
    queueMessageEl.textContent = `Could not load queue (${response.status}): ${detail}`;
    queueMessageEl.className = "status error";
    return;
  }

  const data = await response.json();

  if (data.done) {
    currentImage = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    boxListEl.innerHTML = "";
    queueMessageEl.textContent = data.message || "Queue empty -- nothing left to review.";
    queueMessageEl.className = "status empty";
    return;
  }

  let imageBlob;
  try {
    const imgRes = await fetch(`${API_BASE}${data.image_url}`);
    if (!imgRes.ok) throw new Error(`HTTP ${imgRes.status}`);
    imageBlob = await imgRes.blob();
  } catch (err) {
    queueMessageEl.textContent = `Could not load image for ${data.filename}: ${err.message}`;
    queueMessageEl.className = "status error";
    return;
  }

  currentFile = imageBlob;
  currentFilename = data.filename;

  if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
  currentObjectUrl = URL.createObjectURL(imageBlob);

  try {
    await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        currentImage = img;
        imgWidth = img.naturalWidth;
        imgHeight = img.naturalHeight;
        resolve();
      };
      img.onerror = () => reject(new Error("failed to decode image"));
      img.src = currentObjectUrl;
    });
  } catch (err) {
    queueMessageEl.textContent = `Could not display image for ${data.filename}: ${err.message}`;
    queueMessageEl.className = "status error";
    return;
  }

  currentBoxes = data.boxes.map((b) => ({
    x_min: b.x_min,
    y_min: b.y_min,
    x_max: b.x_max,
    y_max: b.y_max,
    score: b.score,
  }));
  editingEnabled = true;

  drawImageOnly();
  render();
  renderBoxList(currentBoxes);
  updateSaveButtonState();

  queueMessageEl.textContent = `Reviewing: ${data.filename}`;
  queueMessageEl.className = "status";

  const level = describeConfidence(data.uncertainty);
  queueUncertaintyEl.textContent = `Model confidence: ${level} (uncertainty score ${data.uncertainty.toFixed(3)})`;
  queueUncertaintyEl.className = `uncertainty-indicator uncertainty-${level}`;

  await refreshQueueStats();
}

// Pick up an already-running (or already-finished) retrain across a page reload.
refreshCorrectionCount();
refreshRetrainStatus();
