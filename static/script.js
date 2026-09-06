const BATCH_SIZE = 40; // upload images in small batches so 1000+ files don't hit request limits

async function uploadInBatches(files, endpoint, progressFillEl, progressTextEl, progressWrapEl, countEl, countKey) {
  if (!files.length) return;
  progressWrapEl.style.display = "flex";
  let uploaded = 0;

  for (let i = 0; i < files.length; i += BATCH_SIZE) {
    const batch = files.slice(i, i + BATCH_SIZE);
    const formData = new FormData();
    batch.forEach(f => formData.append("images", f));

    const res = await fetch(endpoint, { method: "POST", body: formData });
    const data = await res.json();

    uploaded += batch.length;
    const pct = Math.round((uploaded / files.length) * 100);
    progressFillEl.style.width = pct + "%";
    progressTextEl.textContent = `${uploaded} / ${files.length}`;
    if (data[countKey] !== undefined) countEl.textContent = data[countKey];
  }
}

function setupDropzone(dropzoneEl, fileInputEls, onFiles) {
  dropzoneEl.addEventListener("dragover", e => {
    e.preventDefault();
    dropzoneEl.classList.add("dragover");
  });
  dropzoneEl.addEventListener("dragleave", () => dropzoneEl.classList.remove("dragover"));
  dropzoneEl.addEventListener("drop", e => {
    e.preventDefault();
    dropzoneEl.classList.remove("dragover");
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("image/"));
    onFiles(files);
  });
  fileInputEls.forEach(inputEl => {
    inputEl.addEventListener("change", () => onFiles(Array.from(inputEl.files)));
  });
}

// ---- Training upload ----
const trainDropzone = document.getElementById("train-dropzone");
const trainFileInput = document.getElementById("train-files");
const trainFolderInput = document.getElementById("train-folder");
const trainProgressFill = document.getElementById("train-progress-fill");
const trainProgressText = document.getElementById("train-progress-text");
const trainProgressWrap = document.getElementById("train-progress-wrap");
const trainCount = document.getElementById("train-count");

setupDropzone(trainDropzone, [trainFileInput, trainFolderInput], files => {
  uploadInBatches(files, "/upload_train", trainProgressFill, trainProgressText, trainProgressWrap, trainCount, "total_train_images");
});

// ---- Test upload ----
const testDropzone = document.getElementById("test-dropzone");
const testFileInput = document.getElementById("test-files");
const testProgressFill = document.getElementById("test-progress-fill");
const testProgressText = document.getElementById("test-progress-text");
const testProgressWrap = document.getElementById("test-progress-wrap");
const testCount = document.getElementById("test-count");

setupDropzone(testDropzone, [testFileInput], files => {
  uploadInBatches(files, "/upload_test", testProgressFill, testProgressText, testProgressWrap, testCount, "total_test_images");
});

// ---- Train model (build feature memory bank) ----
document.getElementById("train-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("train-status");
  statusEl.textContent = "Extracting features and building memory bank... this can take a bit for many images.";

  try {
    const res = await fetch("/train", { method: "POST" });
    const data = await res.json();
    if (data.error) {
      statusEl.textContent = "Error: " + data.error;
      return;
    }
    statusEl.textContent = `Trained on ${data.images_used} images.`;
    document.getElementById("model-status").textContent = "Trained ✅";
  } catch (err) {
    statusEl.textContent = "Training failed: " + err;
  }
});

// ---- Sensitivity slider ----
const sensitivitySlider = document.getElementById("sensitivity");
const sensitivityValue = document.getElementById("sensitivity-value");
sensitivitySlider.addEventListener("input", () => {
  sensitivityValue.textContent = sensitivitySlider.value;
});

// ---- Run detection ----
document.getElementById("detect-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("detect-status");
  statusEl.textContent = "Running inference on test images...";

  try {
    const formData = new FormData();
    formData.append("sensitivity", sensitivitySlider.value);
    const res = await fetch("/detect", { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) {
      statusEl.textContent = "Error: " + data.error;
      return;
    }
    statusEl.textContent = `Done. ${data.results.length} images processed.`;
    renderResults(data.results);
  } catch (err) {
    statusEl.textContent = "Detection failed: " + err;
  }
});

function renderResults(results) {
  const grid = document.getElementById("results-grid");
  grid.innerHTML = "";
  results.forEach(r => {
    const item = document.createElement("div");
    item.className = "result-item " + (r.is_anomaly ? "anomaly" : "normal");
    item.innerHTML = `
      <div class="img-pair">
        <img src="${r.original_url}" alt="original">
        <img src="${r.heatmap_url}" alt="heatmap overlay">
      </div>
      <div class="label">${r.is_anomaly ? "🔴 ANOMALY" : "🟢 NORMAL"} (score: ${r.score}, threshold: ${r.threshold})</div>
      <div class="filename">${r.filename}</div>
    `;
    grid.appendChild(item);
  });
}

// ---- Reset ----
document.getElementById("reset-btn").addEventListener("click", async () => {
  if (!confirm("This will delete all uploaded images and the trained model. Continue?")) return;
  await fetch("/reset", { method: "POST" });
  location.reload();
});
