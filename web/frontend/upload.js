"use strict";

let selectedFile = null;

const dropZone   = document.getElementById("drop-zone");
const fileInput  = document.getElementById("file-input");
const uploadBtn  = document.getElementById("upload-btn");
const fileInfo   = document.getElementById("file-info");
const uploadPrompt   = document.getElementById("upload-prompt");
const loadingState   = document.getElementById("loading-state");
const resultsContainer = document.getElementById("results-container");

// ── Drop zone interactions ────────────────────────────────────────────────────

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("border-white", "bg-white/10");
});

dropZone.addEventListener("dragleave", e => {
    e.preventDefault();
    dropZone.classList.remove("border-white", "bg-white/10");
});

dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("border-white", "bg-white/10");
    if (e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", e => {
    if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
});

// ── File selection ────────────────────────────────────────────────────────────

function handleFileSelect(file) {
    if (!file.name.endsWith(".csv")) {
        showToast("Only CSV files are accepted.", "error");
        return;
    }
    selectedFile = file;
    document.getElementById("file-name").textContent = file.name;
    document.getElementById("file-size").textContent = formatFileSize(file.size);
    fileInfo.classList.remove("hidden");
    uploadBtn.classList.remove("hidden");
    resultsContainer.classList.add("hidden");
}

document.getElementById("remove-file").addEventListener("click", e => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = "";
    fileInfo.classList.add("hidden");
    uploadBtn.classList.add("hidden");
});

uploadBtn.addEventListener("click", () => {
    if (selectedFile) uploadFile(selectedFile);
});

// ── Upload → /api/predict/batch ───────────────────────────────────────────────

async function uploadFile(file) {
    const form = new FormData();
    form.append("file", file);

    uploadPrompt.classList.add("hidden");
    loadingState.classList.remove("hidden");
    uploadBtn.classList.add("hidden");
    fileInfo.classList.add("hidden");

    try {
        const res  = await fetch("/api/predict/batch", { method: "POST", body: form });
        const data = await res.json();

        if (res.ok) {
            showResults(data);
            showToast("Analysis complete!", "success");
        } else {
            showError(data.detail || "An error occurred.");
            showToast(data.detail || "Upload failed", "error");
        }
    } catch (err) {
        showError("Network error: " + err.message);
        showToast("Network error occurred", "error");
    } finally {
        uploadPrompt.classList.remove("hidden");
        loadingState.classList.add("hidden");
        selectedFile = null;
        fileInput.value = "";
    }
}

// ── Results rendering ─────────────────────────────────────────────────────────

function showResults(data) {
    const { summary, results, filename } = data;

    resultsContainer.innerHTML = `
      <div class="bg-white/5 rounded-3xl border border-white/10 p-8 mb-8">
        <h2 class="text-3xl font-bold mb-6">Analysis Results</h2>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div class="bg-white/5 rounded-xl p-6 border border-white/10">
            <p class="text-sm text-gray-400 mb-2">Total Analyzed</p>
            <p class="text-4xl font-bold">${formatNumber(summary.total)}</p>
          </div>
          <div class="bg-red-500/10 rounded-xl p-6 border border-red-500/20">
            <p class="text-sm text-gray-400 mb-2">Malicious</p>
            <p class="text-4xl font-bold text-red-400">${formatNumber(summary.malicious)}</p>
          </div>
          <div class="bg-green-500/10 rounded-xl p-6 border border-green-500/20">
            <p class="text-sm text-gray-400 mb-2">Benign</p>
            <p class="text-4xl font-bold text-green-400">${formatNumber(summary.benign)}</p>
          </div>
        </div>

        <h3 class="text-xl font-semibold mb-4">Sample Detections (First 10)</h3>
        <div class="space-y-3">
          ${results.slice(0, 10).map(r => `
            <div class="bg-white/5 rounded-xl p-4 border border-white/10 flex items-center justify-between hover:bg-white/10 transition-all">
              <div>
                <p class="font-semibold">${r.prediction}</p>
                <p class="text-sm text-gray-400">Confidence: ${(r.confidence * 100).toFixed(2)}%</p>
              </div>
              ${severityBadge(r.severity, r.is_malicious ? "⚠ Malicious" : "✓ Benign")}
            </div>
          `).join("")}
        </div>

        <div class="mt-6 text-center">
          <a href="/dashboard.html"
             class="inline-block px-8 py-3 bg-white text-black rounded-full font-semibold hover:bg-gray-200 transition-all">
            View Live Dashboard →
          </a>
        </div>
      </div>
    `;

    resultsContainer.classList.remove("hidden");
}

function showError(message) {
    resultsContainer.innerHTML = `
      <div class="bg-red-500/10 rounded-3xl border border-red-500/20 p-8 text-center">
        <svg class="w-16 h-16 mx-auto mb-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3 class="text-2xl font-bold mb-2">Analysis Failed</h3>
        <p class="text-gray-400">${message}</p>
      </div>
    `;
    resultsContainer.classList.remove("hidden");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatFileSize(bytes) {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + " " + sizes[i];
}
