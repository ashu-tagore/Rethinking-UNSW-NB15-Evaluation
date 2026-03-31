"use strict";

// ── Constants ─────────────────────────────────────────────────────────────────
const CLASSES = ["Normal","Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance","Shellcode","Worms"];
const CLASS_COLORS = ["#22c55e","#facc15","#a855f7","#ef4444","#f97316","#3b82f6","#6b7280","#06b6d4","#ec4899","#84cc16"];
const TIMELINE_LEN = 40;
const CHART_CFG = {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { labels: { color: "#9ca3af", font: { size: 11 }, boxWidth: 12 } } },
    scales: {
        x: { ticks: { color: "#6b7280", maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#6b7280" }, grid: { color: "rgba(255,255,255,0.05)" }, beginAtZero: true },
    },
};

// ── State ─────────────────────────────────────────────────────────────────────
let tlNormal = new Array(TIMELINE_LEN).fill(0);
let tlAttack = new Array(TIMELINE_LEN).fill(0);
let tlTick   = Date.now();

let classCounts = Object.fromEntries(CLASSES.map(c => [c, 0]));
let allAlerts   = [];
let totalFlows  = 0;
let totalAttack = 0;
let eventSource = null;

let chartTimeline  = null;
let chartBreakdown = null;
let chartPerclass  = null;

// ── Charts init ───────────────────────────────────────────────────────────────
function initCharts() {
    const tickLabels = Array.from({ length: TIMELINE_LEN }, (_, i) => `-${TIMELINE_LEN - i}s`);

    chartTimeline = new Chart(document.getElementById("chart-timeline"), {
        type: "line",
        data: {
            labels: tickLabels,
            datasets: [
                { label: "Normal", data: [...tlNormal], borderColor: "#22c55e", backgroundColor: "rgba(34,197,94,0.08)", tension: 0.4, fill: true, pointRadius: 0 },
                { label: "Attack", data: [...tlAttack], borderColor: "#ef4444", backgroundColor: "rgba(239,68,68,0.08)", tension: 0.4, fill: true, pointRadius: 0 },
            ],
        },
        options: { ...CHART_CFG },
    });

    chartBreakdown = new Chart(document.getElementById("chart-breakdown"), {
        type: "doughnut",
        data: {
            labels: CLASSES,
            datasets: [{ data: CLASSES.map(() => 0), backgroundColor: CLASS_COLORS, borderWidth: 0, hoverBorderWidth: 2 }],
        },
        options: { responsive: true, maintainAspectRatio: false, animation: false, cutout: "60%",
            plugins: { legend: { labels: { color: "#9ca3af", font: { size: 10 }, boxWidth: 10 } } } },
    });
}

// ── Timeline tick ─────────────────────────────────────────────────────────────
function tickTimeline() {
    const now = Date.now();
    if (now - tlTick >= 1000) {
        tlNormal.shift(); tlNormal.push(0);
        tlAttack.shift();  tlAttack.push(0);
        tlTick = now;
    }
}

function pushToTimeline(isAttack) {
    tickTimeline();
    if (isAttack) tlAttack[tlAttack.length - 1]++;
    else          tlNormal[tlNormal.length - 1]++;
    chartTimeline.data.datasets[0].data = [...tlNormal];
    chartTimeline.data.datasets[1].data = [...tlAttack];
    chartTimeline.update("none");
}

// ── Live capture ──────────────────────────────────────────────────────────────
async function startCapture() {
    const iface = document.getElementById("iface-input").value.trim();
    const url   = iface
        ? `/api/monitor/start?interface=${encodeURIComponent(iface)}`
        : "/api/monitor/start";
    const res = await fetch(url, { method: "POST" });
    if (!res.ok) { showToast("Failed to start capture.", "error"); return; }

    document.getElementById("btn-start").disabled = true;
    document.getElementById("btn-stop").disabled  = false;
    document.getElementById("capture-badge").classList.remove("hidden");
    document.getElementById("capture-badge").classList.add("flex");

    eventSource = new EventSource("/api/monitor/stream");
    eventSource.onmessage = e => handleEvent(JSON.parse(e.data));
    eventSource.onerror   = () => showToast("Stream error — check backend.", "error");
    showToast("Capture started.", "success");
}

async function stopCapture() {
    await fetch("/api/monitor/stop", { method: "POST" });
    if (eventSource) { eventSource.close(); eventSource = null; }
    document.getElementById("btn-start").disabled = false;
    document.getElementById("btn-stop").disabled  = true;
    document.getElementById("capture-badge").classList.add("hidden");
    document.getElementById("capture-badge").classList.remove("flex");
    showToast("Capture stopped.", "info");
}

// ── Event handler ─────────────────────────────────────────────────────────────
function handleEvent(data) {
    totalFlows++;
    if (data.is_attack) totalAttack++;

    allAlerts.unshift(data);
    if (allAlerts.length > 500) allAlerts.pop();

    classCounts[data.label] = (classCounts[data.label] || 0) + 1;

    // Update stat cards
    document.getElementById("stat-total").textContent   = formatNumber(totalFlows);
    document.getElementById("stat-attacks").textContent = formatNumber(totalAttack);
    document.getElementById("stat-normal").textContent  = formatNumber(totalFlows - totalAttack);
    const rate = totalFlows ? ((totalAttack / totalFlows) * 100).toFixed(1) : "0";
    document.getElementById("stat-rate").textContent = rate + "%";

    // Update charts
    pushToTimeline(data.is_attack);
    chartBreakdown.data.datasets[0].data = CLASSES.map(c => classCounts[c]);
    chartBreakdown.update("none");

    renderAlerts();
}

function renderAlerts() {
    const attacksOnly = document.getElementById("chk-attacks").checked;
    const rows = (attacksOnly ? allAlerts.filter(a => a.is_attack) : allAlerts).slice(0, 200);

    if (rows.length === 0) {
        document.getElementById("alert-body").innerHTML =
            `<tr><td colspan="7" class="py-8 text-center text-gray-500">No events yet</td></tr>`;
        return;
    }

    document.getElementById("alert-body").innerHTML = rows.map(a => {
        const t    = new Date(a.timestamp * 1000).toLocaleTimeString();
        const badge = a.is_attack
            ? `<span class="px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 border border-red-500/30 text-xs font-medium">⚠ ${a.label}</span>`
            : `<span class="px-2 py-0.5 rounded-full bg-green-500/20 text-green-400 border border-green-500/30 text-xs font-medium">✓ ${a.label}</span>`;
        return `<tr class="hover:bg-white/5 transition-all">
          <td class="py-2.5 pr-4 text-gray-400 font-mono text-xs">${t}</td>
          <td class="py-2.5 pr-4 font-mono text-xs">${a.src}</td>
          <td class="py-2.5 pr-4 font-mono text-xs">${a.dst}</td>
          <td class="py-2.5 pr-4 text-gray-400 uppercase text-xs">${a.proto}</td>
          <td class="py-2.5 pr-4">${badge}</td>
          <td class="py-2.5 pr-4 text-gray-400 text-xs">${a.confidence}%</td>
          <td class="py-2.5 text-gray-400 text-xs">${formatBytes(a.bytes)}</td>
        </tr>`;
    }).join("");
}

// ── Model stats ───────────────────────────────────────────────────────────────
async function loadExperiments() {
    const res  = await fetch("/api/experiments");
    const data = await res.json();
    const sel  = document.getElementById("exp-select");
    sel.innerHTML = data.experiments.length
        ? data.experiments.map(e => `<option value="${e}">${e}</option>`).join("")
        : `<option value="exp01">exp01</option>`;
}

async function loadStats() {
    const exp = document.getElementById("exp-select")?.value || "exp01";
    const res = await fetch(`/api/model/stats?exp=${exp}`);
    if (!res.ok) {
        document.getElementById("perf-cards").innerHTML =
            `<p class="col-span-4 text-gray-500 text-sm">No metrics file found for ${exp}. Save classification_report.json from training.</p>`;
        return;
    }
    const data = await res.json();
    renderPerfCards(data);
    renderPerclassChart(data);
    renderPerclassTable(data);
}

function _get(obj, key) {
    return obj?.[key] ?? obj?.[key.replace("-", "_")] ?? 0;
}

function renderPerfCards(data) {
    const macro = data["macro avg"] || data["macro_avg"] || {};
    const acc   = data.accuracy || 0;
    const cards = [
        { label: "Accuracy",        val: (acc * 100).toFixed(1) + "%" },
        { label: "Macro F1",        val: (_get(macro, "f1-score") * 100).toFixed(1) + "%" },
        { label: "Macro Precision", val: (_get(macro, "precision") * 100).toFixed(1) + "%" },
        { label: "Macro Recall",    val: (_get(macro, "recall") * 100).toFixed(1) + "%" },
    ];
    document.getElementById("perf-cards").innerHTML = cards.map(c => `
      <div class="bg-white/5 rounded-xl p-5 border border-white/10 text-center">
        <p class="text-3xl font-bold mb-1">${c.val}</p>
        <p class="text-xs text-gray-400 uppercase tracking-widest">${c.label}</p>
      </div>
    `).join("");
}

function renderPerclassChart(data) {
    const classes = CLASSES.filter(c => data[c] != null);
    if (chartPerclass) chartPerclass.destroy();
    chartPerclass = new Chart(document.getElementById("chart-perclass"), {
        type: "bar",
        data: {
            labels: classes,
            datasets: [
                { label: "Precision", data: classes.map(c => _get(data[c], "precision")), backgroundColor: "rgba(59,130,246,0.55)" },
                { label: "Recall",    data: classes.map(c => _get(data[c], "recall")),    backgroundColor: "rgba(250,204,21,0.55)" },
                { label: "F1",        data: classes.map(c => _get(data[c], "f1-score")), backgroundColor: "rgba(34,197,94,0.55)" },
            ],
        },
        options: { ...CHART_CFG, scales: { ...CHART_CFG.scales, y: { ...CHART_CFG.scales.y, max: 1 } } },
    });
}

function renderPerclassTable(data) {
    const classes = CLASSES.filter(c => data[c] != null);
    document.getElementById("perclass-body").innerHTML = classes.map(c => {
        const f1  = _get(data[c], "f1-score");
        const col = f1 >= 0.5 ? "text-green-400" : f1 >= 0.2 ? "text-yellow-400" : "text-red-400";
        return `<tr class="hover:bg-white/5 transition-all">
          <td class="py-2.5 pr-6 font-medium">${c}</td>
          <td class="py-2.5 pr-6 text-gray-300">${(_get(data[c], "precision") * 100).toFixed(1)}%</td>
          <td class="py-2.5 pr-6 text-gray-300">${(_get(data[c], "recall") * 100).toFixed(1)}%</td>
          <td class="py-2.5 pr-6 font-bold ${col}">${(f1 * 100).toFixed(1)}%</td>
          <td class="py-2.5 text-gray-400">${_get(data[c], "support") || "—"}</td>
        </tr>`;
    }).join("");
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
    initCharts();

    // Keep timeline scrolling even when idle
    setInterval(() => {
        tickTimeline();
        chartTimeline.data.datasets[0].data = [...tlNormal];
        chartTimeline.data.datasets[1].data = [...tlAttack];
        chartTimeline.update("none");
    }, 1000);

    // Check if capture already running (page reload)
    try {
        const res  = await fetch("/api/monitor/status");
        const data = await res.json();
        if (data.running) {
            document.getElementById("btn-start").disabled = true;
            document.getElementById("btn-stop").disabled  = false;
            document.getElementById("capture-badge").classList.remove("hidden");
            document.getElementById("capture-badge").classList.add("flex");
            eventSource = new EventSource("/api/monitor/stream");
            eventSource.onmessage = e => handleEvent(JSON.parse(e.data));
        }
    } catch {
        showToast("Could not reach backend.", "error");
    }

    await loadExperiments();
    await loadStats();
})();
