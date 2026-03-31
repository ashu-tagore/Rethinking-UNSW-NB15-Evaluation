"use strict";

// ── Toast notification system ─────────────────────────────────────────────────
function showToast(message, type = "info") {
    const colors = {
        success: "bg-green-500/20 border-green-500/30 text-green-400",
        error:   "bg-red-500/20 border-red-500/30 text-red-400",
        info:    "bg-blue-500/20 border-blue-500/30 text-blue-400",
        warning: "bg-yellow-500/20 border-yellow-500/30 text-yellow-400",
    };
    const icons = { success: "✓", error: "✗", info: "ℹ", warning: "⚠" };

    const toast = document.createElement("div");
    toast.className = `px-6 py-4 rounded-xl border ${colors[type]} backdrop-blur-md flex items-center space-x-3 animate-slide-in`;
    toast.innerHTML = `<span class="text-xl">${icons[type]}</span><span class="font-medium text-sm">${message}</span>`;

    const container = document.getElementById("toast-container");
    if (container) container.appendChild(toast);

    setTimeout(() => {
        toast.style.transition = "opacity 0.3s, transform 0.3s";
        toast.style.opacity = "0";
        toast.style.transform = "translateX(100%)";
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ── Utility helpers ───────────────────────────────────────────────────────────
function formatNumber(n) {
    return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatDate(s) {
    return new Date(s).toLocaleDateString("en-US", {
        year: "numeric", month: "short", day: "numeric",
        hour: "2-digit", minute: "2-digit",
    });
}

function formatBytes(b) {
    if (!b) return "0 B";
    if (b < 1024)    return b + " B";
    if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
    return (b / 1048576).toFixed(1) + " MB";
}

function debounce(fn, wait) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), wait); };
}

function severityBadge(severity, text) {
    const map = {
        safe:     "bg-green-500/20 text-green-400 border border-green-500/30",
        low:      "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
        medium:   "bg-orange-500/20 text-orange-400 border border-orange-500/30",
        high:     "bg-red-500/20 text-red-400 border border-red-500/30",
        critical: "bg-purple-500/20 text-purple-400 border border-purple-500/30",
    };
    return `<span class="px-3 py-1 rounded-full text-xs font-medium ${map[severity] || map.medium}">${text}</span>`;
}

// ── Active nav highlight (client-side since no server templating) ─────────────
(function setActiveNav() {
    const path = window.location.pathname.replace(/\/$/, "") || "/";
    document.querySelectorAll("[data-nav]").forEach(link => {
        const target = link.dataset.nav;
        const isActive =
            (target === "/" && (path === "/" || path === "/index.html")) ||
            (target !== "/" && path.includes(target));
        if (isActive) {
            link.classList.add("bg-white", "text-black");
            link.classList.remove("text-white");
        }
    });
})();
