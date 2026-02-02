window.__APP_JS_LOADED = true;
const POPUPS_VERSION = window.__EXPLAIN_POPUPS_VERSION || "1";

const fields = [
  "user_id",
  "merchant_id",
  "device_id",
  "ip_id",
  "amount",
  "currency",
  "country",
  "channel",
  "drift_phase",
];

const scoreBtn = document.getElementById("scoreBtn");
const randomBtn = document.getElementById("randomBtn");
const scoreError = document.getElementById("scoreError");
const driftPreset = document.getElementById("driftPreset");
const intuitionBtn = document.getElementById("intuitionBtn");
let intuitionModal = document.getElementById("intuitionModal");
let intuitionClose = document.getElementById("intuitionClose");
let intuitionContent = document.getElementById("intuitionContent");
let SELECTED_EVENT_ID = null;

function escapeHtml(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderStaticSections(sections) {
  return (sections || [])
    .map((section) => {
      const type = section?.type;
      if (type === "paragraph") {
        return `<p class="text-slate-700">${escapeHtml(section.text)}</p>`;
      }
      if (type === "callout") {
        return `
          <div class="bg-slate-50 border rounded p-3">
            <div class="font-semibold">${escapeHtml(section.title || "")}</div>
            <div class="text-slate-600">${escapeHtml(section.text || "")}</div>
          </div>
        `;
      }
      if (type === "bullets") {
        const items = (section.items || [])
          .map(
            (item) => `
              <li>
                <span class="font-semibold">${escapeHtml(item.title || "")}</span>
                <span class="text-slate-600"> ${escapeHtml(item.text || "")}</span>
              </li>
            `
          )
          .join("");
        return `
          <div>
            <div class="font-semibold">${escapeHtml(section.title || "")}</div>
            ${items ? `<ul class="list-disc ml-5 mt-2 space-y-1">${items}</ul>` : ""}
          </div>
        `;
      }
      if (type === "table") {
        const columns = section.columns || [];
        const rows = section.rows || [];
        const header = columns
          .map((col) => `<th class="py-1 pr-3 text-left">${escapeHtml(col)}</th>`)
          .join("");
        const body = rows
          .map(
            (row) =>
              `<tr>${(row || [])
                .map((cell) => `<td class="py-1 pr-3">${escapeHtml(cell)}</td>`)
                .join("")}</tr>`
          )
          .join("");
        return `
          <div>
            <div class="font-semibold">${escapeHtml(section.title || "")}</div>
            <div class="overflow-auto mt-2">
              <table class="w-full text-sm">
                <thead class="text-slate-500"><tr>${header}</tr></thead>
                <tbody>${body}</tbody>
              </table>
            </div>
          </div>
        `;
      }
      return "";
    })
    .join("");
}

function setDecisionBadge(decision) {
  const badge = document.getElementById("decisionBadge");
  if (!badge) {
    return;
  }
  badge.textContent = decision || "n/a";
  badge.className = "px-2 py-1 rounded text-sm";
  const styles = {
    approve: "bg-emerald-100 text-emerald-800",
    step_up: "bg-amber-100 text-amber-800",
    review: "bg-orange-100 text-orange-800",
    block: "bg-rose-100 text-rose-800",
  };
  badge.className += " " + (styles[decision] || "bg-slate-200");
}

function fillRandom() {
  const preset = driftPreset?.value || "normal";
  const countries = ["US", "GB", "CA", "DE", "IN", "SG", "AU", "FR", "JP"];
  const hotMerchants = [101, 202, 303, 404];

  document.getElementById("user_id").value = Math.floor(Math.random() * 10000);
  document.getElementById("merchant_id").value = Math.floor(Math.random() * 5000);
  document.getElementById("device_id").value = Math.floor(Math.random() * 2000);
  document.getElementById("ip_id").value = Math.floor(Math.random() * 100000);
  document.getElementById("amount").value = (Math.random() * 5000).toFixed(2);
  document.getElementById("currency").value = "USD";
  document.getElementById("country").value = countries[Math.floor(Math.random() * countries.length)];
  document.getElementById("channel").value = Math.random() > 0.5 ? "web" : "mobile";
  document.getElementById("drift_phase").value = 0;

  if (preset === "new_country") {
    document.getElementById("country").value = "NG";
    document.getElementById("drift_phase").value = 1;
  } else if (preset === "ato_spike") {
    document.getElementById("drift_phase").value = 1;
    document.getElementById("amount").value = (200 + Math.random() * 800).toFixed(2);
    document.getElementById("device_id").value = 50000 + Math.floor(Math.random() * 50000);
    document.getElementById("ip_id").value = 200000 + Math.floor(Math.random() * 50000);
  } else if (preset === "merchant_campaign") {
    document.getElementById("drift_phase").value = 1;
    document.getElementById("merchant_id").value = hotMerchants[Math.floor(Math.random() * hotMerchants.length)];
    document.getElementById("amount").value = (1000 + Math.random() * 3000).toFixed(2);
  }
}

function collectPayload() {
  return {
    event_id: `evt_${Date.now()}`,
    event_type: "txn.created",
    event_ts: new Date().toISOString(),
    user_id: Number(document.getElementById("user_id").value),
    merchant_id: Number(document.getElementById("merchant_id").value),
    device_id: Number(document.getElementById("device_id").value),
    ip_id: Number(document.getElementById("ip_id").value),
    amount: Number(document.getElementById("amount").value),
    currency: document.getElementById("currency").value,
    country: document.getElementById("country").value,
    channel: document.getElementById("channel").value,
    drift_phase: Number(document.getElementById("drift_phase").value),
  };
}

function updateLatestResult(data) {
  setDecisionBadge(data.decision);
  const finalScore = document.getElementById("finalScore");
  if (finalScore) finalScore.textContent = data.final_score?.toFixed(4) ?? "-";
  const rulesScore = document.getElementById("rulesScore");
  if (rulesScore) rulesScore.textContent = data.scores?.rules?.toFixed(4) ?? "-";
  const championScore = document.getElementById("championScore");
  if (championScore) {
    championScore.textContent =
      data.scores?.champion !== null && data.scores?.champion !== undefined
        ? data.scores.champion.toFixed(4)
        : "-";
  }
  const anomalyScore = document.getElementById("anomalyScore");
  if (anomalyScore) {
    anomalyScore.textContent =
      data.scores?.anomaly !== null && data.scores?.anomaly !== undefined
        ? data.scores.anomaly.toFixed(4)
        : "-";
  }
  const fallbacksList = document.getElementById("fallbacksList");
  if (fallbacksList) {
    fallbacksList.textContent =
      (data.fallbacks && Object.keys(data.fallbacks).filter((k) => data.fallbacks[k]).join(", ")) ||
      "none";
  }
  const latencyTotal = document.getElementById("latencyTotal");
  if (latencyTotal) latencyTotal.textContent = data.latency_ms?.total?.toFixed(2) ?? "-";
  const championBadge = document.getElementById("championBadge");
  if (championBadge) {
    championBadge.textContent = `champion: ${data.model_versions?.champion_type || "-"}`;
  }
  const registryBadge = document.getElementById("registryBadge");
  if (registryBadge) {
    registryBadge.textContent = `registry: ${data.model_versions?.registry_mode || "-"}`;
  }
  const championRef = document.getElementById("championRef");
  if (championRef) championRef.textContent = data.model_versions?.champion_ref || "-";

  const featureTable = document.getElementById("featureTable");
  if (featureTable) {
    featureTable.innerHTML = "";
  }
  const features = data.feature_snapshot || {};
  Object.keys(features).forEach((key) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td class="py-1 pr-4 text-slate-500">${key}</td><td class="py-1">${features[key]}</td>`;
    if (featureTable) {
      featureTable.appendChild(row);
    }
  });
  const featureWarning = document.getElementById("featureWarning");
  if (featureWarning) {
    featureWarning.classList.toggle("hidden", !data.feast_failed);
  }
  const feastStatus = document.getElementById("feastStatus");
  if (feastStatus) feastStatus.textContent = data.feast_failed ? "feast failed" : "feast ok";
}

async function score() {
  if (scoreError) {
    scoreError.classList.add("hidden");
  }
  try {
    const payload = collectPayload();
    const resp = await fetch("/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      throw new Error(`Request failed (${resp.status})`);
    }
    const data = await resp.json();
    updateLatestResult(data);
  } catch (err) {
    if (scoreError) {
      scoreError.textContent = err.message;
      scoreError.classList.remove("hidden");
    }
  }
}

async function pollStats() {
  try {
    const resp = await fetch("/api/stats");
    const data = await resp.json();
    const reqTotal = document.getElementById("reqTotal");
    if (reqTotal) reqTotal.textContent = data.counters?.requests_total ?? "-";
    const errTotal = document.getElementById("errTotal");
    if (errTotal) errTotal.textContent = data.counters?.errors_total ?? "-";
    const feastOk = document.getElementById("feastOk");
    if (feastOk) feastOk.textContent = data.feast?.ok ? "yes" : "no";
    const redisOk = document.getElementById("redisOk");
    if (redisOk) redisOk.textContent = data.redis?.ok ? "yes" : "no";
    const fallbackCounts = document.getElementById("fallbackCounts");
    if (fallbackCounts) {
      fallbackCounts.textContent = JSON.stringify(data.counters?.fallbacks_total || {});
    }
  } catch (err) {
    // ignore
  }
}

async function pollDecisions() {
  try {
    const resp = await fetch("/api/recent-decisions?limit=20");
    const data = await resp.json();
    const counts = { approve: 0, step_up: 0, review: 0, block: 0 };
    const latencies = [];
    data.forEach((row) => {
      if (row.decision && counts[row.decision] !== undefined) {
        counts[row.decision] += 1;
      }
      if (row.latency_ms !== null && row.latency_ms !== undefined) {
        latencies.push(row.latency_ms);
      }
    });
    const total = Math.max(1, data.length);
    const distApprove = document.getElementById("distApprove");
    if (distApprove) distApprove.style.width = `${(counts.approve / total) * 100}%`;
    const distStepUp = document.getElementById("distStepUp");
    if (distStepUp) distStepUp.style.width = `${(counts.step_up / total) * 100}%`;
    const distReview = document.getElementById("distReview");
    if (distReview) distReview.style.width = `${(counts.review / total) * 100}%`;
    const distBlock = document.getElementById("distBlock");
    if (distBlock) distBlock.style.width = `${(counts.block / total) * 100}%`;
    const distCounts = document.getElementById("distCounts");
    if (distCounts) {
      distCounts.textContent = `A:${counts.approve} S:${counts.step_up} R:${counts.review} B:${counts.block}`;
    }

    if (latencies.length > 0) {
      latencies.sort((a, b) => a - b);
      const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const p95 = latencies[Math.floor(latencies.length * 0.95)];
      const latencySummary = document.getElementById("latencySummary");
      if (latencySummary) {
        latencySummary.textContent = `${avg.toFixed(2)} / ${p95.toFixed(2)} ms`;
      }
    } else {
      const latencySummary = document.getElementById("latencySummary");
      if (latencySummary) latencySummary.textContent = "-";
    }

    const table = document.getElementById("decisionTable");
    if (table) {
      table.innerHTML = "";
    }
    const rows = data.slice().reverse();
    if (!SELECTED_EVENT_ID && rows.length > 0) {
      SELECTED_EVENT_ID = rows[0].event_id || null;
    }
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="py-1">${row.ts?.slice(11, 19) || "-"}</td>
        <td class="py-1">${row.event_id || "-"}</td>
        <td class="py-1">${row.user_id || "-"}</td>
        <td class="py-1">${row.amount ?? "-"}</td>
        <td class="py-1">${row.decision || "-"}</td>
        <td class="py-1">${row.final_score?.toFixed(3) ?? "-"}</td>
      `;
      tr.dataset.eventId = row.event_id || "";
      tr.classList.toggle("bg-slate-100", row.event_id && row.event_id === SELECTED_EVENT_ID);
      tr.addEventListener("click", () => {
        SELECTED_EVENT_ID = row.event_id || null;
        const allRows = table?.querySelectorAll("tr") || [];
        allRows.forEach((r) => r.classList.remove("bg-slate-100"));
        tr.classList.add("bg-slate-100");
      });
      if (table) {
        table.appendChild(tr);
      }
    });
  } catch (err) {
    // ignore
  }
}

randomBtn?.addEventListener("click", fillRandom);
scoreBtn?.addEventListener("click", score);
if (intuitionBtn) {
  intuitionBtn.addEventListener("click", async () => {
    refreshIntuitionNodes();
    if (!intuitionModal || !intuitionClose || !intuitionContent) {
      return;
    }
    intuitionModal.classList.remove("hidden");
    try {
      const resp = await fetch("/reports/latest/model_intuition.json");
      const data = await resp.json();
      const models = data.models || [];
      intuitionContent.innerHTML = models
        .map((m) => {
          const feats = (m.top_features || [])
            .slice(0, 5)
            .map((f) => `<li>${f.feature}: ${f.importance ?? f.mean_abs_shap ?? ""}</li>`)
            .join("");
          const thresholds = m.thresholds
            ? `<pre class="bg-slate-50 p-2 rounded">${JSON.stringify(m.thresholds, null, 2)}</pre>`
            : "";
          return `
            <div class="border rounded p-3">
              <div class="font-semibold">${m.title}</div>
              <div class="text-slate-600">${m.summary || ""}</div>
              ${feats ? `<ul class="list-disc ml-5 mt-2">${feats}</ul>` : ""}
              ${thresholds}
            </div>
          `;
        })
        .join("");
    } catch (err) {
      intuitionContent.textContent = "Intuition file not found. Run: python -m src.demo.model_intuition";
    }
  });
}

document.addEventListener("click", (event) => {
  if (event.target?.id !== "intuitionClose") return;
  refreshIntuitionNodes();
  intuitionModal?.classList.add("hidden");
});

function refreshIntuitionNodes() {
  intuitionModal = document.getElementById("intuitionModal");
  intuitionClose = document.getElementById("intuitionClose");
  intuitionContent = document.getElementById("intuitionContent");
}

function renderLiveExplainability(modelId) {
  if (!SELECTED_EVENT_ID) {
    return `
      <div class="mt-4 border-t pt-4 text-sm text-slate-600">
        Select a transaction from Recent Decisions to see live explainability.
      </div>
    `;
  }
  return `
    <div class="mt-4 border-t pt-4 text-sm text-slate-600" id="liveExplainContainer">
      Loading live explainability…
    </div>
  `;
}

async function loadLiveExplainability(modelId) {
  if (!SELECTED_EVENT_ID) return;
  const container = document.getElementById("liveExplainContainer");
  if (!container) return;
  try {
    const resp = await fetch(`/api/txns/${encodeURIComponent(SELECTED_EVENT_ID)}/explain?top_k=5`);
    if (!resp.ok) throw new Error("Live explain unavailable");
    const data = await resp.json();
    if (data.status !== "ok") throw new Error("Live explain unavailable");

    const rulesList = (data.rules?.triggered || [])
      .map((item) => `<li>${escapeHtml(item)}</li>`)
      .join("");
    const gbdtList = (data.gbdt?.top_contributors || [])
      .map(
        (item) =>
          `<li>${escapeHtml(item.feature)}: ${Number(item.shap).toFixed(4)} (${escapeHtml(
            item.direction || ""
          )})</li>`
      )
      .join("");
    const anomalyList = (data.anomaly?.reasons || [])
      .map((item) => `<li>${escapeHtml(item)}</li>`)
      .join("");

    const showRules = modelId === "rules_engine" || modelId === "ensemble";
    const showGbdt = modelId === "gbdt" || modelId === "ensemble";
    const showAnomaly = modelId === "anomaly" || modelId === "ensemble";

    container.innerHTML = `
      <div class="font-semibold text-slate-800">For this transaction</div>
      <div class="text-sm text-slate-700 mt-1">Decision: ${escapeHtml(
        data.decision || "-"
      )} | Final score: ${Number(data.final_score ?? 0).toFixed(4)}</div>
      ${
        showRules
          ? `
        <div class="mt-2">
          <div class="font-medium text-sm">Rules tripwires</div>
          ${rulesList ? `<ul class="list-disc ml-5 text-sm">${rulesList}</ul>` : `<div class="text-sm">None</div>`}
        </div>`
          : ""
      }
      ${
        showGbdt
          ? `
        <div class="mt-2">
          <div class="font-medium text-sm">GBDT top drivers (SHAP)</div>
          ${gbdtList ? `<ul class="list-disc ml-5 text-sm">${gbdtList}</ul>` : `<div class="text-sm">Unavailable</div>`}
        </div>`
          : ""
      }
      ${
        showAnomaly
          ? `
        <div class="mt-2">
          <div class="font-medium text-sm">Anomaly signals</div>
          <div class="text-sm">Score: ${Number(data.anomaly?.score ?? 0).toFixed(4)}</div>
          ${anomalyList ? `<ul class="list-disc ml-5 text-sm">${anomalyList}</ul>` : `<div class="text-sm">None</div>`}
        </div>`
          : ""
      }
    `;
  } catch (err) {
    container.textContent = "Live explain unavailable.";
  }
}

window.openIntuition = async (modelId) => {
  refreshIntuitionNodes();
  if (!intuitionModal || !intuitionContent) {
    return;
  }
  try {
    const resp = await fetch(`/reports/latest/explainability_popups.json?v=${encodeURIComponent(POPUPS_VERSION)}`);
    if (!resp.ok) {
      throw new Error(`Popup fetch failed (${resp.status})`);
    }
    const popups = await resp.json();
    const spec = popups.explainability_popups?.[modelId];
    if (!spec) {
      return;
    }
    const title = intuitionModal.querySelector("h3");
    if (title) title.textContent = spec.title || "Model";
    intuitionContent.innerHTML =
      renderStaticSections(spec.sections || []) + renderLiveExplainability(modelId);
    intuitionModal.classList.remove("hidden");
    await loadLiveExplainability(modelId);
  } catch (err) {
    intuitionContent.textContent = "Explainability popups not found.";
    intuitionModal.classList.remove("hidden");
  }
};

document.addEventListener("click", (event) => {
  const btn = event.target.closest?.(".modelInfo");
  if (!btn) return;
  window.openIntuition?.(btn.dataset.model);
});

const infoButtons = [
  document.getElementById("infoEnsemble"),
  document.getElementById("infoRules"),
  document.getElementById("infoGbdt"),
  document.getElementById("infoAnomaly"),
];
infoButtons.forEach((btn) => {
  if (!btn) return;
  btn.addEventListener("click", (event) => {
    event.stopPropagation();
    window.openIntuition?.(btn.dataset.model);
  });
});

fillRandom();
pollStats();
pollDecisions();
setInterval(pollStats, 2000);
setInterval(pollDecisions, 2000);
