const scoreBtn = document.getElementById("scoreBtn");
const randomBtn = document.getElementById("randomBtn");
const scoreError = document.getElementById("scoreError");
const driftPreset = document.getElementById("driftPreset");

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
  const servedBy = document.getElementById("servedBy");
  if (servedBy) servedBy.textContent = data.served_by || "-";
  const fallbacksList = document.getElementById("fallbacksList");
  if (fallbacksList) {
    fallbacksList.textContent =
      (data.fallbacks && Object.keys(data.fallbacks).filter((k) => data.fallbacks[k]).join(", ")) ||
      "none";
  }
  const latencyTotal = document.getElementById("latencyTotal");
  if (latencyTotal) latencyTotal.textContent = data.latency_ms?.total?.toFixed(2) ?? "-";
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
    const routerMode = document.getElementById("routerMode");
    if (routerMode) routerMode.textContent = data.router?.mode || "-";
    const canaryPercent = document.getElementById("canaryPercent");
    if (canaryPercent) canaryPercent.textContent = data.router?.canary_percent ?? "-";
    const v1Ok = document.getElementById("v1Ok");
    if (v1Ok) v1Ok.textContent = data.v1?.status === "ok" ? "yes" : "no";
    const v2Ok = document.getElementById("v2Ok");
    if (v2Ok) v2Ok.textContent = data.v2?.status === "ok" ? "yes" : "no";
  } catch (err) {
    // ignore
  }
}

async function pollShadow() {
  try {
    const resp = await fetch("/api/shadow-comparisons?limit=20");
    const data = await resp.json();
    const table = document.getElementById("shadowTable");
    if (table) {
      table.innerHTML = "";
    }
    data.slice().reverse().forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="py-1">${row.ts?.slice(11, 19) || "-"}</td>
        <td class="py-1">${row.user_id || "-"}</td>
        <td class="py-1">${row.v1_score?.toFixed?.(3) ?? "-"}</td>
        <td class="py-1">${row.v2_score?.toFixed?.(3) ?? "-"}</td>
        <td class="py-1">${row.score_delta?.toFixed?.(3) ?? "-"}</td>
        <td class="py-1">${row.decision_diff ? "yes" : "no"}</td>
      `;
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

fillRandom();
pollStats();
pollShadow();
setInterval(pollStats, 2000);
setInterval(pollShadow, 2000);
