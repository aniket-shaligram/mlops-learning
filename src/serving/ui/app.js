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

function setDecisionBadge(decision) {
  const badge = document.getElementById("decisionBadge");
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
  document.getElementById("finalScore").textContent = data.final_score?.toFixed(4) ?? "-";
  document.getElementById("rulesScore").textContent = data.scores?.rules?.toFixed(4) ?? "-";
  document.getElementById("championScore").textContent =
    data.scores?.champion !== null && data.scores?.champion !== undefined
      ? data.scores.champion.toFixed(4)
      : "-";
  document.getElementById("anomalyScore").textContent =
    data.scores?.anomaly !== null && data.scores?.anomaly !== undefined
      ? data.scores.anomaly.toFixed(4)
      : "-";
  document.getElementById("fallbacksList").textContent =
    data.fallbacks && Object.keys(data.fallbacks).filter((k) => data.fallbacks[k]).join(", ") || "none";
  document.getElementById("latencyTotal").textContent = data.latency_ms?.total?.toFixed(2) ?? "-";
  document.getElementById("championBadge").textContent =
    `champion: ${data.model_versions?.champion_type || "-"}`;
  document.getElementById("registryBadge").textContent =
    `registry: ${data.model_versions?.registry_mode || "-"}`;
  document.getElementById("championRef").textContent = data.model_versions?.champion_ref || "-";

  const featureTable = document.getElementById("featureTable");
  featureTable.innerHTML = "";
  const features = data.feature_snapshot || {};
  Object.keys(features).forEach((key) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td class="py-1 pr-4 text-slate-500">${key}</td><td class="py-1">${features[key]}</td>`;
    featureTable.appendChild(row);
  });
  document.getElementById("featureWarning").classList.toggle("hidden", !data.feast_failed);
  document.getElementById("feastStatus").textContent = data.feast_failed ? "feast failed" : "feast ok";
}

async function score() {
  scoreError.classList.add("hidden");
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
    scoreError.textContent = err.message;
    scoreError.classList.remove("hidden");
  }
}

async function pollStats() {
  try {
    const resp = await fetch("/api/stats");
    const data = await resp.json();
    document.getElementById("reqTotal").textContent = data.counters?.requests_total ?? "-";
    document.getElementById("errTotal").textContent = data.counters?.errors_total ?? "-";
    document.getElementById("feastOk").textContent = data.feast?.ok ? "yes" : "no";
    document.getElementById("redisOk").textContent = data.redis?.ok ? "yes" : "no";
    document.getElementById("fallbackCounts").textContent = JSON.stringify(data.counters?.fallbacks_total || {});
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
    document.getElementById("distApprove").style.width = `${(counts.approve / total) * 100}%`;
    document.getElementById("distStepUp").style.width = `${(counts.step_up / total) * 100}%`;
    document.getElementById("distReview").style.width = `${(counts.review / total) * 100}%`;
    document.getElementById("distBlock").style.width = `${(counts.block / total) * 100}%`;
    document.getElementById("distCounts").textContent = `A:${counts.approve} S:${counts.step_up} R:${counts.review} B:${counts.block}`;

    if (latencies.length > 0) {
      latencies.sort((a, b) => a - b);
      const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const p95 = latencies[Math.floor(latencies.length * 0.95)];
      document.getElementById("latencySummary").textContent = `${avg.toFixed(2)} / ${p95.toFixed(2)} ms`;
    } else {
      document.getElementById("latencySummary").textContent = "-";
    }

    const table = document.getElementById("decisionTable");
    table.innerHTML = "";
    data.slice().reverse().forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="py-1">${row.ts?.slice(11, 19) || "-"}</td>
        <td class="py-1">${row.event_id || "-"}</td>
        <td class="py-1">${row.user_id || "-"}</td>
        <td class="py-1">${row.amount ?? "-"}</td>
        <td class="py-1">${row.decision || "-"}</td>
        <td class="py-1">${row.final_score?.toFixed(3) ?? "-"}</td>
      `;
      table.appendChild(tr);
    });
  } catch (err) {
    // ignore
  }
}

randomBtn.addEventListener("click", fillRandom);
scoreBtn.addEventListener("click", score);

fillRandom();
pollStats();
pollDecisions();
setInterval(pollStats, 2000);
setInterval(pollDecisions, 2000);
