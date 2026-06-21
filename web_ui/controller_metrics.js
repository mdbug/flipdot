const summaryEl = document.getElementById("metricsSummary");
const eventsEl = document.getElementById("metricsEvents");
const updatedEl = document.getElementById("metricsUpdated");
const connectionRangeEl = document.getElementById("connectionRange");
const libraryWarningEl = document.getElementById("metricsLibraryWarning");
const resetZoomBtn = document.getElementById("metricsResetZoom");
const windowPresetsEl = document.getElementById("metricsWindowPresets");
const signalPanelEl = document.getElementById("signalPanel");
const connectionParamsPanelEl = document.getElementById("connectionParamsPanel");
const disconnectReasonPanelEl = document.getElementById("disconnectReasonPanel");
const disconnectReasonHistogramEl = document.getElementById("disconnectReasonHistogram");
const chartEls = {
  connection: document.getElementById("connectionChart"),
  freshness: document.getElementById("freshnessChart"),
  button: document.getElementById("buttonChart"),
  signal: document.getElementById("signalChart"),
  connectionParams: document.getElementById("connectionParamsChart"),
  battery: document.getElementById("batteryChart"),
};

const COLORS = ["#2ab7a9", "#ef6b55", "#7fb069", "#e2bf52"];
const BUTTONS = ["A", "B", "X", "Y", "L1", "R1", "L2", "R2", "Start", "Select", "D-Up", "D-Down", "D-Left", "D-Right", "Home"];
const charts = {};
const MIN_WINDOW_SEC = 10;
const WHEEL_ZOOM_FACTOR = 1.2;
const WINDOW_PRESETS = [
  { label: "30s", seconds: 30 },
  { label: "2m", seconds: 120 },
  { label: "10m", seconds: 600 },
  { label: "30m", seconds: 1800 },
  { label: "1h", seconds: 3600 },
];
let selectedWindowSec = null;
let latestMetrics = null;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function fmtTime(timestamp) {
  if (!Number.isFinite(timestamp)) {
    return "--";
  }
  return new Date(timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function fmtDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0s";
  }
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  if (seconds < 3600) {
    return `${Math.round(seconds / 60)}m`;
  }
  return `${(seconds / 3600).toFixed(1)}h`;
}

function fmtMs(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  if (value < 1000) {
    return `${Math.round(value)}ms`;
  }
  return `${(value / 1000).toFixed(1)}s`;
}

function fmtRate(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  return `${Number(value).toFixed(2)}/s`;
}

function fmtIdleMs(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  if (value < 1000) {
    return "<1s";
  }
  return fmtDuration(Number(value) / 1000);
}

function metricNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function normalizeMetrics(payload) {
  return {
    generated_at: Number(payload.generated_at),
    window_sec: Number(payload.window_sec) || 3600,
    controllers: Array.isArray(payload.controllers) ? payload.controllers : [],
    samples: Array.isArray(payload.samples) ? payload.samples : [],
    events: Array.isArray(payload.events) ? payload.events : [],
    button_events: Array.isArray(payload.button_events) ? payload.button_events : [],
    panel_latency_events: Array.isArray(payload.panel_latency_events) ? payload.panel_latency_events : [],
  };
}

function controllerKeys(metrics) {
  const keys = [];
  for (const controller of metrics.controllers) {
    const key = String(controller.key || "");
    if (key && !keys.includes(key)) {
      keys.push(key);
    }
  }
  for (const sample of metrics.samples) {
    for (const status of sample.controllers || []) {
      const key = String(status.key || "");
      if (key && !keys.includes(key)) {
        keys.push(key);
      }
    }
  }
  return keys;
}

function controllerLabel(metrics, key, index) {
  const summary = metrics.controllers.find((item) => item.key === key);
  return summary && summary.label ? summary.label : `P${index + 1}`;
}

function sampleStatus(sample, key) {
  return (sample.controllers || []).find((item) => item.key === key) || null;
}

function chartAvailable() {
  return typeof window.Chart === "function";
}

function effectiveWindowSec(metrics) {
  const maxWindowSec = Number(metrics.window_sec) > 0 ? Number(metrics.window_sec) : 3600;
  if (!Number.isFinite(selectedWindowSec)) {
    return maxWindowSec;
  }
  return clamp(selectedWindowSec, MIN_WINDOW_SEC, maxWindowSec);
}

function renderWindowPresets(metrics) {
  if (!windowPresetsEl) {
    return;
  }
  const maxWindowSec = Number(metrics.window_sec) > 0 ? Number(metrics.window_sec) : 3600;
  const activeWindowSec = effectiveWindowSec(metrics);
  windowPresetsEl.innerHTML = "";
  for (const preset of WINDOW_PRESETS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "metrics-preset-btn";
    button.textContent = preset.label;
    button.disabled = preset.seconds > maxWindowSec;
    if (Math.abs(activeWindowSec - preset.seconds) < 0.5) {
      button.classList.add("active");
      button.setAttribute("aria-pressed", "true");
    } else {
      button.setAttribute("aria-pressed", "false");
    }
    button.addEventListener("click", () => {
      selectedWindowSec = clamp(preset.seconds, MIN_WINDOW_SEC, maxWindowSec);
      renderMetricsFromNormalized(metrics);
    });
    windowPresetsEl.appendChild(button);
  }
}

function chartBounds(metrics) {
  const now = Number.isFinite(metrics.generated_at) ? metrics.generated_at : Date.now() / 1000;
  const start = now - effectiveWindowSec(metrics);
  return { min: start * 1000, max: now * 1000 };
}

function timeScale(metrics) {
  const bounds = chartBounds(metrics);
  return {
    type: "linear",
    min: bounds.min,
    max: bounds.max,
    grid: { color: "#343841" },
    ticks: {
      color: "#a8adb7",
      callback: (value) => fmtTime(Number(value) / 1000),
      maxTicksLimit: 8,
    },
  };
}

function baseOptions(metrics, yOptions) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    parsing: false,
    interaction: { mode: "nearest", intersect: false },
    plugins: {
      legend: { labels: { color: "#f3efe6", boxWidth: 12, boxHeight: 8 } },
      tooltip: {
        callbacks: {
          title: (items) => (items.length > 0 ? fmtTime(Number(items[0].parsed.x) / 1000) : ""),
        },
      },
    },
    scales: {
      x: timeScale(metrics),
      y: yOptions,
    },
  };
}

function upsertChart(name, canvas, type, data, options) {
  const nextFullBounds = {
    min: Number(options.scales.x.min),
    max: Number(options.scales.x.max),
  };
  if (!charts[name]) {
    charts[name] = new Chart(canvas, { type, data, options });
    charts[name].$metricsFullBounds = nextFullBounds;
    return;
  }
  const chart = charts[name];
  chart.data.datasets = data.datasets;
  chart.options.scales = options.scales;
  chart.options.scales.x.min = nextFullBounds.min;
  chart.options.scales.x.max = nextFullBounds.max;
  chart.$metricsFullBounds = nextFullBounds;
  chart.update("none");
}

function point(timestamp, value, extra = {}) {
  return { x: Number(timestamp) * 1000, y: value, ...extra };
}

function lineDatasets(metrics, valueGetter, options = {}) {
  const keys = controllerKeys(metrics);
  return keys.map((key, index) => ({
    label: `${controllerLabel(metrics, key, index)} ${options.labelSuffix || ""}`.trim(),
    data: metrics.samples
      .map((sample) => {
        const status = sampleStatus(sample, key);
        if (!status) {
          return null;
        }
        const value = metricNumber(valueGetter(status));
        return value !== null ? point(sample.timestamp, value) : null;
      })
      .filter(Boolean),
    borderColor: options.color ? options.color(index) : COLORS[index % COLORS.length],
    backgroundColor: options.color ? options.color(index) : COLORS[index % COLORS.length],
    stepped: Boolean(options.stepped),
    tension: options.stepped ? 0 : 0.18,
    borderWidth: 2,
    pointRadius: options.pointRadius ?? 0,
    spanGaps: false,
  }));
}

function renderConnectionChart(metrics) {
  const data = { datasets: lineDatasets(metrics, (status) => (status.connected ? 1 : 0), { stepped: true, pointRadius: 2 }) };
  const options = baseOptions(metrics, {
    min: -0.05,
    max: 1.05,
    grid: { color: "#343841" },
    ticks: { color: "#a8adb7", callback: (value) => (Number(value) >= 0.5 ? "connected" : "down") },
  });
  upsertChart("connection", chartEls.connection, "line", data, options);
}

function buildButtonRateSeries(metrics, key, windowSec = 10) {
  const sampleTimestamps = metrics.samples.map((sample) => Number(sample.timestamp)).filter(Number.isFinite);
  if (sampleTimestamps.length === 0) {
    return [];
  }

  const eventTimestamps = metrics.button_events
    .filter((event) => event.key === key)
    .map((event) => Number(event.timestamp))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);

  const points = [];
  let left = 0;
  let right = 0;
  for (const t of sampleTimestamps) {
    while (right < eventTimestamps.length && eventTimestamps[right] <= t) {
      right += 1;
    }
    while (left < right && eventTimestamps[left] < t - windowSec) {
      left += 1;
    }
    const count = right - left;
    points.push(point(t, count / windowSec));
  }
  return points;
}

function renderActivityRateChart(metrics) {
  const keys = controllerKeys(metrics);
  const datasets = keys.map((key, index) => ({
    label: `${controllerLabel(metrics, key, index)} activity`,
    data: buildButtonRateSeries(metrics, key, 10),
    borderColor: COLORS[index % COLORS.length],
    backgroundColor: COLORS[index % COLORS.length],
    tension: 0.18,
    borderWidth: 2,
    pointRadius: 0,
    spanGaps: false,
  }));
  const data = { datasets };

  const options = baseOptions(metrics, {
    beginAtZero: true,
    suggestedMax: 4,
    grid: { color: "#343841" },
    ticks: { color: "#a8adb7", callback: (value) => fmtRate(Number(value)) },
  });
  options.plugins.tooltip.callbacks.label = (context) => `${context.dataset.label}: ${fmtRate(Number(context.parsed.y))}`;
  upsertChart("freshness", chartEls.freshness, "line", data, options);
}

function renderBatteryChart(metrics) {
  const data = { datasets: lineDatasets(metrics, (status) => status.battery_percentage, { labelSuffix: "battery" }) };
  const options = baseOptions(metrics, {
    min: 0,
    max: 100,
    grid: { color: "#343841" },
    ticks: { color: "#a8adb7", callback: (value) => `${value}%` },
  });
  upsertChart("battery", chartEls.battery, "line", data, options);
}

function renderSignalChart(metrics) {
  const keys = controllerKeys(metrics);
  const datasets = [];
  let signalPointCount = 0;
  keys.forEach((key, index) => {
    const baseColor = COLORS[index % COLORS.length];
    const rssiData = metrics.samples
      .map((sample) => {
        const status = sampleStatus(sample, key);
        const value = status ? metricNumber(status.rssi_dbm) : null;
        return value !== null ? point(sample.timestamp, value) : null;
      })
      .filter(Boolean);
    const txPowerData = metrics.samples
      .map((sample) => {
        const status = sampleStatus(sample, key);
        const value = status ? metricNumber(status.tx_power_dbm) : null;
        return value !== null ? point(sample.timestamp, value) : null;
      })
      .filter(Boolean);
    signalPointCount += rssiData.length + txPowerData.length;
    datasets.push({
      label: `${controllerLabel(metrics, key, index)} RSSI`,
      data: rssiData,
      borderColor: baseColor,
      backgroundColor: baseColor,
      borderWidth: 2,
      pointRadius: 2,
      tension: 0.18,
      spanGaps: false,
    });
    datasets.push({
      label: `${controllerLabel(metrics, key, index)} TX power`,
      data: txPowerData,
      borderColor: baseColor,
      backgroundColor: baseColor,
      borderDash: [5, 4],
      borderWidth: 2,
      pointRadius: 1,
      tension: 0.18,
      spanGaps: false,
    });
  });
  if (signalPanelEl) {
    signalPanelEl.classList.toggle("hidden", signalPointCount === 0);
  }
  if (signalPointCount === 0) {
    if (charts.signal) {
      charts.signal.destroy();
      delete charts.signal;
    }
    return;
  }
  const options = baseOptions(metrics, {
    suggestedMin: -100,
    suggestedMax: 10,
    grid: { color: "#343841" },
    ticks: { color: "#a8adb7", callback: (value) => `${value} dBm` },
  });
  upsertChart("signal", chartEls.signal, "line", { datasets }, options);
}

function renderConnectionParamsChart(metrics) {
  const keys = controllerKeys(metrics);
  const datasets = [];
  let pointCount = 0;
  keys.forEach((key, index) => {
    const baseColor = COLORS[index % COLORS.length];
    const intervalData = metrics.samples
      .map((sample) => {
        const status = sampleStatus(sample, key);
        const value = status ? metricNumber(status.connection_interval_ms) : null;
        return value !== null ? point(sample.timestamp, value) : null;
      })
      .filter(Boolean);
    const timeoutData = metrics.samples
      .map((sample) => {
        const status = sampleStatus(sample, key);
        const value = status ? metricNumber(status.supervision_timeout_ms) : null;
        return value !== null ? point(sample.timestamp, value) : null;
      })
      .filter(Boolean);
    pointCount += intervalData.length + timeoutData.length;
    datasets.push({
      label: `${controllerLabel(metrics, key, index)} interval`,
      data: intervalData,
      borderColor: baseColor,
      backgroundColor: baseColor,
      borderWidth: 2,
      pointRadius: 1,
      tension: 0.18,
      spanGaps: false,
      yAxisID: "y",
    });
    datasets.push({
      label: `${controllerLabel(metrics, key, index)} supervision`,
      data: timeoutData,
      borderColor: baseColor,
      backgroundColor: baseColor,
      borderDash: [5, 4],
      borderWidth: 2,
      pointRadius: 1,
      tension: 0.18,
      spanGaps: false,
      yAxisID: "y2",
    });
  });

  if (connectionParamsPanelEl) {
    connectionParamsPanelEl.classList.toggle("hidden", pointCount === 0);
  }
  if (pointCount === 0) {
    if (charts.connectionParams) {
      charts.connectionParams.destroy();
      delete charts.connectionParams;
    }
    return;
  }

  const options = {
    ...baseOptions(metrics, {
      beginAtZero: true,
      suggestedMax: 60,
      grid: { color: "#343841" },
      ticks: { color: "#a8adb7", callback: (value) => `${Math.round(Number(value))}ms` },
    }),
  };
  options.scales.y2 = {
    position: "right",
    beginAtZero: true,
    suggestedMax: 6000,
    grid: { drawOnChartArea: false, color: "#343841" },
    ticks: { color: "#a8adb7", callback: (value) => `${Math.round(Number(value))}ms` },
  };
  upsertChart("connectionParams", chartEls.connectionParams, "line", { datasets }, options);
}

function renderDisconnectReasons(metrics) {
  if (!disconnectReasonHistogramEl) {
    return;
  }
  const totals = {};
  for (const controller of metrics.controllers) {
    const counts = controller.disconnect_reason_counts || {};
    for (const [code, countRaw] of Object.entries(counts)) {
      const count = Number(countRaw);
      if (!Number.isFinite(count) || count <= 0) {
        continue;
      }
      totals[code] = (totals[code] || 0) + count;
    }
  }

  const rows = Object.entries(totals).sort((a, b) => Number(b[1]) - Number(a[1]));
  if (disconnectReasonPanelEl) {
    disconnectReasonPanelEl.classList.toggle("hidden", rows.length === 0);
  }
  disconnectReasonHistogramEl.innerHTML = "";
  if (rows.length === 0) {
    return;
  }

  const maxCount = Math.max(...rows.map(([, count]) => Number(count)));
  for (const [code, count] of rows) {
    const row = document.createElement("div");
    row.className = "metrics-event";
    const widthPct = maxCount > 0 ? (Number(count) / maxCount) * 100 : 0;
    row.innerHTML = `
      <strong>${code}</strong>
      <span>${count}</span>
      <span style="flex:1; margin-left:0.5rem; height:0.4rem; border-radius:999px; background:#283035; overflow:hidden;">
        <span style="display:block; width:${widthPct.toFixed(1)}%; height:100%; background:#ef6b55;"></span>
      </span>
    `;
    disconnectReasonHistogramEl.appendChild(row);
  }
}

function renderButtonChart(metrics) {
  const keys = controllerKeys(metrics);
  const datasets = keys.map((key, index) => ({
    label: `${controllerLabel(metrics, key, index)} buttons`,
    data: metrics.button_events
      .filter((event) => event.key === key)
      .map((event) => {
        const button = String(event.button || "");
        let y = BUTTONS.indexOf(button);
        if (y < 0) {
          y = BUTTONS.length;
        }
        return point(event.timestamp, y, { button, event: event.event });
      }),
    borderColor: COLORS[index % COLORS.length],
    backgroundColor: COLORS[index % COLORS.length],
    pointRadius: 5,
    pointHoverRadius: 7,
  }));
  const options = baseOptions(metrics, {
    min: -0.5,
    max: BUTTONS.length + 0.5,
    grid: { color: "#343841" },
    ticks: {
      color: "#a8adb7",
      stepSize: 1,
      callback: (value) => BUTTONS[Number(value)] || "Other",
    },
  });
  options.plugins.tooltip.callbacks.label = (context) => {
    const raw = context.raw || {};
    return `${context.dataset.label}: ${raw.button || "button"} ${raw.event || ""}`.trim();
  };
  upsertChart("button", chartEls.button, "scatter", { datasets }, options);
}

function renderSummary(metrics) {
  summaryEl.innerHTML = "";
  if (metrics.controllers.length === 0) {
    const empty = document.createElement("p");
    empty.className = "system-note";
    empty.textContent = "No controller metrics collected yet.";
    summaryEl.appendChild(empty);
    return;
  }

  for (const controller of metrics.controllers) {
    const item = document.createElement("article");
    item.className = "metrics-summary-card";
    const latest = controller.latest || {};
    const connected = Boolean(latest.connected);
    const ratio = metricNumber(controller.connected_ratio);
    const uptime = Number.isFinite(ratio) ? `${Math.round(ratio * 100)}%` : "--";
    const disconnectsPerHour = metricNumber(controller.disconnects_per_hour);
    const mttrSec = metricNumber(controller.mttr_sec);
    const latencyP50 = metricNumber(controller.panel_latency_p50_ms);
    const latencyP95 = metricNumber(controller.panel_latency_p95_ms);
    const latencyP99 = metricNumber(controller.panel_latency_p99_ms);
    const latencySamples = metricNumber(controller.panel_latency_samples);
    const reconnectAttempts = metricNumber(latest.bluetooth_connect_attempts);
    const reconnectFailures = metricNumber(latest.bluetooth_connect_failures);
    const reconnectFailRate = (
      Number.isFinite(reconnectAttempts) && reconnectAttempts > 0 && Number.isFinite(reconnectFailures)
        ? `${Math.round((reconnectFailures / reconnectAttempts) * 100)}%`
        : "--"
    );
    const batterySourceRaw = String(latest.battery_source || "").trim();
    const batterySource = batterySourceRaw || "--";
    const batteryAgeMs = metricNumber(latest.battery_age_ms);
    const batteryPollMs = metricNumber(latest.battery_poll_duration_ms);
    const bluetoothPollMs = metricNumber(latest.bluetooth_metrics_poll_duration_ms);
    const rssi = metricNumber(latest.rssi_dbm);
    const txPower = metricNumber(latest.tx_power_dbm);
    const linkQuality = metricNumber(latest.link_quality);
    const signalSourceRaw = String(latest.signal_source || "").trim();
    const signalSource = signalSourceRaw || "--";
    const connIntervalMs = metricNumber(latest.connection_interval_ms);
    const connLatency = metricNumber(latest.connection_latency);
    const supervisionTimeoutMs = metricNumber(latest.supervision_timeout_ms);
    const connParamsSourceRaw = String(latest.connection_params_source || "").trim();
    const connParamsSource = connParamsSourceRaw || "--";
    const battery = metricNumber(latest.battery_percentage);
    item.innerHTML = `
      <p class="eyebrow">${controller.label || "Controller"}</p>
      <h2>${connected ? "Connected" : "Disconnected"}</h2>
      <dl>
        <div><dt>Address</dt><dd>${controller.address || "--"}</dd></div>
        <div><dt>Disconnects</dt><dd>${controller.disconnects || 0}</dd></div>
        <div><dt>Reconnects</dt><dd>${controller.reconnects || 0}</dd></div>
        <div><dt>Disconnects/h</dt><dd>${disconnectsPerHour !== null ? disconnectsPerHour.toFixed(2) : "--"}</dd></div>
        <div><dt>Mean recovery</dt><dd>${mttrSec !== null ? fmtDuration(mttrSec) : "--"}</dd></div>
        <div><dt>Panel latency p50</dt><dd>${fmtMs(latencyP50)}</dd></div>
        <div><dt>Panel latency p95</dt><dd>${fmtMs(latencyP95)}</dd></div>
        <div><dt>Panel latency p99</dt><dd>${fmtMs(latencyP99)}</dd></div>
        <div><dt>Latency samples</dt><dd>${latencySamples !== null ? latencySamples : "--"}</dd></div>
        <div><dt>Button events</dt><dd>${controller.button_event_count || 0}</dd></div>
        <div><dt>Connected samples</dt><dd>${uptime}</dd></div>
        <div><dt>Idle since input</dt><dd>${fmtIdleMs(latest.last_event_age_ms)}</dd></div>
        <div><dt>Reconnect attempts</dt><dd>${reconnectAttempts !== null ? reconnectAttempts : "--"}</dd></div>
        <div><dt>Reconnect fail rate</dt><dd>${reconnectFailRate}</dd></div>
        <div><dt>Battery</dt><dd>${battery !== null ? `${battery}%` : "--"}</dd></div>
        <div><dt>Battery source</dt><dd>${batterySource}</dd></div>
        <div><dt>Battery age</dt><dd>${fmtIdleMs(batteryAgeMs)}</dd></div>
        <div><dt>Battery poll</dt><dd>${fmtMs(batteryPollMs)}</dd></div>
        <div><dt>BT info poll</dt><dd>${fmtMs(bluetoothPollMs)}</dd></div>
        <div><dt>Signal source</dt><dd>${signalSource}</dd></div>
        <div><dt>Conn params source</dt><dd>${connParamsSource}</dd></div>
        <div><dt>Conn interval</dt><dd>${connIntervalMs !== null ? `${connIntervalMs}ms` : "--"}</dd></div>
        <div><dt>Conn latency</dt><dd>${connLatency !== null ? connLatency : "--"}</dd></div>
        <div><dt>Supervision timeout</dt><dd>${supervisionTimeoutMs !== null ? `${supervisionTimeoutMs}ms` : "--"}</dd></div>
        <div><dt>RSSI</dt><dd>${Number.isFinite(rssi) ? `${rssi} dBm` : "--"}</dd></div>
        <div><dt>TX power</dt><dd>${Number.isFinite(txPower) ? `${txPower} dBm` : "--"}</dd></div>
        <div><dt>Link quality</dt><dd>${Number.isFinite(linkQuality) ? linkQuality : "--"}</dd></div>
      </dl>
    `;
    summaryEl.appendChild(item);
  }
}

function renderEvents(metrics) {
  eventsEl.innerHTML = "";
  const connectionEvents = metrics.events.map((event) => ({ ...event, kind: "connection" }));
  const buttonEvents = metrics.button_events.slice(-80).map((event) => ({ ...event, kind: "button" }));
  const recent = [...connectionEvents, ...buttonEvents]
    .sort((a, b) => Number(b.timestamp) - Number(a.timestamp))
    .slice(0, 80);
  if (recent.length === 0) {
    const empty = document.createElement("p");
    empty.className = "system-note";
    empty.textContent = "No connection or button events recorded in this process yet.";
    eventsEl.appendChild(empty);
    return;
  }
  for (const event of recent) {
    const row = document.createElement("div");
    const isBad = event.event === "disconnected";
    row.className = `metrics-event ${isBad ? "bad" : "good"}`;
    if (event.kind === "button") {
      row.textContent = `${fmtTime(Number(event.timestamp))} ${event.label || "Controller"} ${event.button || "button"} ${event.event || ""}`;
    } else {
      row.textContent = `${fmtTime(Number(event.timestamp))} ${event.label || "Controller"} ${event.event} ${event.address || ""}`;
    }
    eventsEl.appendChild(row);
  }
}

function renderMetrics(payload) {
  renderMetricsFromNormalized(normalizeMetrics(payload));
}

function renderMetricsFromNormalized(metrics) {
  latestMetrics = metrics;
  renderSummary(metrics);
  renderEvents(metrics);
  renderDisconnectReasons(metrics);
  renderWindowPresets(metrics);
  connectionRangeEl.textContent = `${fmtDuration(effectiveWindowSec(metrics))} rolling window`;
  updatedEl.textContent = `Updated ${fmtTime(metrics.generated_at)}`;

  if (!chartAvailable()) {
    libraryWarningEl.classList.remove("hidden");
    return;
  }
  libraryWarningEl.classList.add("hidden");
  renderConnectionChart(metrics);
  renderActivityRateChart(metrics);
  renderButtonChart(metrics);
  renderSignalChart(metrics);
  renderConnectionParamsChart(metrics);
  renderBatteryChart(metrics);
}

function handleWheelZoom(event) {
  if (!latestMetrics) {
    return;
  }
  event.preventDefault();
  const currentWindowSec = effectiveWindowSec(latestMetrics);
  const maxWindowSec = Number(latestMetrics.window_sec) > 0 ? Number(latestMetrics.window_sec) : 3600;
  const factor = event.deltaY > 0 ? WHEEL_ZOOM_FACTOR : 1 / WHEEL_ZOOM_FACTOR;
  selectedWindowSec = clamp(currentWindowSec * factor, MIN_WINDOW_SEC, maxWindowSec);
  renderMetricsFromNormalized(latestMetrics);
}

async function loadMetrics() {
  try {
    const response = await fetch("/api/controller/metrics", { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    renderMetrics(await response.json());
  } catch (_err) {
    updatedEl.textContent = "Metrics unavailable";
  }
}

resetZoomBtn.addEventListener("click", () => {
  selectedWindowSec = null;
  if (latestMetrics) {
    renderMetricsFromNormalized(latestMetrics);
  }
});

for (const canvas of Object.values(chartEls)) {
  if (!canvas) {
    continue;
  }
  canvas.addEventListener("wheel", handleWheelZoom, { passive: false });
  canvas.style.touchAction = "pan-y";
}

loadMetrics();
window.setInterval(loadMetrics, 1000);