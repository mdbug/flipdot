const canvas = document.getElementById("matrix");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("statusText");
const sourceText = document.getElementById("sourceText");
const controllerStatuses = document.getElementById("controllerStatuses");
const modeControls = document.getElementById("modeControls");
const boardEditor = document.getElementById("boardEditor");
const settingsToggle = document.getElementById("settingsToggle");
const sleepSettings = document.getElementById("sleepSettings");
const clockSettings = document.getElementById("clockSettings");
const clockDisplayStyle = document.getElementById("clockDisplayStyle");
const clockSeconds = document.getElementById("clockSeconds");
const clockSettingsStatus = document.getElementById("clockSettingsStatus");
const fontPreviewSettings = document.getElementById("fontPreviewSettings");
const sleepEnabled = document.getElementById("sleepEnabled");
const sleepStartHour = document.getElementById("sleepStartHour");
const sleepEndHour = document.getElementById("sleepEndHour");
const sleepSettingsStatus = document.getElementById("sleepSettingsStatus");
const poseChainEnabled = document.getElementById("poseChainEnabled");
const poseSettingsStatus = document.getElementById("poseSettingsStatus");
const fontPreviewPhrase = document.getElementById("fontPreviewPhrase");
const fontPreviewSpacing = document.getElementById("fontPreviewSpacing");
const fontPreviewStatus = document.getElementById("fontPreviewStatus");
const fontPreviewSlots = document.getElementById("fontPreviewSlots");

const boardText = document.getElementById("boardText");
const boardApplyText = document.getElementById("boardApplyText");
const boardTextAdd = document.getElementById("boardTextAdd");
const boardTextDelete = document.getElementById("boardTextDelete");
const boardTextObjectSelect = document.getElementById("boardTextObjectSelect");
const boardTextX = document.getElementById("boardTextX");
const boardTextY = document.getElementById("boardTextY");
const boardFontFamily = document.getElementById("boardFontFamily");
const boardFontSize = document.getElementById("boardFontSize");
const boardFontStyle = document.getElementById("boardFontStyle");
const boardGlyphSpacing = document.getElementById("boardGlyphSpacing");
const boardTextScroll = document.getElementById("boardTextScroll");
const boardScrollSpeed = document.getElementById("boardScrollSpeed");
const boardSelectionSummary = document.getElementById("boardSelectionSummary");
const boardDeleteSelection = document.getElementById("boardDeleteSelection");

const boardClear = document.getElementById("boardClear");
const boardUndo = document.getElementById("boardUndo");
const boardDrawToggle = document.getElementById("boardDrawToggle");
const boardTool = document.getElementById("boardTool");
const boardDrawLineWidth = document.getElementById("boardDrawLineWidth");
const boardDrawColor = document.getElementById("boardDrawColor");
const toolbarButtons = Array.from(document.querySelectorAll(".tool-btn[data-tool]"));
const shapeButtons = Array.from(document.querySelectorAll(".shape-btn[data-shape]"));
const contextBlocks = Array.from(document.querySelectorAll("#boardContext .context-block"));
const boardsMenuToggle = document.getElementById("boardsMenuToggle");
const boardsMenu = document.getElementById("boardsMenu");

const boardImageFile = document.getElementById("boardImageFile");
const boardImageMode = document.getElementById("boardImageMode");
const boardImageThreshold = document.getElementById("boardImageThreshold");
const boardImageX = document.getElementById("boardImageX");
const boardImageY = document.getElementById("boardImageY");
const boardUploadImage = document.getElementById("boardUploadImage");

const boardNameInput = document.getElementById("boardNameInput");
const boardList = document.getElementById("boardList");
const boardSaveNamed = document.getElementById("boardSaveNamed");
const boardLoadNamed = document.getElementById("boardLoadNamed");
const boardDeleteNamed = document.getElementById("boardDeleteNamed");

const DEFAULT_GRID = 28;
const HAS_POINTER_EVENTS = "PointerEvent" in window;
const DRAW_TOOLS = new Set(["freehand", "line", "rectangle", "circle"]);
const THEME = window.getComputedStyle(document.documentElement);

let lastVersion = -1;
let pollingTimer = null;
let controllerPollingTimer = null;
let lastTouchPos = null;
let controlsSignature = "";
let currentMode = "";
let boardStrokeActive = false;
let boardLastPoint = null;
let boardShapeStart = null;
let boardState = null;
let boardFonts = {};
let selectedTextId = "";
let selectedTextIds = new Set();
let selectedImageIds = new Set();
let dragState = null;
let dragMoveInFlight = false;
let dragQueuedPosition = null;
// Panel dimensions come from the frame payload (width/height); the defaults
// only cover the moment before the first frame arrives.
let gridWidth = DEFAULT_GRID;
let gridHeight = DEFAULT_GRID;
let latestFramePixels = Array.from({ length: gridHeight }, () => Array(gridWidth).fill(0));
let activeBoardTool = "select";
let previousDrawTool = "freehand";
let sleepStatusTimer = null;
let fontPreviewCatalog = {};
let fontPreviewVariants = [null, null, null, null];

/**
 * Coerce a raw controller status into a fully-populated, validated shape.
 * @param {Object} raw - The raw status payload (may be null/partial).
 * @returns {Object} Normalized controller status.
 */
function normalizeControllerStatus(raw) {
  if (!raw || typeof raw !== "object") {
    return {
      enabled: false,
      connected: false,
      pressed_buttons: [],
      last_event_age_ms: null,
      battery_percentage: null,
    };
  }

  const pressed = Array.isArray(raw.pressed_buttons)
    ? raw.pressed_buttons
        .map((label) => String(label || "").trim())
        .filter((label) => label.length > 0)
    : [];

  const battery = Number(raw.battery_percentage);
  const batteryPercentage =
    Number.isFinite(battery) && battery >= 0 && battery <= 100 ? Math.round(battery) : null;
  const lastEventAge = Number(raw.last_event_age_ms);
  const lastEventAgeMs =
    Number.isFinite(lastEventAge) && lastEventAge >= 0 ? Math.round(lastEventAge) : null;

  return {
    enabled: Boolean(raw.enabled),
    connected: Boolean(raw.connected),
    address: String(raw.address || ""),
    device_name: String(raw.device_name || ""),
    pressed_buttons: pressed,
    last_event_age_ms: lastEventAgeMs,
    battery_percentage: batteryPercentage,
  };
}

/** Format a last-event age in ms as a compact label. @param {number} ageMs @returns {string} */
function controllerEventAgeLabel(ageMs) {
  if (!Number.isFinite(ageMs)) {
    return "--";
  }
  if (ageMs < 1000) {
    return `${ageMs}ms`;
  }
  if (ageMs < 10000) {
    return `${(ageMs / 1000).toFixed(1)}s`;
  }
  return `${Math.round(ageMs / 1000)}s`;
}

/** @param {number} index @returns {string} The player tag ("P1", "P2", …). */
function controllerPlayerTag(index) {
  return index === 0 ? "P1" : `P${index + 1}`;
}

/** @param {Object} status @returns {string} "unavailable", "connected", or "disconnected". */
function controllerStatusState(status) {
  if (!status.enabled) {
    return "unavailable";
  }
  return status.connected ? "connected" : "disconnected";
}

/** Map a button label to its display glyph (arrows for the D-pad). @param {string} label @returns {string} */
function displayControllerButtonLabel(label) {
  const normalized = String(label || "").trim();
  const dpadArrows = {
    "D-Up": "↑",
    "D-Down": "↓",
    "D-Left": "←",
    "D-Right": "→",
  };
  return dpadArrows[normalized] || normalized;
}

/**
 * Extract a list of normalized controller statuses from a frame payload,
 * supporting both the multi-controller and legacy single-controller shapes.
 * @param {Object} payload
 * @returns {Object[]} One normalized status per controller.
 */
function controllerStatusEntries(payload) {
  if (payload && Array.isArray(payload.controller_statuses)) {
    const normalized = payload.controller_statuses.map((item) => normalizeControllerStatus(item));
    if (normalized.length > 0) {
      return normalized;
    }
  }
  return [normalizeControllerStatus(payload ? payload.controller_status : null)];
}

/**
 * Render the controller status pills (connection, battery, freshness, buttons).
 * @param {Object} payload - The latest frame payload carrying controller status.
 */
function renderControllerStatus(payload) {
  if (!controllerStatuses) {
    return;
  }

  const BUTTON_SLOT_COUNT = 3;

  const statuses = controllerStatusEntries(payload);
  controllerStatuses.innerHTML = "";

  for (let index = 0; index < statuses.length; index += 1) {
    const status = statuses[index];
    const item = document.createElement("span");
    item.className = `status-pill controller-status-item${status.connected ? "" : " muted"}`;
    const lastEventLabel = controllerEventAgeLabel(status.last_event_age_ms);
    const connState = controllerStatusState(status);
    item.title = `${status.device_name || status.address || "Controller"} | ${connState} | last event ${lastEventLabel} ago`;

    const text = document.createElement("span");
    text.className = "controller-pill-label";
    const dot = document.createElement("span");
    dot.className = `controller-conn-dot ${connState}`;
    dot.setAttribute("aria-label", connState);
    text.appendChild(dot);
    text.appendChild(document.createTextNode(controllerPlayerTag(index)));
    item.appendChild(text);

    const battery = document.createElement("span");
    battery.className = "controller-pill-battery";
    const batteryIcon = document.createElement("span");
    batteryIcon.className = "battery-icon";
    const batteryLevel = document.createElement("span");
    batteryLevel.className = "battery-level";
    batteryIcon.appendChild(batteryLevel);
    const batteryText = document.createElement("span");
    batteryText.className = "battery-text";

    if (!status.enabled || !status.connected || status.battery_percentage === null) {
      battery.classList.add("unknown");
      batteryText.textContent = "--%";
      batteryLevel.style.width = "0%";
    } else {
      const value = Math.max(0, Math.min(100, status.battery_percentage));
      batteryText.textContent = `${value}%`;
      batteryLevel.style.width = `${value}%`;
      if (value <= 20) {
        battery.classList.add("low");
      } else if (value <= 50) {
        battery.classList.add("medium");
      } else {
        battery.classList.add("good");
      }
    }
    battery.appendChild(batteryIcon);
    battery.appendChild(batteryText);
    item.appendChild(battery);

    const freshness = document.createElement("span");
    freshness.className = "controller-pill-freshness";
    freshness.textContent = lastEventLabel;
    if (!status.enabled || !status.connected || status.last_event_age_ms === null) {
      freshness.classList.add("unknown");
    } else if (status.last_event_age_ms > 250) {
      freshness.classList.add("stale");
    }
    item.appendChild(freshness);

    const buttons = document.createElement("span");
    buttons.className = "controller-pill-buttons";

    const slotLabels = [];
    if (!status.enabled || !status.connected) {
      slotLabels.push("-");
    } else if (status.pressed_buttons.length === 0) {
      slotLabels.push("idle");
    } else {
      slotLabels.push(...status.pressed_buttons.slice(0, BUTTON_SLOT_COUNT));
    }

    for (let slot = 0; slot < BUTTON_SLOT_COUNT; slot += 1) {
      const chip = document.createElement("span");
      const label = slotLabels[slot] || "";
      chip.className = "controller-chip";
      if (label.length > 0 && label !== "idle" && label !== "-") {
        chip.classList.add("active");
      }
      if (!label) {
        chip.classList.add("slot-empty");
      }
      chip.textContent = displayControllerButtonLabel(label);
      buttons.appendChild(chip);
    }

    item.appendChild(buttons);
    controllerStatuses.appendChild(item);
  }
}

/** @returns {HTMLElement[]} The settings panels for the current mode. */
function activeSettingsPanels() {
  if (currentMode === "font_preview") {
    return [fontPreviewSettings];
  }
  if (currentMode === "clock") {
    // Clock mode gets its own settings plus the general sleep/person-detection panel.
    return [clockSettings, sleepSettings];
  }
  return [sleepSettings];
}

/** Hide every mode settings panel. */
function hideAllSettingsPanels() {
  if (sleepSettings) {
    sleepSettings.classList.add("hidden");
  }
  if (clockSettings) {
    clockSettings.classList.add("hidden");
  }
  if (fontPreviewSettings) {
    fontPreviewSettings.classList.add("hidden");
  }
}

/** Collapse whitespace and cap a phrase at 32 chars, defaulting to "FLIPDOT". @param {string} value @returns {string} */
function cleanPhrase(value) {
  const compact = String(value || "")
    .replace(/\s+/g, " ")
    .trim();
  if (!compact) {
    return "FLIPDOT";
  }
  return compact.slice(0, 32);
}

/** Clamp font-preview spacing to [0, 6]. @param {*} value @returns {number} */
function clampFontPreviewSpacing(value) {
  const parsed = toInt(value, 0);
  return Math.max(0, Math.min(6, parsed));
}

/** Clamp board glyph spacing to [0, 6]. @param {*} value @returns {number} */
function clampBoardGlyphSpacing(value) {
  return Math.max(0, Math.min(6, toInt(value, 1)));
}

/** Snap a pixel size down to a whole multiple of the grid cell. @param {number} value @returns {number} */
function snapToGridMultiple(value) {
  const n = Math.max(gridWidth, Number.isFinite(value) ? Math.floor(value) : gridWidth);
  return Math.max(gridWidth, Math.floor(n / gridWidth) * gridWidth);
}

/**
 * Adopt the panel dimensions reported by a frame payload, resizing the canvas
 * when they change (defaults cover the moment before the first frame).
 * @param {number} width - Panel width in dots.
 * @param {number} height - Panel height in dots.
 */
function updateGridSize(width, height) {
  const w = toInt(width, DEFAULT_GRID);
  const h = toInt(height, DEFAULT_GRID);
  if (w > 0 && h > 0 && (w !== gridWidth || h !== gridHeight)) {
    gridWidth = w;
    gridHeight = h;
    syncCanvasResolution();
  }
}

/** Resize the canvas backing store to match its CSS size and DPR, then redraw. */
function syncCanvasResolution() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    return;
  }

  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssSize = snapToGridMultiple(rect.width);
  const deviceSize = snapToGridMultiple(cssSize * dpr);

  const cssSizePx = `${cssSize}px`;
  if (canvas.style.width !== cssSizePx) {
    canvas.style.width = cssSizePx;
    canvas.style.height = cssSizePx;
  }

  if (canvas.width === deviceSize && canvas.height === deviceSize) {
    return;
  }

  canvas.width = deviceSize;
  canvas.height = deviceSize;
  drawGrid(latestFramePixels);
}

/**
 * Deep-copy a panel-sized pixel array, coercing values to 0/1.
 * @param {number[][]} pixels
 * @returns {number[][]} A fresh copy.
 */
function clonePixels(pixels) {
  const out = Array.from({ length: gridHeight }, () => Array(gridWidth).fill(0));
  for (let y = 0; y < gridHeight; y += 1) {
    for (let x = 0; x < gridWidth; x += 1) {
      out[y][x] = pixels[y] && pixels[y][x] === 1 ? 1 : 0;
    }
  }
  return out;
}

/** Read a CSS custom property, falling back if unset. @param {string} name @param {string} fallback @returns {string} */
function themeColor(name, fallback) {
  const value = THEME.getPropertyValue(name).trim();
  return value || fallback;
}

/** Draw the full flip-dot grid (and any selection overlay) for the given pixels. @param {number[][]} pixels */
function drawGrid(pixels) {
  ctx.imageSmoothingEnabled = false;
  const cell = canvas.width / gridWidth;
  const panelBack = themeColor("--surface-soft", "#15171b");
  const dotBack = themeColor("--ink", "#0b0d10");
  const dotOn = themeColor("--dot-on", "#f7d15c");
  const dotOff = themeColor("--dot-off", "#39404b");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = panelBack;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < gridHeight; y += 1) {
    for (let x = 0; x < gridWidth; x += 1) {
      const on = pixels[y] && pixels[y][x] === 1;
      const cx = x * cell + cell * 0.5;
      const cy = y * cell + cell * 0.5;
      const radius = cell * 0.45;

      ctx.beginPath();
      ctx.arc(cx, cy, radius + cell * 0.04, 0, Math.PI * 2);
      ctx.fillStyle = dotBack;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = on ? dotOn : dotOff;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx - radius * 0.32, cy - radius * 0.32, radius * 0.32, 0, Math.PI * 2);
      ctx.fillStyle = on ? "rgba(255,255,255,0.24)" : "rgba(255,255,255,0.07)";
      ctx.fill();
    }
  }

  drawSelectionOverlay(cell);
}

/** Outline the currently selected board text/image objects. @param {number} cell - Pixel size of one grid cell. */
function drawSelectionOverlay(cell) {
  if (!isBoardMode() || !boardState) {
    return;
  }

  const textObjects = Array.isArray(boardState.text_objects) ? boardState.text_objects : [];
  const imageObjects = Array.isArray(boardState.image_objects) ? boardState.image_objects : [];

  const drawBounds = (bounds, color, lineWidth = 2) => {
    if (!bounds || bounds.width <= 0 || bounds.height <= 0) {
      return;
    }
    const x = bounds.x * cell + 1;
    const y = bounds.y * cell + 1;
    const width = bounds.width * cell - 2;
    const height = bounds.height * cell - 2;
    if (width <= 0 || height <= 0) {
      return;
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x, y, width, height);
  };

  for (const item of textObjects) {
    if (!selectedTextIds.has(item.id)) {
      continue;
    }
    const isPrimary = item.id === selectedTextId;
    drawBounds(item.bounds, isPrimary ? "#6ef0d5" : "#4cc0ab", isPrimary ? 2.5 : 1.5);
  }

  for (const item of imageObjects) {
    if (!selectedImageIds.has(item.id)) {
      continue;
    }
    drawBounds(
      {
        x: Number(item.x) || 0,
        y: Number(item.y) || 0,
        width: Number(item.width) || 0,
        height: Number(item.height) || 0,
      },
      "#f5a85b",
      1.5
    );
  }
}

/**
 * Paint a square brush of pixels centered on (x, y) into a pixel array.
 * @param {number[][]} pixels @param {number} x @param {number} y
 * @param {number} [drawValue] - 1 to set, 0 to clear.
 * @param {number} [lineWidth] - Brush size in pixels.
 */
function setPreviewPixel(pixels, x, y, drawValue = 1, lineWidth = 1) {
  if (x < 0 || x >= gridWidth || y < 0 || y >= gridHeight) {
    return;
  }
  const width = Math.max(1, Math.min(8, toInt(lineWidth, 1)));
  const value = drawValue === 0 ? 0 : 1;
  const x0 = x - Math.floor(width / 2);
  const y0 = y - Math.floor(width / 2);

  for (let row = y0; row < y0 + width; row += 1) {
    if (row < 0 || row >= gridHeight) {
      continue;
    }
    for (let col = x0; col < x0 + width; col += 1) {
      if (col < 0 || col >= gridWidth) {
        continue;
      }
      pixels[row][col] = value;
    }
  }
}

/** Convert a normalized [0,1] position to integer grid pixel coords. @param {{x:number,y:number}} pos @returns {{x:number,y:number}} */
function pixelPointFromNorm(pos) {
  return {
    x: Math.min(gridWidth - 1, Math.floor(Math.max(0, Math.min(1, Number(pos.x))) * gridWidth)),
    y: Math.min(gridHeight - 1, Math.floor(Math.max(0, Math.min(1, Number(pos.y))) * gridHeight)),
  };
}

/**
 * Rasterize a straight line between two grid points into a pixel array.
 * @param {number[][]} pixels @param {{x:number,y:number}} p0 @param {{x:number,y:number}} p1
 * @param {{drawValue?:number, lineWidth?:number}} [options]
 */
function rasterLine(pixels, p0, p1, options = {}) {
  const drawValue = options.drawValue === 0 ? 0 : 1;
  const lineWidth = options.lineWidth;
  const dx = p1.x - p0.x;
  const dy = p1.y - p0.y;
  const steps = Math.max(Math.abs(dx), Math.abs(dy), 1);
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    const x = Math.round(p0.x + dx * t);
    const y = Math.round(p0.y + dy * t);
    setPreviewPixel(pixels, x, y, drawValue, lineWidth);
  }
}

/**
 * Rasterize a rectangle outline between two corner points.
 * @param {number[][]} pixels @param {{x:number,y:number}} p0 @param {{x:number,y:number}} p1
 * @param {{drawValue?:number, lineWidth?:number}} [options]
 */
function rasterRect(pixels, p0, p1, options = {}) {
  const minX = Math.min(p0.x, p1.x);
  const maxX = Math.max(p0.x, p1.x);
  const minY = Math.min(p0.y, p1.y);
  const maxY = Math.max(p0.y, p1.y);
  rasterLine(pixels, { x: minX, y: minY }, { x: maxX, y: minY }, options);
  rasterLine(pixels, { x: maxX, y: minY }, { x: maxX, y: maxY }, options);
  rasterLine(pixels, { x: maxX, y: maxY }, { x: minX, y: maxY }, options);
  rasterLine(pixels, { x: minX, y: maxY }, { x: minX, y: minY }, options);
}

/**
 * Rasterize a circle whose diameter spans the two points' bounding box.
 * @param {number[][]} pixels @param {{x:number,y:number}} p0 @param {{x:number,y:number}} p1
 * @param {{drawValue?:number, lineWidth?:number}} [options]
 */
function rasterCircle(pixels, p0, p1, options = {}) {
  const drawValue = options.drawValue === 0 ? 0 : 1;
  const lineWidth = options.lineWidth;
  const cx = Math.round((p0.x + p1.x) / 2);
  const cy = Math.round((p0.y + p1.y) / 2);
  const radius = Math.max(
    1,
    Math.round(Math.max(Math.abs(p1.x - p0.x), Math.abs(p1.y - p0.y)) / 2)
  );
  for (let angle = 0; angle < 360; angle += 2) {
    const rad = (angle * Math.PI) / 180;
    const x = Math.round(cx + radius * Math.cos(rad));
    const y = Math.round(cy + radius * Math.sin(rad));
    setPreviewPixel(pixels, x, y, drawValue, lineWidth);
  }
}

/** @returns {number} The selected draw line width, clamped to [1, 8]. */
function getDrawLineWidth() {
  const width = toInt(boardDrawLineWidth && boardDrawLineWidth.value, 1);
  return Math.max(1, Math.min(8, width));
}

/** @returns {string} The selected draw color, "on" or "off". */
function getDrawColor() {
  return boardDrawColor && boardDrawColor.value === "off" ? "off" : "on";
}

/** @returns {number} The pixel value (1/0) for the selected draw color. */
function getDrawValue() {
  return getDrawColor() === "off" ? 0 : 1;
}

/** Redraw the canvas with a live preview of the in-progress shape. @param {{x:number,y:number}} currentPos */
function renderShapePreview(currentPos) {
  if (!boardStrokeActive || !boardShapeStart) {
    return;
  }
  const tool = boardTool.value;
  if (!tool || tool === "freehand") {
    return;
  }

  const previewPixels = clonePixels(latestFramePixels);
  const p0 = pixelPointFromNorm(boardShapeStart);
  const p1 = pixelPointFromNorm(currentPos || boardShapeStart);
  const options = {
    lineWidth: getDrawLineWidth(),
    drawValue: getDrawValue(),
  };

  if (tool === "line") {
    rasterLine(previewPixels, p0, p1, options);
  } else if (tool === "rectangle") {
    rasterRect(previewPixels, p0, p1, options);
  } else if (tool === "circle") {
    rasterCircle(previewPixels, p0, p1, options);
  }

  drawGrid(previewPixels);
}

/**
 * Map a client point to a normalized [0,1] position over the canvas content box.
 * getBoundingClientRect() returns the border box, but the backing store (where dots
 * are drawn) fills only the content box, so we inset by the border via clientLeft/Top
 * and scale by clientWidth/Height to avoid a progressive offset toward the far edges.
 * @param {number} clientX @param {number} clientY @returns {{x:number,y:number}}
 */
function normPosFromClient(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const x = (clientX - rect.left - canvas.clientLeft) / canvas.clientWidth;
  const y = (clientY - rect.top - canvas.clientTop) / canvas.clientHeight;
  return {
    x: Math.max(0, Math.min(1, x)),
    y: Math.max(0, Math.min(1, y)),
  };
}

/** Map a mouse/pointer event to a normalized [0,1] canvas position. @param {MouseEvent} event @returns {{x:number,y:number}} */
function normPosFromEvent(event) {
  return normPosFromClient(event.clientX, event.clientY);
}

/** Map a touch point to a normalized [0,1] canvas position. @param {Touch} touch @returns {{x:number,y:number}} */
function normPosFromTouch(touch) {
  return normPosFromClient(touch.clientX, touch.clientY);
}

/** POST JSON to a URL, returning the response or null on network error. @param {string} url @param {Object} payload @returns {Promise<Response|null>} */
async function postJson(url, payload) {
  try {
    return await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (_err) {
    return null;
  }
}

/** PATCH JSON to a URL, returning the response or null on network error. @param {string} url @param {Object} payload @returns {Promise<Response|null>} */
async function patchJson(url, payload) {
  try {
    return await fetch(url, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (_err) {
    return null;
  }
}

/** Send a DELETE request, returning the response or null on network error. @param {string} url @returns {Promise<Response|null>} */
async function deleteJson(url) {
  try {
    return await fetch(url, { method: "DELETE" });
  } catch (_err) {
    return null;
  }
}

/** GET and parse JSON, throwing on a non-OK response. @param {string} url @returns {Promise<Object>} */
async function getJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`request failed: ${response.status}`);
  }
  return response.json();
}

/** @returns {boolean} Whether the board mode is currently active. */
function isBoardMode() {
  return currentMode === "board";
}

/**
 * Convert a normalized [0,1] coordinate to an integer grid pixel index.
 * @param {number} norm - Normalized coordinate.
 * @param {number} [axisSize] - Grid size along the axis (defaults to the width).
 * @returns {number} Pixel index clamped to the axis.
 */
function toGridPixel(norm, axisSize = gridWidth) {
  const value = Number(norm);
  const clamped = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  return Math.min(axisSize - 1, Math.floor(clamped * axisSize));
}

/** Parse an integer, returning a fallback if invalid. @param {*} value @param {number} fallback @returns {number} */
function toInt(value, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

/** Parse a float, returning a fallback if invalid. @param {*} value @param {number} fallback @returns {number} */
function toFloat(value, fallback) {
  const parsed = Number.parseFloat(String(value));
  return Number.isFinite(parsed) ? parsed : fallback;
}

/** Parse an hour and clamp it to [0, 23]. @param {*} value @param {number} fallback @returns {number} */
function clampHour(value, fallback) {
  return Math.max(0, Math.min(23, toInt(value, fallback)));
}

/** Enable or disable the sleep-schedule hour inputs. @param {boolean} enabled */
function setSleepInputsEnabled(enabled) {
  const disabled = !enabled;
  sleepStartHour.disabled = disabled;
  sleepEndHour.disabled = disabled;
}

/** Set the sleep-settings status line. @param {string} message @param {string} [kind] - "error", "ok", or "". */
function setSleepSettingsStatus(message, kind = "") {
  sleepSettingsStatus.textContent = message;
  sleepSettingsStatus.classList.remove("error", "ok");
  if (kind === "error" || kind === "ok") {
    sleepSettingsStatus.classList.add(kind);
  }
}

/** Format how long ago a timestamp was as "just now"/"Ns ago"/etc. @param {number} since - Epoch ms. @returns {string} */
function formatRelativeTimestamp(since) {
  const seconds = Math.max(0, Math.floor((Date.now() - since) / 1000));
  if (seconds <= 1) {
    return "just now";
  }
  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

/**
 * Show a sleep-settings status that updates its relative timestamp each second.
 * @param {string} prefix - The status text prefix.
 * @param {string} kind - "error", "ok", or "".
 */
function setSleepStatusWithTimestamp(prefix, kind) {
  if (sleepStatusTimer !== null) {
    window.clearInterval(sleepStatusTimer);
    sleepStatusTimer = null;
  }

  const startedAt = Date.now();
  const refresh = () => {
    setSleepSettingsStatus(`${prefix} (${formatRelativeTimestamp(startedAt)})`, kind);
  };

  refresh();
  sleepStatusTimer = window.setInterval(refresh, 1000);
}

/** Set the font-preview status line. @param {string} message @param {string} [kind] - "error", "ok", or "". */
function setFontPreviewStatus(message, kind = "") {
  if (!fontPreviewStatus) {
    return;
  }
  fontPreviewStatus.textContent = message;
  fontPreviewStatus.classList.remove("error", "ok");
  if (kind === "error" || kind === "ok") {
    fontPreviewStatus.classList.add(kind);
  }
}

/** @returns {string[]} The available font family names, sorted. */
function fontPreviewFamilyNames() {
  return Object.keys(fontPreviewCatalog || {}).sort();
}

/** @param {string} family @returns {string[]} The available size keys for a family, sorted numerically. */
function fontPreviewSizeKeys(family) {
  if (!family || !fontPreviewCatalog || !fontPreviewCatalog[family]) {
    return [];
  }
  return Object.keys(fontPreviewCatalog[family]).sort((a, b) => Number(a) - Number(b));
}

/** @param {string} family @param {string} sizeKey @returns {string[]} The available styles for a family/size. */
function fontPreviewStyleNames(family, sizeKey) {
  if (!family || !sizeKey || !fontPreviewCatalog || !fontPreviewCatalog[family]) {
    return [];
  }
  const styles = fontPreviewCatalog[family][sizeKey];
  return Array.isArray(styles) ? styles : [];
}

/**
 * Validate and complete a font-preview variant against the catalog.
 * @param {Object} entry - A {family, size, style} candidate.
 * @returns {{family:string, size:number, style:string}|null} A valid variant, or null.
 */
function normalizeFontPreviewVariant(entry) {
  if (!entry || typeof entry !== "object") {
    return null;
  }

  const family = typeof entry.family === "string" ? entry.family : "";
  const families = fontPreviewFamilyNames();
  if (!families.includes(family)) {
    return null;
  }

  const sizeKeys = fontPreviewSizeKeys(family);
  if (sizeKeys.length === 0) {
    return null;
  }

  const requestedSize = toInt(entry.size, Number(sizeKeys[0]));
  const sizeKey = sizeKeys.includes(String(requestedSize)) ? String(requestedSize) : sizeKeys[0];

  const styles = fontPreviewStyleNames(family, sizeKey);
  if (styles.length === 0) {
    return null;
  }

  const style =
    typeof entry.style === "string" && styles.includes(entry.style) ? entry.style : styles[0];
  return { family, size: Number(sizeKey), style };
}

/** Build a default variant (first size/style) for a family. @param {string} family @returns {Object|null} */
function defaultFontPreviewVariantForFamily(family) {
  const sizeKeys = fontPreviewSizeKeys(family);
  if (sizeKeys.length === 0) {
    return null;
  }
  const styles = fontPreviewStyleNames(family, sizeKeys[0]);
  if (styles.length === 0) {
    return null;
  }
  return { family, size: Number(sizeKeys[0]), style: styles[0] };
}

/** Replace the four preview slots from a saved variants payload. @param {Object[]} variants */
function setFontPreviewVariantsFromPayload(variants) {
  fontPreviewVariants = [null, null, null, null];
  if (!Array.isArray(variants)) {
    return;
  }
  for (let i = 0; i < Math.min(4, variants.length); i += 1) {
    const normalized = normalizeFontPreviewVariant(variants[i]);
    if (normalized) {
      fontPreviewVariants[i] = normalized;
    }
  }
}

/** @returns {Object[]} The valid, deduplicated preview-slot variants (max 4) to persist. */
function collectFontPreviewVariantsPayload() {
  const payload = [];
  const seen = new Set();
  for (const entry of fontPreviewVariants) {
    const normalized = normalizeFontPreviewVariant(entry);
    if (!normalized) {
      continue;
    }
    const key = `${normalized.family}:${normalized.size}:${normalized.style}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    payload.push(normalized);
  }
  return payload.slice(0, 4);
}

/** Render the four font-preview slot rows (family/size/style selectors + handlers). */
function renderFontPreviewSlots() {
  if (!fontPreviewSlots) {
    return;
  }

  fontPreviewSlots.innerHTML = "";
  const families = fontPreviewFamilyNames();
  if (families.length === 0) {
    const empty = document.createElement("p");
    empty.className = "context-hint";
    empty.textContent = "No font catalog available.";
    fontPreviewSlots.appendChild(empty);
    return;
  }

  for (let slotIndex = 0; slotIndex < 4; slotIndex += 1) {
    const slot = fontPreviewVariants[slotIndex];
    const row = document.createElement("div");
    row.className = "font-preview-slot-row";

    const slotLabel = document.createElement("span");
    slotLabel.className = "font-preview-slot-label";
    slotLabel.textContent = `Slot ${slotIndex + 1}`;
    row.appendChild(slotLabel);

    const familySelect = document.createElement("select");
    const noneOption = document.createElement("option");
    noneOption.value = "";
    noneOption.textContent = "(none)";
    familySelect.appendChild(noneOption);
    for (const family of families) {
      const option = document.createElement("option");
      option.value = family;
      option.textContent = family;
      familySelect.appendChild(option);
    }
    familySelect.value = slot ? slot.family : "";

    const sizeSelect = document.createElement("select");
    const activeFamily = slot ? slot.family : "";
    const sizeKeys = fontPreviewSizeKeys(activeFamily);
    for (const sizeKey of sizeKeys) {
      const option = document.createElement("option");
      option.value = sizeKey;
      option.textContent = sizeKey;
      sizeSelect.appendChild(option);
    }
    if (slot && sizeKeys.includes(String(slot.size))) {
      sizeSelect.value = String(slot.size);
    }
    sizeSelect.disabled = !slot;

    const styleSelect = document.createElement("select");
    const styleNames = slot ? fontPreviewStyleNames(slot.family, String(slot.size)) : [];
    for (const styleName of styleNames) {
      const option = document.createElement("option");
      option.value = styleName;
      option.textContent = styleName;
      styleSelect.appendChild(option);
    }
    if (slot && styleNames.includes(slot.style)) {
      styleSelect.value = slot.style;
    }
    styleSelect.disabled = !slot;

    familySelect.addEventListener("change", () => {
      const family = familySelect.value;
      if (!family) {
        fontPreviewVariants[slotIndex] = null;
      } else {
        fontPreviewVariants[slotIndex] = defaultFontPreviewVariantForFamily(family);
      }
      renderFontPreviewSlots();
      saveFontPreviewSettings();
    });

    sizeSelect.addEventListener("change", () => {
      const current = fontPreviewVariants[slotIndex];
      if (!current) {
        return;
      }
      const nextSize = toInt(sizeSelect.value, current.size);
      const styles = fontPreviewStyleNames(current.family, String(nextSize));
      const nextStyle = styles.includes(current.style) ? current.style : styles[0] || "regular";
      fontPreviewVariants[slotIndex] = {
        family: current.family,
        size: nextSize,
        style: nextStyle,
      };
      renderFontPreviewSlots();
      saveFontPreviewSettings();
    });

    styleSelect.addEventListener("change", () => {
      const current = fontPreviewVariants[slotIndex];
      if (!current) {
        return;
      }
      fontPreviewVariants[slotIndex] = {
        family: current.family,
        size: current.size,
        style: styleSelect.value,
      };
      saveFontPreviewSettings();
    });

    row.appendChild(familySelect);
    row.appendChild(sizeSelect);
    row.appendChild(styleSelect);
    fontPreviewSlots.appendChild(row);
  }
}

/** Load the sleep schedule from the backend into the settings form. */
async function loadSleepSettings() {
  try {
    const payload = await getJson("/api/settings/sleep");
    sleepEnabled.checked = Boolean(payload.enabled);
    sleepStartHour.value = String(clampHour(payload.start_hour, 0));
    sleepEndHour.value = String(clampHour(payload.end_hour, 7));
    setSleepInputsEnabled(sleepEnabled.checked);
    setSleepStatusWithTimestamp("Sleep settings loaded", "ok");
  } catch (_err) {
    setSleepSettingsStatus("Sleep settings unavailable.", "error");
  }
}

/** Persist the sleep schedule to the backend and reflect the saved result. */
async function saveSleepSettings() {
  const payload = {
    enabled: sleepEnabled.checked,
    start_hour: clampHour(sleepStartHour.value, 0),
    end_hour: clampHour(sleepEndHour.value, 7),
  };

  sleepStartHour.value = String(payload.start_hour);
  sleepEndHour.value = String(payload.end_hour);

  const response = await postJson("/api/settings/sleep", payload);
  if (!response || !response.ok) {
    setSleepSettingsStatus("Failed to save sleep settings.", "error");
    return;
  }

  try {
    const saved = await response.json();
    sleepEnabled.checked = Boolean(saved.enabled);
    sleepStartHour.value = String(clampHour(saved.start_hour, payload.start_hour));
    sleepEndHour.value = String(clampHour(saved.end_hour, payload.end_hour));
    setSleepInputsEnabled(sleepEnabled.checked);
    setSleepStatusWithTimestamp("Sleep settings saved", "ok");
  } catch (_err) {
    setSleepStatusWithTimestamp("Sleep settings saved", "ok");
  }
}

/** Set the person-detection status line. @param {string} message @param {string} [kind] - "error", "ok", or "". */
function setPoseSettingsStatus(message, kind = "") {
  poseSettingsStatus.textContent = message;
  poseSettingsStatus.classList.remove("error", "ok");
  if (kind === "error" || kind === "ok") {
    poseSettingsStatus.classList.add(kind);
  }
}

/** Load the person-detection (auto sandfall/caricature chain) toggle from the backend. */
async function loadPoseSettings() {
  try {
    const payload = await getJson("/api/settings/pose");
    poseChainEnabled.checked = Boolean(payload.enabled);
    setPoseSettingsStatus("Person detection loaded", "ok");
  } catch (_err) {
    setPoseSettingsStatus("Person detection unavailable.", "error");
  }
}

/** Persist the person-detection toggle and reflect the saved result. */
async function savePoseSettings() {
  const response = await postJson("/api/settings/pose", {
    enabled: poseChainEnabled.checked,
  });
  if (!response || !response.ok) {
    setPoseSettingsStatus("Failed to save person detection.", "error");
    return;
  }

  try {
    const saved = await response.json();
    poseChainEnabled.checked = Boolean(saved.enabled);
    setPoseSettingsStatus("Person detection saved", "ok");
  } catch (_err) {
    setPoseSettingsStatus("Person detection saved", "ok");
  }
}

/** Set the clock-settings status line. @param {string} message @param {string} [kind] - "error", "ok", or "". */
function setClockSettingsStatus(message, kind = "") {
  if (!clockSettingsStatus) {
    return;
  }
  clockSettingsStatus.textContent = message;
  clockSettingsStatus.classList.remove("error", "ok");
  if (kind) {
    clockSettingsStatus.classList.add(kind);
  }
}

/** Load the clock face style and reflect it in the form. */
async function loadClockSettings() {
  if (!clockDisplayStyle) {
    return;
  }
  try {
    const payload = await getJson("/api/settings/clock");
    clockDisplayStyle.value = payload.style === "analog" ? "analog" : "digital";
    if (clockSeconds) {
      clockSeconds.checked = Boolean(payload.seconds);
    }
    setClockSettingsStatus("Clock settings loaded.", "ok");
  } catch (_err) {
    setClockSettingsStatus("Clock settings unavailable.", "error");
  }
}

/** Persist the clock face style and reflect the saved result. */
async function saveClockSettings() {
  if (!clockDisplayStyle) {
    return;
  }
  const payload = {
    style: clockDisplayStyle.value === "analog" ? "analog" : "digital",
    seconds: clockSeconds ? clockSeconds.checked : false,
  };
  const response = await postJson("/api/settings/clock", payload);
  if (!response || !response.ok) {
    setClockSettingsStatus("Failed to save clock settings.", "error");
    return;
  }
  try {
    const saved = await response.json();
    clockDisplayStyle.value = saved.style === "analog" ? "analog" : "digital";
    if (clockSeconds) {
      clockSeconds.checked = Boolean(saved.seconds);
    }
  } catch (_err) {
    // Keep the optimistic value if the response body is unreadable.
  }
  setClockSettingsStatus("Clock settings saved.", "ok");
}

/** Load font-preview settings and the variant catalog, then render the form. */
async function loadFontPreviewSettings() {
  if (!fontPreviewPhrase || !fontPreviewSlots || !fontPreviewSpacing) {
    return;
  }
  try {
    const [payload, catalog] = await Promise.all([
      getJson("/api/settings/font-preview"),
      getJson("/api/font-preview/variants"),
    ]);
    fontPreviewCatalog = catalog && typeof catalog === "object" ? catalog : {};
    fontPreviewPhrase.value = cleanPhrase(payload.phrase);
    fontPreviewSpacing.value = String(clampFontPreviewSpacing(payload.spacing));
    setFontPreviewVariantsFromPayload(payload.variants);
    renderFontPreviewSlots();
    setFontPreviewStatus("Font preview settings loaded.", "ok");
  } catch (_err) {
    setFontPreviewStatus("Font preview settings unavailable.", "error");
  }
}

/** Persist font-preview phrase, spacing, and variants, then reflect the result. */
async function saveFontPreviewSettings() {
  if (!fontPreviewPhrase || !fontPreviewSpacing) {
    return;
  }
  const payload = {
    phrase: cleanPhrase(fontPreviewPhrase.value),
    spacing: clampFontPreviewSpacing(fontPreviewSpacing.value),
    variants: collectFontPreviewVariantsPayload(),
  };
  fontPreviewPhrase.value = payload.phrase;
  fontPreviewSpacing.value = String(payload.spacing);

  const response = await postJson("/api/settings/font-preview", payload);
  if (!response || !response.ok) {
    setFontPreviewStatus("Failed to save font preview settings.", "error");
    return;
  }

  try {
    const saved = await response.json();
    fontPreviewPhrase.value = cleanPhrase(saved.phrase);
    fontPreviewSpacing.value = String(clampFontPreviewSpacing(saved.spacing));
    setFontPreviewVariantsFromPayload(saved.variants);
    renderFontPreviewSlots();
    setFontPreviewStatus("Font preview settings saved.", "ok");
  } catch (_err) {
    setFontPreviewStatus("Font preview settings saved.", "ok");
  }
}

/** @returns {Object|null} The primary-selected board text object, if any. */
function selectedTextObject() {
  if (!boardState || !Array.isArray(boardState.text_objects)) {
    return null;
  }
  return boardState.text_objects.find((item) => item.id === selectedTextId) || null;
}

/** @returns {Object|null} The first selected board image object, if any. */
function selectedImageObject() {
  if (!boardState || !Array.isArray(boardState.image_objects) || selectedImageIds.size === 0) {
    return null;
  }
  const firstId = Array.from(selectedImageIds)[0];
  return boardState.image_objects.find((item) => item.id === firstId) || null;
}

/** Find a board object by kind and id. @param {string} kind - "text" or "image". @param {string} id @returns {Object|null} */
function getObjectById(kind, id) {
  if (!boardState || !id) {
    return null;
  }
  if (kind === "text") {
    return (boardState.text_objects || []).find((item) => item.id === id) || null;
  }
  if (kind === "image") {
    return (boardState.image_objects || []).find((item) => item.id === id) || null;
  }
  return null;
}

/** Whether a board object is selected. @param {string} kind - "text" or "image". @param {string} id @returns {boolean} */
function isSelected(kind, id) {
  if (kind === "text") {
    return selectedTextIds.has(id);
  }
  if (kind === "image") {
    return selectedImageIds.has(id);
  }
  return false;
}

/** Update the selection summary text to reflect the current selection. */
function refreshSelectionSummary() {
  const textCount = selectedTextIds.size;
  const imageCount = selectedImageIds.size;
  const total = textCount + imageCount;
  if (!boardSelectionSummary) {
    return;
  }
  if (total === 0) {
    boardSelectionSummary.textContent = "No object selected";
    return;
  }
  if (total === 1) {
    const primaryText = selectedTextObject();
    const primaryImage = selectedImageObject();
    if (primaryText) {
      boardSelectionSummary.textContent = `Selected text: ${primaryText.id}`;
      return;
    }
    if (primaryImage) {
      boardSelectionSummary.textContent = `Selected image: ${primaryImage.id}`;
      return;
    }
  }
  boardSelectionSummary.textContent = `Selected ${textCount} text, ${imageCount} image objects`;
}

/** Populate the editor fields from the currently selected text/image objects. */
function syncSelectionFields() {
  const textObj = selectedTextObject();
  applyTextObjectToFields(textObj);
  const imageObj = selectedImageObject();
  if (imageObj) {
    boardImageX.value = String(imageObj.x ?? 0);
    boardImageY.value = String(imageObj.y ?? 0);
  }
  refreshSelectionSummary();
}

/**
 * Replace the current selection with the given ids (filtered to existing objects).
 * @param {string[]} nextTextIds @param {string[]} nextImageIds
 */
function replaceSelection(nextTextIds, nextImageIds) {
  const validText = new Set();
  const validImage = new Set();
  const textIds = Array.isArray(nextTextIds) ? nextTextIds : [];
  const imageIds = Array.isArray(nextImageIds) ? nextImageIds : [];

  for (const id of textIds) {
    if (getObjectById("text", id)) {
      validText.add(id);
    }
  }
  for (const id of imageIds) {
    if (getObjectById("image", id)) {
      validImage.add(id);
    }
  }

  selectedTextIds = validText;
  selectedImageIds = validImage;

  if (selectedTextIds.has(selectedTextId)) {
    // Keep primary selection.
  } else if (selectedTextIds.size > 0) {
    selectedTextId = Array.from(selectedTextIds)[0];
  } else {
    selectedTextId = "";
  }

  if (boardTextObjectSelect && selectedTextId) {
    boardTextObjectSelect.value = selectedTextId;
  }

  syncSelectionFields();
  renderBoardContext();
  drawGrid(latestFramePixels);
}

/** Clear the entire board selection. */
function clearSelection() {
  replaceSelection([], []);
}

/** Select exactly one object. @param {string} kind - "text" or "image". @param {string} id */
function selectOnly(kind, id) {
  if (!id) {
    clearSelection();
    return;
  }
  if (kind === "text") {
    selectedTextId = id;
    replaceSelection([id], []);
    return;
  }
  if (kind === "image") {
    replaceSelection([], [id]);
  }
}

/** Add or remove one object from the selection. @param {string} kind - "text" or "image". @param {string} id */
function toggleSelection(kind, id) {
  if (!id) {
    return;
  }
  const nextText = new Set(selectedTextIds);
  const nextImage = new Set(selectedImageIds);

  if (kind === "text") {
    if (nextText.has(id)) {
      nextText.delete(id);
      if (selectedTextId === id) {
        selectedTextId = nextText.size > 0 ? Array.from(nextText)[0] : "";
      }
    } else {
      nextText.add(id);
      selectedTextId = id;
    }
  } else if (kind === "image") {
    if (nextImage.has(id)) {
      nextImage.delete(id);
    } else {
      nextImage.add(id);
    }
  }

  replaceSelection(Array.from(nextText), Array.from(nextImage));
}

/** Populate the board font-family dropdown from the loaded fonts. */
function renderFontFamilyOptions() {
  boardFontFamily.innerHTML = "";
  const families = Object.keys(boardFonts).sort();
  if (families.length === 0) {
    return;
  }
  for (const family of families) {
    const option = document.createElement("option");
    option.value = family;
    option.textContent = family;
    boardFontFamily.appendChild(option);
  }
}

/** Populate the board font-size dropdown for the selected family, preserving the choice if valid. */
function renderFontSizeOptions() {
  const family = boardFontFamily.value;
  const sizeMap = boardFonts[family] || {};
  const sizes = Object.keys(sizeMap).sort((a, b) => Number(a) - Number(b));

  const currentValue = boardFontSize.value;
  boardFontSize.innerHTML = "";
  for (const size of sizes) {
    const option = document.createElement("option");
    option.value = size;
    option.textContent = size;
    boardFontSize.appendChild(option);
  }
  if (sizes.includes(currentValue)) {
    boardFontSize.value = currentValue;
  }
}

/** Populate the board font-style dropdown for the selected family/size, preserving the choice if valid. */
function renderFontStyleOptions() {
  const family = boardFontFamily.value;
  const size = boardFontSize.value;
  const styleList = (boardFonts[family] && boardFonts[family][size]) || [];

  const currentValue = boardFontStyle.value;
  boardFontStyle.innerHTML = "";
  for (const style of styleList) {
    const option = document.createElement("option");
    option.value = style;
    option.textContent = style;
    boardFontStyle.appendChild(option);
  }
  if (styleList.includes(currentValue)) {
    boardFontStyle.value = currentValue;
  }
}

/** Load a text object's properties into the editor fields (or reset them if null). @param {Object|null} item */
function applyTextObjectToFields(item) {
  if (!item) {
    boardText.value = "";
    boardTextX.value = "0";
    boardTextY.value = "11";
    boardGlyphSpacing.value = "1";
    boardTextScroll.checked = false;
    boardScrollSpeed.value = "7";
    return;
  }

  boardText.value = item.text || "";
  boardTextX.value = String(item.x ?? 0);
  boardTextY.value = String(item.y ?? 11);
  boardGlyphSpacing.value = String(clampBoardGlyphSpacing(item.spacing));
  boardTextScroll.checked = Boolean(item.scroll);
  boardScrollSpeed.value = String(item.scroll_speed ?? 7);

  if (item.font && Object.prototype.hasOwnProperty.call(boardFonts, item.font)) {
    boardFontFamily.value = item.font;
    renderFontSizeOptions();
    boardFontSize.value = String(item.size);
    renderFontStyleOptions();
    boardFontStyle.value = item.style;
  }
}

/** Rebuild the text-object dropdown and sync the primary selection + fields. */
function renderTextObjectList() {
  const current = selectedTextId;
  boardTextObjectSelect.innerHTML = "";
  const objects = (boardState && boardState.text_objects) || [];

  for (const obj of objects) {
    const option = document.createElement("option");
    option.value = obj.id;
    option.textContent = `${obj.id}: ${obj.text || "(empty)"}`;
    boardTextObjectSelect.appendChild(option);
  }

  if (objects.some((item) => item.id === current)) {
    selectedTextId = current;
  } else if (selectedTextIds.size > 0) {
    const next = Array.from(selectedTextIds).find((id) => objects.some((obj) => obj.id === id));
    selectedTextId = next || (objects.length > 0 ? objects[0].id : "");
  } else {
    selectedTextId = objects.length > 0 ? objects[0].id : "";
  }

  if (selectedTextId) {
    boardTextObjectSelect.value = selectedTextId;
  }
  applyTextObjectToFields(selectedTextObject());
}

/** Populate the saved-board dropdown and highlight the active board. */
function renderBoardLibrary() {
  boardList.innerHTML = "";
  const boards = (boardState && boardState.boards) || [];
  const active = (boardState && boardState.active_board) || "default";

  for (const boardName of boards) {
    const option = document.createElement("option");
    option.value = boardName;
    option.textContent = boardName;
    boardList.appendChild(option);
  }
  if (boards.includes(active)) {
    boardList.value = active;
    boardNameInput.value = active;
  }
}

/** Reflect the active board tool as a CSS class on the canvas (drives the cursor). */
function updateCanvasToolClass() {
  canvas.classList.remove("tool-select", "tool-draw", "tool-text", "tool-image", "dragging");
  if (activeBoardTool === "select") {
    canvas.classList.add("tool-select");
  } else if (activeBoardTool === "draw") {
    canvas.classList.add("tool-draw");
  } else if (activeBoardTool === "text") {
    canvas.classList.add("tool-text");
  } else if (activeBoardTool === "image") {
    canvas.classList.add("tool-image");
  }
}

/** Sync the toolbar/shape buttons' active state and the canvas tool class. */
function renderToolbarState() {
  for (const button of toolbarButtons) {
    button.classList.toggle("is-active", button.dataset.tool === activeBoardTool);
  }
  for (const button of shapeButtons) {
    button.classList.toggle("is-active", button.dataset.shape === previousDrawTool);
  }
  updateCanvasToolClass();
}

/** Show only the context block (draw/text/image/empty) relevant to the active tool/selection. */
function renderBoardContext() {
  let context = "empty";
  if (activeBoardTool === "draw") {
    context = "draw";
  } else if (activeBoardTool === "text") {
    context = "text";
  } else if (activeBoardTool === "image") {
    context = "image";
  } else if (activeBoardTool === "select") {
    if (selectedTextIds.size > 0) {
      context = "text";
    } else if (selectedImageIds.size > 0) {
      context = "image";
    } else {
      context = "empty";
    }
  }

  for (const block of contextBlocks) {
    block.hidden = block.dataset.context !== context;
  }

  // Placement hints only make sense for the active placing tools.
  const textPlaceHint = document.querySelector('[data-context-hint="text-place"]');
  if (textPlaceHint) {
    textPlaceHint.hidden = activeBoardTool !== "text";
  }
  const imagePlaceHint = document.querySelector('[data-context-hint="image-place"]');
  if (imagePlaceHint) {
    imagePlaceHint.hidden = activeBoardTool !== "image";
  }
}

/** Switch the active board tool and refresh the toolbar/context UI. @param {string} nextTool */
function setBoardTool(nextTool) {
  const tool = typeof nextTool === "string" ? nextTool : "select";
  activeBoardTool = tool;

  if (tool === "draw") {
    if (!DRAW_TOOLS.has(previousDrawTool)) {
      previousDrawTool = "freehand";
    }
    boardTool.value = previousDrawTool;
    boardDrawToggle.checked = true;
  } else {
    boardDrawToggle.checked = false;
    if (DRAW_TOOLS.has(previousDrawTool)) {
      boardTool.value = previousDrawTool;
    }
  }

  renderToolbarState();
  renderBoardContext();
}

/** Select a draw shape (freehand/line/rect/circle) and activate the draw tool. @param {string} shape */
function setDrawShape(shape) {
  if (!DRAW_TOOLS.has(shape)) {
    return;
  }
  previousDrawTool = shape;
  boardTool.value = shape;
  boardDrawToggle.checked = true;
  renderToolbarState();
}

/** Fetch the board font catalog and rebuild the family/size/style dropdowns. */
async function loadBoardFonts() {
  if (!isBoardMode()) {
    return;
  }
  try {
    boardFonts = await getJson("/api/board/fonts");
    renderFontFamilyOptions();
    renderFontSizeOptions();
    renderFontStyleOptions();
  } catch (_err) {
    // Ignore transient sync failures.
  }
}

/** Refresh just the saved-board list (names + active) without the full board state. */
async function refreshBoardLibraryOnly() {
  if (!isBoardMode()) {
    return;
  }
  try {
    const payload = await getJson("/api/boards");
    boardState = {
      ...(boardState || {}),
      boards: payload.boards || [],
      active_board: payload.active || "default",
    };
    renderBoardLibrary();
  } catch (_err) {
    // Ignore transient sync failures.
  }
}

/** Fetch the full board state and refresh selection, lists, fonts, and library. */
async function syncBoardState() {
  if (!isBoardMode()) {
    return;
  }
  try {
    const payload = await getJson("/api/board/state");
    boardState = payload;
    if (!Array.isArray(boardState.boards)) {
      boardState.boards = [];
    }

    const stateTextIds = Array.isArray(boardState.selected_text_ids)
      ? boardState.selected_text_ids
      : boardState.selected_text_id
        ? [boardState.selected_text_id]
        : [];
    const stateImageIds = Array.isArray(boardState.selected_image_ids)
      ? boardState.selected_image_ids
      : boardState.selected_image_id
        ? [boardState.selected_image_id]
        : [];

    replaceSelection(stateTextIds, stateImageIds);
    renderTextObjectList();
    renderBoardLibrary();
    if (Object.keys(boardFonts).length === 0) {
      await loadBoardFonts();
    }
    await refreshBoardLibraryOnly();
  } catch (_err) {
    // Ignore transient sync failures.
  }
}

/** Send a freehand stroke (list of normalized points) to the board. @param {Object[]} points */
async function sendBoardStroke(points) {
  if (!isBoardMode()) {
    return;
  }
  if (!Array.isArray(points) || points.length === 0) {
    return;
  }
  await postJson("/api/board/draw", {
    points,
    line_width: getDrawLineWidth(),
    color: getDrawColor(),
  });
}

/**
 * Hit-test a normalized position against board objects on the server.
 * @param {{x:number,y:number}} pos
 * @param {{allHits?:boolean, select?:boolean}} [options]
 * @returns {Promise<{hit:Object|null, hits:Object[]}>}
 */
async function boardHitTest(pos, options = {}) {
  try {
    const response = await fetch("/api/board/hit-test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: pos.x,
        y: pos.y,
        all_hits: Boolean(options.allHits),
        select: Boolean(options.select),
      }),
    });
    if (!response.ok) {
      return { hit: null, hits: [] };
    }
    const payload = await response.json();
    const hits = Array.isArray(payload.hits) ? payload.hits : payload.hit ? [payload.hit] : [];
    return {
      hit: payload.hit || hits[0] || null,
      hits,
    };
  } catch (_err) {
    return { hit: null, hits: [] };
  }
}

/** Reflect an object's position in the editor fields. @param {string} kind @param {number} x @param {number} y */
function updateEditorPositionFields(kind, x, y) {
  if (kind === "text") {
    boardTextX.value = String(x);
    boardTextY.value = String(y);
    return;
  }
  if (kind === "image") {
    boardImageX.value = String(x);
    boardImageY.value = String(y);
  }
}

/** Update an object's position in the local board state. @param {string} kind @param {string} id @param {number} x @param {number} y */
function updateLocalObjectPosition(kind, id, x, y) {
  const item = getObjectById(kind, id);
  if (!item) {
    return;
  }
  item.x = x;
  item.y = y;
}

/** POST a drag-move payload to an endpoint. @param {Object} payload @param {string} endpoint @returns {Promise<boolean>} Success. */
async function sendDragMove(payload, endpoint) {
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return response.ok;
  } catch (_err) {
    return false;
  }
}

/** Send the latest queued drag position, coalescing rapid moves into one in-flight request. */
async function flushQueuedDragMove() {
  if (!dragState || dragMoveInFlight || !dragQueuedPosition) {
    return;
  }

  const next = dragQueuedPosition;
  dragQueuedPosition = null;
  dragMoveInFlight = true;

  const ok = await sendDragMove({ ids: next.items }, "/api/board/drag/move");
  dragMoveInFlight = false;

  if (!ok) {
    dragState = null;
    dragQueuedPosition = null;
    await syncBoardState();
    return;
  }

  if (dragQueuedPosition) {
    flushQueuedDragMove();
  }
}

/** Update local positions for dragged items, redraw, and queue a server sync. @param {Object[]} items */
function queueDragMove(items) {
  if (!dragState || !Array.isArray(items) || items.length === 0) {
    return;
  }

  for (const next of items) {
    const dragItem = dragState.items.find((item) => item.id === next.id && item.kind === next.kind);
    if (!dragItem) {
      continue;
    }
    dragItem.currentX = next.x;
    dragItem.currentY = next.y;
    updateLocalObjectPosition(next.kind, next.id, next.x, next.y);
  }

  const primary = dragState.items[0];
  if (primary) {
    updateEditorPositionFields(primary.kind, primary.currentX, primary.currentY);
  }

  dragQueuedPosition = { items };
  drawGrid(latestFramePixels);
  flushQueuedDragMove();
}

/** @returns {Object[]} The selected text/image objects with their kind, id, and position. */
function getSelectedObjects() {
  const selected = [];
  for (const id of selectedTextIds) {
    const item = getObjectById("text", id);
    if (item) {
      selected.push({ kind: "text", id, x: Number(item.x) || 0, y: Number(item.y) || 0 });
    }
  }
  for (const id of selectedImageIds) {
    const item = getObjectById("image", id);
    if (item) {
      selected.push({ kind: "image", id, x: Number(item.x) || 0, y: Number(item.y) || 0 });
    }
  }
  return selected;
}

/**
 * Hit-test a pointer position and update the selection accordingly.
 * @param {{x:number,y:number}} pos @param {boolean} additive - Toggle vs. replace selection.
 * @returns {Promise<Object|null>} The hit object, or null.
 */
async function selectFromPointer(pos, additive) {
  const hitResponse = await boardHitTest(pos, { allHits: true, select: false });
  const hit = hitResponse.hit;
  if (!hit || !hit.id || !hit.kind) {
    if (!additive) {
      clearSelection();
    }
    return null;
  }

  if (additive) {
    toggleSelection(hit.kind, hit.id);
  } else if (!isSelected(hit.kind, hit.id) || selectedTextIds.size + selectedImageIds.size > 1) {
    selectOnly(hit.kind, hit.id);
  }

  if (hit.kind === "text") {
    selectedTextId = hit.id;
    if (boardTextObjectSelect) {
      boardTextObjectSelect.value = hit.id;
    }
  }

  syncSelectionFields();
  return hit;
}

/** Begin dragging the object(s) under the pointer. @param {{x:number,y:number}} pos @returns {Promise<boolean>} Whether a drag started. */
async function beginObjectDrag(pos) {
  const hit = await selectFromPointer(pos, false);
  if (!hit) {
    return false;
  }

  const selected = getSelectedObjects();
  if (selected.length === 0) {
    return false;
  }

  const pixelX = toGridPixel(pos.x);
  const pixelY = toGridPixel(pos.y, gridHeight);
  dragState = {
    anchorX: pixelX,
    anchorY: pixelY,
    items: selected.map((item) => ({
      ...item,
      originX: item.x,
      originY: item.y,
      currentX: item.x,
      currentY: item.y,
    })),
  };
  dragQueuedPosition = null;
  dragMoveInFlight = false;
  canvas.classList.add("dragging");

  const primary = dragState.items[0];
  if (primary) {
    updateEditorPositionFields(primary.kind, primary.originX, primary.originY);
  }
  return true;
}

/** Move dragged objects to follow the pointer. @param {{x:number,y:number}} pos */
function updateObjectDrag(pos) {
  if (!dragState) {
    return;
  }

  const pixelX = toGridPixel(pos.x);
  const pixelY = toGridPixel(pos.y, gridHeight);
  const deltaX = pixelX - dragState.anchorX;
  const deltaY = pixelY - dragState.anchorY;
  const nextItems = dragState.items.map((item) => ({
    kind: item.kind,
    id: item.id,
    x: item.originX + deltaX,
    y: item.originY + deltaY,
  }));

  queueDragMove(nextItems);
}

/** Commit the final dragged positions to the server and resync. */
async function endObjectDrag() {
  if (!dragState) {
    return;
  }

  const commitPayload = {
    ids: dragState.items.map((item) => ({
      kind: item.kind,
      id: item.id,
      x: item.currentX,
      y: item.currentY,
    })),
  };

  dragState = null;
  dragQueuedPosition = null;
  dragMoveInFlight = false;
  canvas.classList.remove("dragging");
  await sendDragMove(commitPayload, "/api/board/drag/commit");
  await syncBoardState();
}

/** Create a text object at a pointer position using the current editor fields. @param {{x:number,y:number}} pos */
async function placeTextAt(pos) {
  const response = await postJson("/api/board/text-objects", {
    text: boardText.value || "",
    x: toGridPixel(pos.x),
    y: toGridPixel(pos.y, gridHeight),
    font: boardFontFamily.value || "classic",
    size: toInt(boardFontSize.value, 5),
    style: boardFontStyle.value || "regular",
    spacing: clampBoardGlyphSpacing(boardGlyphSpacing && boardGlyphSpacing.value),
    scroll: boardTextScroll.checked,
    scroll_speed: toFloat(boardScrollSpeed.value, 7),
  });
  if (response && response.ok) {
    await syncBoardState();
  }
}

/**
 * Upload the chosen image file to the board at grid (x, y).
 * @param {number} x @param {number} y
 * @returns {Promise<boolean>} Whether the upload succeeded.
 */
async function uploadImageAt(x, y) {
  const file = boardImageFile.files && boardImageFile.files[0];
  if (!file) {
    return false;
  }
  const formData = new FormData();
  formData.append("file", file);
  formData.append("mode", boardImageMode.value || "stamp");
  formData.append("x", String(toInt(x, 0)));
  formData.append("y", String(toInt(y, 0)));
  formData.append("threshold", String(toInt(boardImageThreshold.value, 128)));
  try {
    const response = await fetch("/api/board/image/upload", {
      method: "POST",
      body: formData,
    });
    if (response.ok) {
      await syncBoardState();
      return true;
    }
  } catch (_err) {
    // Ignore transient network failures.
  }
  return false;
}

/** Switch the active display mode and reconcile the board editor and settings panels. @param {string} nextMode */
function setMode(nextMode) {
  const previousMode = currentMode;
  currentMode = typeof nextMode === "string" ? nextMode : "";
  const modeChanged = previousMode !== currentMode;
  if (modeChanged) {
    hideAllSettingsPanels();
  }
  const boardVisible = isBoardMode();
  boardEditor.classList.toggle("hidden", !boardVisible);
  if (boardVisible && modeChanged) {
    syncBoardState();
    setBoardTool("select");
  }
  if (!boardVisible) {
    boardStrokeActive = false;
    boardLastPoint = null;
    boardShapeStart = null;
    dragState = null;
    dragQueuedPosition = null;
    dragMoveInFlight = false;
    selectedTextIds = new Set();
    selectedImageIds = new Set();
    selectedTextId = "";
    setBoardsMenuOpen(false);
  }

  if (modeChanged && currentMode === "font_preview" && fontPreviewSettings) {
    fontPreviewSettings.classList.remove("hidden");
  }
}

/** Map a control action to its icon glyph. @param {string} action @returns {string} */
function getControlGlyph(action) {
  const glyphs = {
    toggle_menu: "=",
    paint_clear: "x",
    autodrum_next_song: ">",
    board_clear: "x",
    board_undo: "<",
    font_preview_prev: "<",
    font_preview_next: ">",
  };
  return glyphs[action] || "o";
}

/** Render the mode's on-screen control buttons (diffed against the last render). @param {Object[]} controls */
function renderControls(controls) {
  const normalized = Array.isArray(controls) ? controls : [];
  const signature = JSON.stringify(normalized);
  if (signature === controlsSignature) {
    return;
  }
  controlsSignature = signature;

  modeControls.innerHTML = "";
  for (const control of normalized) {
    if (!control || !control.action || !control.label) {
      continue;
    }
    if (isBoardMode() && (control.action === "board_clear" || control.action === "board_undo")) {
      // Board clear/undo are handled by icon actions in the board toolbar.
      continue;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = control.variant === "secondary" ? "secondary" : "accent";
    button.classList.add("mode-control-btn");
    if (control.action === "toggle_menu") {
      button.classList.add("mode-control-menu");
    }
    button.title = control.label;
    button.setAttribute("aria-label", control.label);
    const glyph = document.createElement("span");
    glyph.className = "mode-control-glyph";
    glyph.setAttribute("aria-hidden", "true");
    glyph.textContent = getControlGlyph(control.action);
    const label = document.createElement("span");
    label.textContent = control.label;
    button.append(glyph, label);
    button.addEventListener("click", () => {
      postJson("/api/input/action", { action: control.action });
    });
    modeControls.appendChild(button);
  }
}

sleepEnabled.addEventListener("change", () => {
  setSleepInputsEnabled(sleepEnabled.checked);
  saveSleepSettings();
});

sleepStartHour.addEventListener("change", () => {
  saveSleepSettings();
});

sleepEndHour.addEventListener("change", () => {
  saveSleepSettings();
});

poseChainEnabled.addEventListener("change", () => {
  savePoseSettings();
});

if (clockDisplayStyle) {
  clockDisplayStyle.addEventListener("change", () => {
    saveClockSettings();
  });
}

if (clockSeconds) {
  clockSeconds.addEventListener("change", () => {
    saveClockSettings();
  });
}

settingsToggle.addEventListener("click", () => {
  const panels = activeSettingsPanels().filter(Boolean);
  if (!panels.length) {
    return;
  }
  const shouldOpen = panels.some((panel) => panel.classList.contains("hidden"));
  hideAllSettingsPanels();
  for (const panel of panels) {
    panel.classList.toggle("hidden", !shouldOpen);
  }
});

if (fontPreviewPhrase) {
  fontPreviewPhrase.addEventListener("change", () => {
    saveFontPreviewSettings();
  });
}

if (fontPreviewSpacing) {
  fontPreviewSpacing.addEventListener("change", () => {
    saveFontPreviewSettings();
  });
}

canvas.addEventListener("pointermove", (event) => {
  const pos = normPosFromEvent(event);
  sourceText.textContent = "⊳ web";

  if (isBoardMode() && boardDrawToggle.checked && boardStrokeActive) {
    if (boardTool.value === "freehand") {
      sendBoardStroke([boardLastPoint || pos, pos]);
      boardLastPoint = pos;
    } else {
      renderShapePreview(pos);
    }
    return;
  }

  if (isBoardMode() && activeBoardTool === "select" && dragState) {
    updateObjectDrag(pos);
    return;
  }

  postJson("/api/input/pointer", pos);
});

canvas.addEventListener("pointerdown", async (event) => {
  const pos = normPosFromEvent(event);
  sourceText.textContent = "⊳ web";

  if (isBoardMode() && activeBoardTool === "text") {
    await placeTextAt(pos);
    return;
  }

  if (isBoardMode() && activeBoardTool === "image") {
    await uploadImageAt(toGridPixel(pos.x), toGridPixel(pos.y, gridHeight));
    return;
  }

  if (isBoardMode() && boardDrawToggle.checked) {
    boardStrokeActive = true;
    boardLastPoint = pos;
    if (boardTool.value === "freehand") {
      sendBoardStroke([pos]);
    } else {
      boardShapeStart = pos;
      renderShapePreview(pos);
    }
    return;
  }

  if (isBoardMode() && activeBoardTool === "select") {
    if (event.shiftKey) {
      await selectFromPointer(pos, true);
      return;
    }
    await beginObjectDrag(pos);
    return;
  }

  postJson("/api/input/pointer", pos);
  postJson("/api/input/button", { down: true });
});

canvas.addEventListener("pointerup", (event) => {
  const pos = normPosFromEvent(event);
  if (isBoardMode() && boardDrawToggle.checked) {
    if (boardStrokeActive && boardTool.value !== "freehand" && boardShapeStart) {
      postJson("/api/board/shapes", {
        tool: boardTool.value,
        start: boardShapeStart,
        end: pos,
        line_width: getDrawLineWidth(),
        color: getDrawColor(),
      }).then(() => syncBoardState());
    }
    boardStrokeActive = false;
    boardLastPoint = null;
    boardShapeStart = null;
    return;
  }

  if (isBoardMode() && activeBoardTool === "select") {
    endObjectDrag();
    return;
  }

  postJson("/api/input/button", { down: false });
  postJson("/api/input/click", pos);
});

canvas.addEventListener("pointercancel", () => {
  boardStrokeActive = false;
  boardLastPoint = null;
  boardShapeStart = null;
  if (isBoardMode() && activeBoardTool === "select") {
    endObjectDrag();
    return;
  }
  postJson("/api/input/button", { down: false });
});

canvas.addEventListener("pointerleave", (event) => {
  if (event.buttons === 0) {
    postJson("/api/input/button", { down: false });
  }
});

canvas.addEventListener(
  "touchstart",
  (event) => {
    event.preventDefault();
    if (!HAS_POINTER_EVENTS && event.touches.length > 0) {
      const pos = normPosFromTouch(event.touches[0]);
      lastTouchPos = pos;
      sourceText.textContent = "⊳ web";
      if (isBoardMode() && boardDrawToggle.checked) {
        boardStrokeActive = true;
        boardLastPoint = pos;
        if (boardTool.value === "freehand") {
          sendBoardStroke([pos]);
        } else {
          boardShapeStart = pos;
          renderShapePreview(pos);
        }
        return;
      }
      if (isBoardMode() && activeBoardTool === "select") {
        beginObjectDrag(pos);
        return;
      }
      postJson("/api/input/pointer", pos);
      postJson("/api/input/button", { down: true });
    }
  },
  { passive: false }
);

canvas.addEventListener(
  "touchmove",
  (event) => {
    event.preventDefault();
    if (!HAS_POINTER_EVENTS && event.touches.length > 0) {
      const pos = normPosFromTouch(event.touches[0]);
      lastTouchPos = pos;
      sourceText.textContent = "⊳ web";
      if (isBoardMode() && boardDrawToggle.checked && boardStrokeActive) {
        if (boardTool.value === "freehand") {
          sendBoardStroke([boardLastPoint || pos, pos]);
          boardLastPoint = pos;
        } else {
          renderShapePreview(pos);
        }
        return;
      }
      if (isBoardMode() && activeBoardTool === "select" && dragState) {
        updateObjectDrag(pos);
        return;
      }
      postJson("/api/input/pointer", pos);
    }
  },
  { passive: false }
);

canvas.addEventListener(
  "touchend",
  (event) => {
    event.preventDefault();
    if (!HAS_POINTER_EVENTS && isBoardMode() && boardDrawToggle.checked) {
      if (boardStrokeActive && boardTool.value !== "freehand" && boardShapeStart) {
        postJson("/api/board/shapes", {
          tool: boardTool.value,
          start: boardShapeStart,
          end: lastTouchPos || boardShapeStart,
          line_width: getDrawLineWidth(),
          color: getDrawColor(),
        }).then(() => syncBoardState());
      }
      boardStrokeActive = false;
      boardLastPoint = null;
      boardShapeStart = null;
      return;
    }
    if (!HAS_POINTER_EVENTS && isBoardMode() && activeBoardTool === "select") {
      endObjectDrag();
      return;
    }
    if (!HAS_POINTER_EVENTS && lastTouchPos !== null) {
      postJson("/api/input/button", { down: false });
      postJson("/api/input/click", lastTouchPos);
    }
  },
  { passive: false }
);

canvas.addEventListener(
  "gesturestart",
  (event) => {
    event.preventDefault();
  },
  { passive: false }
);

canvas.addEventListener(
  "gesturechange",
  (event) => {
    event.preventDefault();
  },
  { passive: false }
);

/** Open the frame WebSocket, rendering streamed frames and reconnecting on drop. */
function startWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    statusText.textContent = "⬤ On";
    statusText.className = "status-pill ok";
    if (pollingTimer !== null) {
      window.clearInterval(pollingTimer);
      pollingTimer = null;
    }
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    renderControllerStatus(data);
    if (data.version === lastVersion) {
      return;
    }
    lastVersion = data.version;
    updateGridSize(data.width, data.height);
    latestFramePixels = clonePixels(data.pixels);
    setMode(data.mode);
    renderControls(data.controls);
    drawGrid(data.pixels);
    if (
      isBoardMode() &&
      boardDrawToggle.checked &&
      boardStrokeActive &&
      boardTool.value !== "freehand"
    ) {
      renderShapePreview(boardLastPoint || boardShapeStart);
    }
  };

  ws.onclose = () => {
    statusText.textContent = "◎ HTTP";
    statusText.className = "status-pill warn";
    if (pollingTimer === null) {
      pollingTimer = window.setInterval(async () => {
        try {
          const response = await fetch("/api/frame", { cache: "no-store" });
          if (!response.ok) {
            return;
          }
          const data = await response.json();
          renderControllerStatus(data);
          if (data.version === lastVersion) {
            return;
          }
          lastVersion = data.version;
          updateGridSize(data.width, data.height);
          latestFramePixels = clonePixels(data.pixels);
          setMode(data.mode);
          renderControls(data.controls);
          drawGrid(data.pixels);
          if (
            isBoardMode() &&
            boardDrawToggle.checked &&
            boardStrokeActive &&
            boardTool.value !== "freehand"
          ) {
            renderShapePreview(boardLastPoint || boardShapeStart);
          }
        } catch (_err) {
          // Ignore transient network failures.
        }
      }, 120);
    }
    setTimeout(startWebSocket, 1000);
  };

  ws.onerror = () => {
    ws.close();
  };
}

/** Open the controller-status WebSocket, updating the status pills and reconnecting on drop. */
function startControllerWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/controller-status`);

  ws.onopen = () => {
    if (controllerPollingTimer !== null) {
      window.clearInterval(controllerPollingTimer);
      controllerPollingTimer = null;
    }
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    renderControllerStatus(data);
  };

  ws.onclose = () => {
    if (controllerPollingTimer === null) {
      controllerPollingTimer = window.setInterval(async () => {
        try {
          const response = await fetch("/api/controller/status", { cache: "no-store" });
          if (!response.ok) {
            return;
          }
          const data = await response.json();
          renderControllerStatus(data);
        } catch (_err) {
          // Ignore transient network failures.
        }
      }, 120);
    }
    setTimeout(startControllerWebSocket, 1000);
  };

  ws.onerror = () => {
    ws.close();
  };
}

renderControllerStatus(null);

boardFontFamily.addEventListener("change", () => {
  renderFontSizeOptions();
  renderFontStyleOptions();
});

boardFontSize.addEventListener("change", () => {
  renderFontStyleOptions();
});

boardTextObjectSelect.addEventListener("change", () => {
  selectedTextId = boardTextObjectSelect.value;
  if (selectedTextId) {
    replaceSelection([selectedTextId], Array.from(selectedImageIds));
  } else {
    clearSelection();
  }
  syncSelectionFields();
});

boardTextAdd.addEventListener("click", async () => {
  if (!isBoardMode()) {
    return;
  }
  const response = await postJson("/api/board/text-objects", {
    text: boardText.value || "",
    x: toInt(boardTextX.value, 0),
    y: toInt(boardTextY.value, 11),
    font: boardFontFamily.value || "classic",
    size: toInt(boardFontSize.value, 5),
    style: boardFontStyle.value || "regular",
    spacing: clampBoardGlyphSpacing(boardGlyphSpacing && boardGlyphSpacing.value),
    scroll: boardTextScroll.checked,
    scroll_speed: toFloat(boardScrollSpeed.value, 7),
  });
  if (response && response.ok) {
    await syncBoardState();
  }
});

boardApplyText.addEventListener("click", async () => {
  if (!isBoardMode()) {
    return;
  }

  if (!selectedTextId) {
    const response = await postJson("/api/board/text", { text: boardText.value || "" });
    if (response && response.ok) {
      await syncBoardState();
    }
    return;
  }

  const response = await patchJson(`/api/board/text-objects/${selectedTextId}`, {
    text: boardText.value || "",
    x: toInt(boardTextX.value, 0),
    y: toInt(boardTextY.value, 11),
    font: boardFontFamily.value || "classic",
    size: toInt(boardFontSize.value, 5),
    style: boardFontStyle.value || "regular",
    spacing: clampBoardGlyphSpacing(boardGlyphSpacing && boardGlyphSpacing.value),
    scroll: boardTextScroll.checked,
    scroll_speed: toFloat(boardScrollSpeed.value, 7),
  });
  if (response && response.ok) {
    await syncBoardState();
  }
});

boardText.addEventListener("keydown", async (event) => {
  if (event.key !== "Enter") {
    return;
  }
  event.preventDefault();
  if (!isBoardMode()) {
    return;
  }
  boardApplyText.click();
});

boardTextDelete.addEventListener("click", async () => {
  if (!isBoardMode() || !selectedTextId) {
    return;
  }
  const response = await deleteJson(`/api/board/text-objects/${selectedTextId}`);
  if (response && response.ok) {
    await syncBoardState();
  }
});

/** Delete all currently selected board text/image objects and resync. */
async function deleteSelection() {
  if (!isBoardMode()) {
    return;
  }
  const textToDelete = Array.from(selectedTextIds);
  const imageToDelete = Array.from(selectedImageIds);
  let changed = false;

  for (const id of textToDelete) {
    const response = await deleteJson(`/api/board/text-objects/${id}`);
    if (response && response.ok) {
      changed = true;
    }
  }
  for (const id of imageToDelete) {
    const response = await deleteJson(`/api/board/image-objects/${id}`);
    if (response && response.ok) {
      changed = true;
    }
  }

  if (changed) {
    await syncBoardState();
  }
}

boardDeleteSelection.addEventListener("click", () => {
  deleteSelection();
});

boardUploadImage.addEventListener("click", async () => {
  if (!isBoardMode()) {
    return;
  }
  const ok = await uploadImageAt(boardImageX.value, boardImageY.value);
  if (ok) {
    boardImageFile.value = "";
  }
});

boardSaveNamed.addEventListener("click", async () => {
  if (!isBoardMode()) {
    return;
  }
  const name = boardNameInput.value || boardList.value || "default";
  const response = await postJson("/api/boards/save", { name });
  if (response && response.ok) {
    await syncBoardState();
    setBoardsMenuOpen(false);
  }
});

boardLoadNamed.addEventListener("click", async () => {
  if (!isBoardMode()) {
    return;
  }
  const name = boardList.value || boardNameInput.value || "default";
  const response = await postJson("/api/boards/load", { name });
  if (response && response.ok) {
    await syncBoardState();
    setBoardsMenuOpen(false);
  }
});

boardDeleteNamed.addEventListener("click", async () => {
  if (!isBoardMode()) {
    return;
  }
  const name = boardList.value || boardNameInput.value;
  if (!name) {
    return;
  }
  const response = await postJson("/api/boards/delete", { name });
  if (response && response.ok) {
    await syncBoardState();
  }
});

boardClear.addEventListener("click", async () => {
  if (!isBoardMode()) {
    return;
  }
  const approved = window.confirm(
    "Clear the board and remove all objects? This can be undone once."
  );
  if (!approved) {
    return;
  }
  const response = await postJson("/api/board/clear", {});
  if (response && response.ok) {
    await syncBoardState();
  }
});

boardUndo.addEventListener("click", async () => {
  if (!isBoardMode()) {
    return;
  }
  const response = await postJson("/api/board/undo", {});
  if (response && response.ok) {
    await syncBoardState();
  }
});

for (const button of toolbarButtons) {
  button.addEventListener("click", () => {
    setBoardTool(button.dataset.tool || "select");
  });
}

for (const button of shapeButtons) {
  button.addEventListener("click", () => {
    setDrawShape(button.dataset.shape || "freehand");
  });
}

/** Open or close the saved-boards dropdown menu. @param {boolean} open */
function setBoardsMenuOpen(open) {
  if (!boardsMenu || !boardsMenuToggle) {
    return;
  }
  boardsMenu.classList.toggle("hidden", !open);
  boardsMenuToggle.setAttribute("aria-expanded", open ? "true" : "false");
  boardsMenuToggle.classList.toggle("is-active", open);
}

if (boardsMenuToggle) {
  boardsMenuToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    setBoardsMenuOpen(boardsMenu.classList.contains("hidden"));
  });
}

document.addEventListener("click", (event) => {
  if (!boardsMenu || boardsMenu.classList.contains("hidden")) {
    return;
  }
  if (boardsMenu.contains(event.target) || boardsMenuToggle.contains(event.target)) {
    return;
  }
  setBoardsMenuOpen(false);
});

/** @param {EventTarget} target @returns {boolean} Whether the target is within a text input/textarea/select. */
function isTextInputTarget(target) {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  return Boolean(target.closest("input, textarea, select"));
}

document.addEventListener("keydown", (event) => {
  if (!isBoardMode()) {
    return;
  }
  if (isTextInputTarget(event.target)) {
    return;
  }

  if (event.key === "Delete" || event.key === "Backspace") {
    event.preventDefault();
    deleteSelection();
    return;
  }

  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    boardUndo.click();
    return;
  }

  if (event.key === "Escape") {
    event.preventDefault();
    setBoardsMenuOpen(false);
    clearSelection();
    setBoardTool("select");
  }
});

setBoardTool("select");
syncCanvasResolution();
window.addEventListener("resize", syncCanvasResolution, { passive: true });
loadSleepSettings();
loadPoseSettings();
loadClockSettings();
loadFontPreviewSettings();
startWebSocket();
startControllerWebSocket();
