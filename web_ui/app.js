const canvas = document.getElementById("matrix");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("statusText");
const sourceText = document.getElementById("sourceText");
const modeControls = document.getElementById("modeControls");
const boardEditor = document.getElementById("boardEditor");
const settingsToggle = document.getElementById("settingsToggle");
const sleepSettings = document.getElementById("sleepSettings");
const fontPreviewSettings = document.getElementById("fontPreviewSettings");
const sleepEnabled = document.getElementById("sleepEnabled");
const sleepStartHour = document.getElementById("sleepStartHour");
const sleepEndHour = document.getElementById("sleepEndHour");
const sleepSettingsStatus = document.getElementById("sleepSettingsStatus");
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
const boardContext = document.getElementById("boardContext");
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

const GRID = 28;
const HAS_POINTER_EVENTS = "PointerEvent" in window;
const DRAW_TOOLS = new Set(["freehand", "line", "rectangle", "circle"]);
const THEME = window.getComputedStyle(document.documentElement);

let lastVersion = -1;
let pollingTimer = null;
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
let latestFramePixels = Array.from({ length: GRID }, () => Array(GRID).fill(0));
let activeBoardTool = "select";
let previousDrawTool = "freehand";
let sleepStatusTimer = null;
let fontPreviewCatalog = {};
let fontPreviewVariants = [null, null, null, null];

function activeSettingsPanel() {
  if (currentMode === "font_preview") {
    return fontPreviewSettings;
  }
  return sleepSettings;
}

function hideAllSettingsPanels() {
  if (sleepSettings) {
    sleepSettings.classList.add("hidden");
  }
  if (fontPreviewSettings) {
    fontPreviewSettings.classList.add("hidden");
  }
}

function cleanPhrase(value) {
  const compact = String(value || "").replace(/\s+/g, " ").trim();
  if (!compact) {
    return "FLIPDOT";
  }
  return compact.slice(0, 32);
}

function clampFontPreviewSpacing(value) {
  const parsed = toInt(value, 0);
  return Math.max(0, Math.min(6, parsed));
}

function clampBoardGlyphSpacing(value) {
  return Math.max(0, Math.min(6, toInt(value, 1)));
}

function snapToGridMultiple(value) {
  const n = Math.max(GRID, Number.isFinite(value) ? Math.floor(value) : GRID);
  return Math.max(GRID, Math.floor(n / GRID) * GRID);
}

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

function clonePixels(pixels) {
  const out = Array.from({ length: GRID }, () => Array(GRID).fill(0));
  for (let y = 0; y < GRID; y += 1) {
    for (let x = 0; x < GRID; x += 1) {
      out[y][x] = pixels[y] && pixels[y][x] === 1 ? 1 : 0;
    }
  }
  return out;
}

function themeColor(name, fallback) {
  const value = THEME.getPropertyValue(name).trim();
  return value || fallback;
}

function drawGrid(pixels) {
  ctx.imageSmoothingEnabled = false;
  const cell = canvas.width / GRID;
  const panelBack = themeColor("--surface-soft", "#15171b");
  const dotBack = themeColor("--ink", "#0b0d10");
  const dotOn = themeColor("--dot-on", "#f7d15c");
  const dotOff = themeColor("--dot-off", "#39404b");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = panelBack;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < GRID; y += 1) {
    for (let x = 0; x < GRID; x += 1) {
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

function setPreviewPixel(pixels, x, y, drawValue = 1, lineWidth = 1) {
  if (x < 0 || x >= GRID || y < 0 || y >= GRID) {
    return;
  }
  const width = Math.max(1, Math.min(8, toInt(lineWidth, 1)));
  const value = drawValue === 0 ? 0 : 1;
  const x0 = x - Math.floor(width / 2);
  const y0 = y - Math.floor(width / 2);

  for (let row = y0; row < y0 + width; row += 1) {
    if (row < 0 || row >= GRID) {
      continue;
    }
    for (let col = x0; col < x0 + width; col += 1) {
      if (col < 0 || col >= GRID) {
        continue;
      }
      pixels[row][col] = value;
    }
  }
}

function pixelPointFromNorm(pos) {
  return {
    x: Math.trunc(Math.max(0, Math.min(1, Number(pos.x))) * (GRID - 1)),
    y: Math.trunc(Math.max(0, Math.min(1, Number(pos.y))) * (GRID - 1)),
  };
}

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

function rasterCircle(pixels, p0, p1, options = {}) {
  const drawValue = options.drawValue === 0 ? 0 : 1;
  const lineWidth = options.lineWidth;
  const cx = Math.round((p0.x + p1.x) / 2);
  const cy = Math.round((p0.y + p1.y) / 2);
  const radius = Math.max(1, Math.round(Math.max(Math.abs(p1.x - p0.x), Math.abs(p1.y - p0.y)) / 2));
  for (let angle = 0; angle < 360; angle += 2) {
    const rad = (angle * Math.PI) / 180;
    const x = Math.round(cx + radius * Math.cos(rad));
    const y = Math.round(cy + radius * Math.sin(rad));
    setPreviewPixel(pixels, x, y, drawValue, lineWidth);
  }
}

function getDrawLineWidth() {
  const width = toInt(boardDrawLineWidth && boardDrawLineWidth.value, 1);
  return Math.max(1, Math.min(8, width));
}

function getDrawColor() {
  return boardDrawColor && boardDrawColor.value === "off" ? "off" : "on";
}

function getDrawValue() {
  return getDrawColor() === "off" ? 0 : 1;
}

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

function normPosFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width;
  const y = (event.clientY - rect.top) / rect.height;
  return {
    x: Math.max(0, Math.min(1, x)),
    y: Math.max(0, Math.min(1, y)),
  };
}

function normPosFromTouch(touch) {
  const rect = canvas.getBoundingClientRect();
  const x = (touch.clientX - rect.left) / rect.width;
  const y = (touch.clientY - rect.top) / rect.height;
  return {
    x: Math.max(0, Math.min(1, x)),
    y: Math.max(0, Math.min(1, y)),
  };
}

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

async function deleteJson(url) {
  try {
    return await fetch(url, { method: "DELETE" });
  } catch (_err) {
    return null;
  }
}

async function getJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`request failed: ${response.status}`);
  }
  return response.json();
}

function isBoardMode() {
  return currentMode === "board";
}

function toGridPixel(norm) {
  const value = Number(norm);
  const clamped = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  return Math.trunc(clamped * (GRID - 1));
}

function toInt(value, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toFloat(value, fallback) {
  const parsed = Number.parseFloat(String(value));
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clampHour(value, fallback) {
  return Math.max(0, Math.min(23, toInt(value, fallback)));
}

function setSleepInputsEnabled(enabled) {
  const disabled = !enabled;
  sleepStartHour.disabled = disabled;
  sleepEndHour.disabled = disabled;
}

function setSleepSettingsStatus(message, kind = "") {
  sleepSettingsStatus.textContent = message;
  sleepSettingsStatus.classList.remove("error", "ok");
  if (kind === "error" || kind === "ok") {
    sleepSettingsStatus.classList.add(kind);
  }
}

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

function fontPreviewFamilyNames() {
  return Object.keys(fontPreviewCatalog || {}).sort();
}

function fontPreviewSizeKeys(family) {
  if (!family || !fontPreviewCatalog || !fontPreviewCatalog[family]) {
    return [];
  }
  return Object.keys(fontPreviewCatalog[family]).sort((a, b) => Number(a) - Number(b));
}

function fontPreviewStyleNames(family, sizeKey) {
  if (!family || !sizeKey || !fontPreviewCatalog || !fontPreviewCatalog[family]) {
    return [];
  }
  const styles = fontPreviewCatalog[family][sizeKey];
  return Array.isArray(styles) ? styles : [];
}

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

  const style = typeof entry.style === "string" && styles.includes(entry.style) ? entry.style : styles[0];
  return { family, size: Number(sizeKey), style };
}

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

function selectedTextObject() {
  if (!boardState || !Array.isArray(boardState.text_objects)) {
    return null;
  }
  return boardState.text_objects.find((item) => item.id === selectedTextId) || null;
}

function selectedImageObject() {
  if (!boardState || !Array.isArray(boardState.image_objects) || selectedImageIds.size === 0) {
    return null;
  }
  const firstId = Array.from(selectedImageIds)[0];
  return boardState.image_objects.find((item) => item.id === firstId) || null;
}

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

function isSelected(kind, id) {
  if (kind === "text") {
    return selectedTextIds.has(id);
  }
  if (kind === "image") {
    return selectedImageIds.has(id);
  }
  return false;
}

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

function clearSelection() {
  replaceSelection([], []);
}

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

function renderToolbarState() {
  for (const button of toolbarButtons) {
    button.classList.toggle("is-active", button.dataset.tool === activeBoardTool);
  }
  for (const button of shapeButtons) {
    button.classList.toggle("is-active", button.dataset.shape === previousDrawTool);
  }
  updateCanvasToolClass();
}

// Show only the context block relevant to the active tool / selection.
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

function setDrawShape(shape) {
  if (!DRAW_TOOLS.has(shape)) {
    return;
  }
  previousDrawTool = shape;
  boardTool.value = shape;
  boardDrawToggle.checked = true;
  renderToolbarState();
}

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
    const hits = Array.isArray(payload.hits)
      ? payload.hits
      : payload.hit
        ? [payload.hit]
        : [];
    return {
      hit: payload.hit || hits[0] || null,
      hits,
    };
  } catch (_err) {
    return { hit: null, hits: [] };
  }
}

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

function updateLocalObjectPosition(kind, id, x, y) {
  const item = getObjectById(kind, id);
  if (!item) {
    return;
  }
  item.x = x;
  item.y = y;
}

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
  const pixelY = toGridPixel(pos.y);
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

function updateObjectDrag(pos) {
  if (!dragState) {
    return;
  }

  const pixelX = toGridPixel(pos.x);
  const pixelY = toGridPixel(pos.y);
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

async function placeTextAt(pos) {
  const response = await postJson("/api/board/text-objects", {
    text: boardText.value || "",
    x: toGridPixel(pos.x),
    y: toGridPixel(pos.y),
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

settingsToggle.addEventListener("click", () => {
  if (currentMode === "font_preview" && fontPreviewSettings) {
    fontPreviewSettings.classList.remove("hidden");
    return;
  }
  const panel = activeSettingsPanel();
  if (!panel) {
    return;
  }
  const shouldOpen = panel.classList.contains("hidden");
  hideAllSettingsPanels();
  panel.classList.toggle("hidden", !shouldOpen);
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
  sourceText.textContent = "Source: web";

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
  sourceText.textContent = "Source: web";

  if (isBoardMode() && activeBoardTool === "text") {
    await placeTextAt(pos);
    return;
  }

  if (isBoardMode() && activeBoardTool === "image") {
    await uploadImageAt(toGridPixel(pos.x), toGridPixel(pos.y));
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
      sourceText.textContent = "Source: web";
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
      sourceText.textContent = "Source: web";
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

function startWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    statusText.textContent = "Connected";
    if (pollingTimer !== null) {
      window.clearInterval(pollingTimer);
      pollingTimer = null;
    }
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.version === lastVersion) {
      return;
    }
    lastVersion = data.version;
    latestFramePixels = clonePixels(data.pixels);
    setMode(data.mode);
    renderControls(data.controls);
    drawGrid(data.pixels);
    if (isBoardMode() && boardDrawToggle.checked && boardStrokeActive && boardTool.value !== "freehand") {
      renderShapePreview(boardLastPoint || boardShapeStart);
    }
  };

  ws.onclose = () => {
    statusText.textContent = "WebSocket unavailable, using HTTP fallback";
    if (pollingTimer === null) {
      pollingTimer = window.setInterval(async () => {
        try {
          const response = await fetch("/api/frame", { cache: "no-store" });
          if (!response.ok) {
            return;
          }
          const data = await response.json();
          if (data.version === lastVersion) {
            return;
          }
          lastVersion = data.version;
          latestFramePixels = clonePixels(data.pixels);
          setMode(data.mode);
          renderControls(data.controls);
          drawGrid(data.pixels);
          if (isBoardMode() && boardDrawToggle.checked && boardStrokeActive && boardTool.value !== "freehand") {
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
  const approved = window.confirm("Clear the board and remove all objects? This can be undone once.");
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
loadFontPreviewSettings();
startWebSocket();
