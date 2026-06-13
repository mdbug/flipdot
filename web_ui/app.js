const canvas = document.getElementById("matrix");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("statusText");
const sourceText = document.getElementById("sourceText");
const modeControls = document.getElementById("modeControls");
const boardEditor = document.getElementById("boardEditor");

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
const boardTextScroll = document.getElementById("boardTextScroll");
const boardScrollSpeed = document.getElementById("boardScrollSpeed");
const boardSelectionSummary = document.getElementById("boardSelectionSummary");
const boardDeleteSelection = document.getElementById("boardDeleteSelection");

const boardClear = document.getElementById("boardClear");
const boardUndo = document.getElementById("boardUndo");
const boardDrawToggle = document.getElementById("boardDrawToggle");
const boardTool = document.getElementById("boardTool");
const toolbarButtons = Array.from(document.querySelectorAll(".tool-btn[data-tool]"));

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

function clonePixels(pixels) {
  const out = Array.from({ length: GRID }, () => Array(GRID).fill(0));
  for (let y = 0; y < GRID; y += 1) {
    for (let x = 0; x < GRID; x += 1) {
      out[y][x] = pixels[y] && pixels[y][x] === 1 ? 1 : 0;
    }
  }
  return out;
}

function drawGrid(pixels) {
  const cell = canvas.width / GRID;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#141d2a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < GRID; y += 1) {
    for (let x = 0; x < GRID; x += 1) {
      const on = pixels[y] && pixels[y][x] === 1;
      const cx = x * cell + cell * 0.5;
      const cy = y * cell + cell * 0.5;
      const radius = cell * 0.45;

      ctx.beginPath();
      ctx.arc(cx, cy, radius + cell * 0.04, 0, Math.PI * 2);
      ctx.fillStyle = "#0d1420";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = on ? "#f2e8ba" : "#253346";
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

function setPreviewPixel(pixels, x, y) {
  if (x < 0 || x >= GRID || y < 0 || y >= GRID) {
    return;
  }
  pixels[y][x] = 1;
}

function pixelPointFromNorm(pos) {
  return {
    x: Math.trunc(Math.max(0, Math.min(1, Number(pos.x))) * (GRID - 1)),
    y: Math.trunc(Math.max(0, Math.min(1, Number(pos.y))) * (GRID - 1)),
  };
}

function rasterLine(pixels, p0, p1) {
  const dx = p1.x - p0.x;
  const dy = p1.y - p0.y;
  const steps = Math.max(Math.abs(dx), Math.abs(dy), 1);
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    const x = Math.round(p0.x + dx * t);
    const y = Math.round(p0.y + dy * t);
    setPreviewPixel(pixels, x, y);
  }
}

function rasterRect(pixels, p0, p1) {
  const minX = Math.min(p0.x, p1.x);
  const maxX = Math.max(p0.x, p1.x);
  const minY = Math.min(p0.y, p1.y);
  const maxY = Math.max(p0.y, p1.y);
  rasterLine(pixels, { x: minX, y: minY }, { x: maxX, y: minY });
  rasterLine(pixels, { x: maxX, y: minY }, { x: maxX, y: maxY });
  rasterLine(pixels, { x: maxX, y: maxY }, { x: minX, y: maxY });
  rasterLine(pixels, { x: minX, y: maxY }, { x: minX, y: minY });
}

function rasterCircle(pixels, p0, p1) {
  const cx = Math.round((p0.x + p1.x) / 2);
  const cy = Math.round((p0.y + p1.y) / 2);
  const radius = Math.max(1, Math.round(Math.max(Math.abs(p1.x - p0.x), Math.abs(p1.y - p0.y)) / 2));
  for (let angle = 0; angle < 360; angle += 2) {
    const rad = (angle * Math.PI) / 180;
    const x = Math.round(cx + radius * Math.cos(rad));
    const y = Math.round(cy + radius * Math.sin(rad));
    setPreviewPixel(pixels, x, y);
  }
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

  if (tool === "line") {
    rasterLine(previewPixels, p0, p1);
  } else if (tool === "rectangle") {
    rasterRect(previewPixels, p0, p1);
  } else if (tool === "circle") {
    rasterCircle(previewPixels, p0, p1);
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
    boardTextScroll.checked = false;
    boardScrollSpeed.value = "7";
    return;
  }

  boardText.value = item.text || "";
  boardTextX.value = String(item.x ?? 0);
  boardTextY.value = String(item.y ?? 11);
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
  } else if (DRAW_TOOLS.has(activeBoardTool)) {
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
  updateCanvasToolClass();
}

function setBoardTool(nextTool) {
  const tool = typeof nextTool === "string" ? nextTool : "select";
  activeBoardTool = tool;

  if (DRAW_TOOLS.has(tool)) {
    previousDrawTool = tool;
    boardTool.value = tool;
    boardDrawToggle.checked = true;
  } else {
    boardDrawToggle.checked = false;
    if (DRAW_TOOLS.has(previousDrawTool)) {
      boardTool.value = previousDrawTool;
    }
  }

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
  await postJson("/api/board/draw", { points });
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
  const boardVisible = isBoardMode();
  boardEditor.classList.toggle("hidden", !boardVisible);
  if (boardVisible && previousMode !== currentMode) {
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
  }
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
    const button = document.createElement("button");
    button.type = "button";
    button.className = control.variant === "secondary" ? "secondary" : "accent";
    button.textContent = control.label;
    button.addEventListener("click", () => {
      postJson("/api/input/action", { action: control.action });
    });
    modeControls.appendChild(button);
  }
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
    clearSelection();
    setBoardTool("select");
  }
});

setBoardTool("select");
startWebSocket();
