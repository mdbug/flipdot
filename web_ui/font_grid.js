const statusText = document.getElementById("statusText");
const glyphTable = document.getElementById("glyphTable");
const glyphColGroup = document.getElementById("glyphColGroup");
const glyphHead = glyphTable.querySelector("thead");
const glyphBody = glyphTable.querySelector("tbody");
const tableWrap = document.getElementById("tableWrap");
const filterFamily = document.getElementById("filterFamily");
const filterSize = document.getElementById("filterSize");
const filterStyle = document.getElementById("filterStyle");
const viewDensity = document.getElementById("viewDensity");
const zoomLevel = document.getElementById("zoomLevel");
const zoomValue = document.getElementById("zoomValue");

const ON_COLOR = "#ffffff";
const OFF_COLOR = "#39404b";
const DOT_BASE = "#0b0d10";

let allVariants = [];
let allCharacters = [];
let selectedRowIndex = -1;
let renderSequence = 0;
let currentVisibleVariants = [];
let glyphHeadCurrent = null;

function getZoomFactor() {
  if (!zoomLevel) {
    return 1;
  }
  const parsed = Number.parseInt(zoomLevel.value, 10);
  return Number.isFinite(parsed) ? Math.max(0.5, Math.min(3, parsed / 100)) : 1;
}

function getColumnWidths(density) {
  const zoom = getZoomFactor();
  const base = density === "comfortable" ? 62 : 54;
  const scaled = Math.round(base * zoom);
  return { glyph: scaled, variant: scaled };
}

function setStatus(message) {
  statusText.textContent = message;
}

function escapeChar(char) {
  if (char === " ") {
    return "SPACE";
  }
  if (char === "\t") {
    return "TAB";
  }
  if (char === "\n") {
    return "LF";
  }
  return char;
}

function charCodeLabel(char) {
  if (!char || typeof char !== "string") {
    return "";
  }
  return `U+${char.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")}`;
}

function drawGlyphCanvas(canvas, rows, density) {
  const glyphRows = Array.isArray(rows) ? rows : [];
  const h = glyphRows.length;
  const w = Math.max(1, ...glyphRows.map((row) => (Array.isArray(row) ? row.length : 0)));

  const zoom = getZoomFactor();
  const baseCell = density === "comfortable" ? 7 : 5;
  const cell = Math.max(2, Math.round(baseCell * zoom));
  const pad = 1;
  canvas.width = w * cell + pad * 2;
  canvas.height = h * cell + pad * 2;

  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0d1015";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const radius = cell * 0.36;

  for (let y = 0; y < h; y += 1) {
    const row = Array.isArray(glyphRows[y]) ? glyphRows[y] : [];
    for (let x = 0; x < row.length; x += 1) {
      const cx = pad + x * cell + cell * 0.5;
      const cy = pad + y * cell + cell * 0.5;
      const on = row[x] === 1;

      ctx.beginPath();
      ctx.arc(cx, cy, radius + cell * 0.07, 0, Math.PI * 2);
      ctx.fillStyle = DOT_BASE;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = on ? ON_COLOR : OFF_COLOR;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx - radius * 0.3, cy - radius * 0.3, radius * 0.34, 0, Math.PI * 2);
      ctx.fillStyle = on ? "rgba(255,255,255,0.23)" : "rgba(255,255,255,0.07)";
      ctx.fill();
    }
  }
}

function renderColGroup(variantCount, density) {
  if (!glyphColGroup) {
    return;
  }
  glyphColGroup.innerHTML = "";
  const widths = getColumnWidths(density);

  const glyphColumn = document.createElement("col");
  glyphColumn.style.width = `${widths.glyph}px`;
  glyphColGroup.appendChild(glyphColumn);

  for (let i = 0; i < variantCount; i += 1) {
    const col = document.createElement("col");
    col.style.width = `${widths.variant}px`;
    glyphColGroup.appendChild(col);
  }
}

function createOption(value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  return option;
}

function populateFilters() {
  const families = new Set();
  const sizes = new Set();
  const styles = new Set();

  for (const variant of allVariants) {
    families.add(String(variant.family));
    sizes.add(String(variant.size));
    styles.add(String(variant.style));
  }

  filterFamily.innerHTML = "";
  filterSize.innerHTML = "";
  filterStyle.innerHTML = "";

  filterFamily.appendChild(createOption("", "All families"));
  filterSize.appendChild(createOption("", "All sizes"));
  filterStyle.appendChild(createOption("", "All styles"));

  Array.from(families)
    .sort()
    .forEach((value) => filterFamily.appendChild(createOption(value, value)));

  Array.from(sizes)
    .sort((a, b) => Number(a) - Number(b))
    .forEach((value) => filterSize.appendChild(createOption(value, value)));

  Array.from(styles)
    .sort()
    .forEach((value) => filterStyle.appendChild(createOption(value, value)));
}

function filterVariants() {
  const family = filterFamily.value;
  const size = filterSize.value;
  const style = filterStyle.value;

  return allVariants.filter((variant) => {
    if (family && variant.family !== family) {
      return false;
    }
    if (size && String(variant.size) !== size) {
      return false;
    }
    if (style && variant.style !== style) {
      return false;
    }
    return true;
  });
}

function setActiveRow(index, shouldScroll = false) {
  const rows = Array.from(glyphBody.querySelectorAll("tr"));
  rows.forEach((row) => row.classList.remove("glyph-row-active"));

  if (!rows.length) {
    selectedRowIndex = -1;
    updateHeaderPreview();
    return;
  }

  const clamped = Math.max(0, Math.min(rows.length - 1, index));
  selectedRowIndex = clamped;
  const active = rows[clamped];
  active.classList.add("glyph-row-active");
  if (shouldScroll) {
    active.scrollIntoView({ block: "nearest", inline: "nearest" });
  }
  updateHeaderPreview();
}

function updateHeaderPreview() {
  if (!glyphHeadCurrent) {
    return;
  }

  if (selectedRowIndex < 0 || selectedRowIndex >= allCharacters.length || currentVisibleVariants.length === 0) {
    glyphHeadCurrent.textContent = "Current row: -";
    for (const canvas of glyphHead.querySelectorAll(".variant-head-preview .glyph-canvas")) {
      drawGlyphCanvas(canvas, [], viewDensity.value);
    }
    return;
  }

  const char = allCharacters[selectedRowIndex];
  glyphHeadCurrent.textContent = `Current row: ${escapeChar(char)} (${charCodeLabel(char)})`;
  const density = viewDensity.value;

  const previewCanvases = Array.from(glyphHead.querySelectorAll(".variant-head-preview .glyph-canvas"));
  for (let i = 0; i < previewCanvases.length; i += 1) {
    const variant = currentVisibleVariants[i];
    const rows = variant ? (variant.glyphs && variant.glyphs[char]) || [] : [];
    drawGlyphCanvas(previewCanvases[i], rows, density);
  }
}

function bindRowHoverSelection() {
  const rows = Array.from(glyphBody.querySelectorAll("tr"));
  rows.forEach((row, index) => {
    row.addEventListener("mouseenter", () => {
      setActiveRow(index, false);
    });
  });
}

function renderRowsChunked(variants, density) {
  const runId = ++renderSequence;
  let index = 0;
  const chunkSize = 8;

  const renderChunk = () => {
    if (runId !== renderSequence) {
      return;
    }
    const end = Math.min(index + chunkSize, allCharacters.length);
    for (; index < end; index += 1) {
      const char = allCharacters[index];
      const row = document.createElement("tr");
      const charCell = document.createElement("td");
      charCell.className = "char-cell";
      const symbol = document.createElement("span");
      symbol.textContent = escapeChar(char);
      const code = document.createElement("span");
      code.className = "char-code";
      code.textContent = charCodeLabel(char);
      charCell.append(symbol, code);
      row.appendChild(charCell);

      for (const variant of variants) {
        const td = document.createElement("td");
        const canvas = document.createElement("canvas");
        canvas.className = "glyph-canvas";
        const rows = (variant.glyphs && variant.glyphs[char]) || [];
        drawGlyphCanvas(canvas, rows, density);
        td.appendChild(canvas);
        row.appendChild(td);
      }

      glyphBody.appendChild(row);
    }

    if (index < allCharacters.length) {
      setStatus(`Rendering ${index}/${allCharacters.length} glyph rows...`);
      requestAnimationFrame(renderChunk);
      return;
    }

    bindRowHoverSelection();
    setActiveRow(0, false);
    setStatus(`Showing ${allCharacters.length} glyphs across ${variants.length} variants. Use arrow keys to scan rows.`);
  };

  requestAnimationFrame(renderChunk);
}

function renderTable() {
  const variants = filterVariants();
  currentVisibleVariants = variants;
  glyphHead.innerHTML = "";
  glyphBody.innerHTML = "";
  renderSequence += 1;

  document.body.classList.toggle("density-comfortable", viewDensity.value === "comfortable");

  if (!variants.length) {
    setStatus("No variants match current filters.");
    updateHeaderPreview();
    return;
  }

  const headRow = document.createElement("tr");
  const charHead = document.createElement("th");
  const glyphMain = document.createElement("div");
  glyphMain.className = "glyph-head-main";
  glyphMain.textContent = "Glyph";
  glyphHeadCurrent = document.createElement("div");
  glyphHeadCurrent.className = "glyph-head-current";
  glyphHeadCurrent.textContent = "Current row: -";
  charHead.append(glyphMain, glyphHeadCurrent);
  headRow.appendChild(charHead);

  for (const variant of variants) {
    const th = document.createElement("th");
    th.className = "variant-head";
    const label = document.createElement("div");
    label.className = "variant-head-label";
    label.innerHTML = `${variant.family}<br><span class="variant-meta">${variant.size} / ${variant.style}</span>`;
    const preview = document.createElement("div");
    preview.className = "variant-head-preview";
    const previewCanvas = document.createElement("canvas");
    previewCanvas.className = "glyph-canvas";
    drawGlyphCanvas(previewCanvas, [], viewDensity.value);
    preview.appendChild(previewCanvas);
    th.append(label, preview);
    headRow.appendChild(th);
  }
  glyphHead.appendChild(headRow);

  const density = viewDensity.value;
  renderColGroup(variants.length, density);
  renderRowsChunked(variants, density);
}

async function loadGlyphGrid() {
  try {
    const response = await fetch("/api/font-preview/glyph-grid", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`request failed: ${response.status}`);
    }
    const payload = await response.json();
    allVariants = Array.isArray(payload.variants) ? payload.variants : [];
    allCharacters = Array.isArray(payload.characters) ? payload.characters : [];

    populateFilters();
    renderTable();
  } catch (_err) {
    setStatus("Failed to load glyph catalog. Ensure font preview mode is attached.");
  }
}

for (const element of [filterFamily, filterSize, filterStyle, viewDensity]) {
  element.addEventListener("change", () => {
    renderTable();
  });
}

if (zoomLevel) {
  zoomLevel.addEventListener("input", () => {
    if (zoomValue) {
      zoomValue.textContent = `${zoomLevel.value}%`;
    }
    renderTable();
  });
}

if (tableWrap) {
  tableWrap.addEventListener("keydown", (event) => {
    const pageJump = Math.max(8, Math.floor((tableWrap.clientHeight || 320) / 32));
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveRow(selectedRowIndex + 1, true);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveRow(selectedRowIndex - 1, true);
      return;
    }
    if (event.key === "Home") {
      event.preventDefault();
      setActiveRow(0, true);
      return;
    }
    if (event.key === "End") {
      event.preventDefault();
      setActiveRow(allCharacters.length - 1, true);
      return;
    }
    if (event.key === "PageDown") {
      event.preventDefault();
      setActiveRow(selectedRowIndex + pageJump, true);
      return;
    }
    if (event.key === "PageUp") {
      event.preventDefault();
      setActiveRow(selectedRowIndex - pageJump, true);
    }
  });
}

loadGlyphGrid();
