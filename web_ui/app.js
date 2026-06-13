const canvas = document.getElementById("matrix");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("statusText");
const sourceText = document.getElementById("sourceText");
const modeControls = document.getElementById("modeControls");

const GRID = 28;
const HAS_POINTER_EVENTS = "PointerEvent" in window;
let lastVersion = -1;
let pollingTimer = null;
let lastTouchPos = null;
let controlsSignature = "";

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

      // Dot cavity/background to mimic panel holes.
      ctx.beginPath();
      ctx.arc(cx, cy, radius + cell * 0.04, 0, Math.PI * 2);
      ctx.fillStyle = "#0d1420";
      ctx.fill();

      // Main flipdot face.
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = on ? "#f2e8ba" : "#253346";
      ctx.fill();

      // Subtle top-left highlight for physical feel.
      ctx.beginPath();
      ctx.arc(cx - radius * 0.32, cy - radius * 0.32, radius * 0.32, 0, Math.PI * 2);
      ctx.fillStyle = on ? "rgba(255,255,255,0.24)" : "rgba(255,255,255,0.07)";
      ctx.fill();
    }
  }
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
    await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (_err) {
    // Ignore transient network failures; websocket status handles UX.
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
  postJson("/api/input/pointer", pos);
});

canvas.addEventListener("pointerdown", (event) => {
  const pos = normPosFromEvent(event);
  sourceText.textContent = "Source: web";
  postJson("/api/input/pointer", pos);
  postJson("/api/input/button", { down: true });
});

canvas.addEventListener("pointerup", (event) => {
  const pos = normPosFromEvent(event);
  postJson("/api/input/button", { down: false });
  postJson("/api/input/click", pos);
});

canvas.addEventListener("pointercancel", () => {
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
      postJson("/api/input/pointer", pos);
    }
  },
  { passive: false }
);

canvas.addEventListener(
  "touchend",
  (event) => {
    event.preventDefault();
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
    renderControls(data.controls);
    drawGrid(data.pixels);
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
          renderControls(data.controls);
          drawGrid(data.pixels);
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

startWebSocket();
