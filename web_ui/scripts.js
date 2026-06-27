const scriptListEl = document.getElementById("scriptList");
const scriptListEmpty = document.getElementById("scriptListEmpty");
const scriptListError = document.getElementById("scriptListError");
const scriptsRefresh = document.getElementById("scriptsRefresh");
const scriptsPlaceholder = document.getElementById("scriptsPlaceholder");
const scriptsDetail = document.getElementById("scriptsDetail");
const scriptsDetailName = document.getElementById("scriptsDetailName");
const scriptsDetailStatus = document.getElementById("scriptsDetailStatus");
const scriptsCodeEl = document.getElementById("scriptsCode").querySelector("code");
const scriptsPlay = document.getElementById("scriptsPlay");
const scriptsDelete = document.getElementById("scriptsDelete");
const scriptsActionStatus = document.getElementById("scriptsActionStatus");

let scripts = [];
let selectedName = null;
let activeName = "";

// ── List ─────────────────────────────────────────────────────────────────────

async function loadList() {
  scriptListError.classList.add("hidden");
  try {
    const res = await fetch("/api/scripts");
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    activeName = data.active || "";
    scripts = (data.scripts || []).map((name) => ({ name }));
    renderList();
  } catch (err) {
    scriptListError.textContent = `Failed to load scripts: ${err.message}`;
    scriptListError.classList.remove("hidden");
  }
}

function renderList() {
  scriptListEl.textContent = "";
  scriptListEmpty.classList.toggle("hidden", scripts.length > 0);

  for (const { name } of scripts) {
    const li = document.createElement("li");
    li.className = "script-item";
    if (name === selectedName) li.classList.add("active");

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "script-open";
    btn.addEventListener("click", () => selectScript(name));

    const nameSpan = document.createElement("span");
    nameSpan.className = "script-name";
    nameSpan.textContent = name;
    btn.appendChild(nameSpan);

    if (name === activeName) {
      const badge = document.createElement("span");
      badge.className = "script-active-badge";
      badge.textContent = "playing";
      btn.appendChild(badge);
    }

    li.appendChild(btn);
    scriptListEl.appendChild(li);
  }

  updateDetailStatus();
}

// ── Detail ────────────────────────────────────────────────────────────────────

async function selectScript(name) {
  selectedName = name;
  renderList();

  scriptsPlaceholder.classList.add("hidden");
  scriptsDetail.classList.remove("hidden");
  scriptsDetailName.textContent = name;
  updateDetailStatus();
  scriptsCodeEl.textContent = "Loading…";
  setActionStatus("", false);

  try {
    const res = await fetch(`/api/scripts/${encodeURIComponent(name)}/code`);
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    scriptsCodeEl.textContent = data.code;
  } catch (err) {
    scriptsCodeEl.textContent = `Error loading code: ${err.message}`;
  }
}

function updateDetailStatus() {
  if (!selectedName) return;
  const isActive = selectedName === activeName;
  scriptsDetailStatus.textContent = isActive ? "playing" : "idle";
  scriptsDetailStatus.className = isActive ? "status-pill ok" : "status-pill muted";
}

// ── Actions ───────────────────────────────────────────────────────────────────

function setActionStatus(msg, isError) {
  if (!msg) {
    scriptsActionStatus.classList.add("hidden");
    return;
  }
  scriptsActionStatus.textContent = msg;
  scriptsActionStatus.className = isError ? "scripts-status error" : "scripts-status ok";
  scriptsActionStatus.classList.remove("hidden");
}

scriptsPlay.addEventListener("click", async () => {
  if (!selectedName) return;
  scriptsPlay.disabled = true;
  setActionStatus("", false);
  try {
    const res = await fetch(`/api/scripts/${encodeURIComponent(selectedName)}/play`, {
      method: "POST",
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `${res.status}`);
    activeName = selectedName;
    renderList();
    setActionStatus("Playing.", false);
  } catch (err) {
    setActionStatus(`Error: ${err.message}`, true);
  } finally {
    scriptsPlay.disabled = false;
  }
});

scriptsDelete.addEventListener("click", async () => {
  if (!selectedName) return;
  if (!confirm(`Delete script "${selectedName}"?`)) return;
  scriptsDelete.disabled = true;
  setActionStatus("", false);
  const nameToDelete = selectedName;
  try {
    const res = await fetch(`/api/scripts/${encodeURIComponent(nameToDelete)}`, {
      method: "DELETE",
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `${res.status}`);
    selectedName = null;
    if (activeName === nameToDelete) activeName = "";
    scriptsDetail.classList.add("hidden");
    scriptsPlaceholder.classList.remove("hidden");
    await loadList();
  } catch (err) {
    setActionStatus(`Error: ${err.message}`, true);
  } finally {
    scriptsDelete.disabled = false;
  }
});

scriptsRefresh.addEventListener("click", loadList);

loadList();
