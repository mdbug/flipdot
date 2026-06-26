// Standalone Claude chat page. Talks to the same /api/chat backend the rest of
// the console exposes; the backend runs the agentic loop and drives the display
// through the MCP tools.

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatSend = document.getElementById("chatSend");
const chatStop = document.getElementById("chatStop");
const chatReset = document.getElementById("chatReset");
const chatNote = document.getElementById("chatNote");
const chatEmpty = document.getElementById("chatEmpty");

let chatBusy = false;
let activeController = null;

if (window.marked) {
  marked.setOptions({ breaks: true, gfm: true });
}

function renderMarkdown(raw) {
  const html = window.marked ? marked.parse(raw) : raw;
  return window.DOMPurify ? DOMPurify.sanitize(html) : html;
}

function hasMessages() {
  return chatLog.querySelector(".chat-bubble, .chat-tool") !== null;
}

function updateEmptyState() {
  if (!chatEmpty) return;
  chatEmpty.classList.toggle("hidden", hasMessages());
}

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function autoGrow() {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 9 * 16)}px`;
}

function resetInputHeight() {
  chatInput.style.height = "auto";
}

function addCopyButton(bubble) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "chat-copy";
  btn.title = "Copy message";
  btn.setAttribute("aria-label", "Copy message");
  btn.innerHTML = "<span aria-hidden=\"true\">⧉</span>";
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bubble.dataset.raw || bubble.textContent);
      btn.classList.add("copied");
      btn.innerHTML = "<span aria-hidden=\"true\">✓</span>";
      setTimeout(() => {
        btn.classList.remove("copied");
        btn.innerHTML = "<span aria-hidden=\"true\">⧉</span>";
      }, 1200);
    } catch (_err) {
      /* clipboard unavailable */
    }
  });
  bubble.appendChild(btn);
}

function addBubble(role, text = "") {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble chat-${role}`;
  if (role === "assistant") {
    bubble.dataset.raw = text;
    bubble.innerHTML = renderMarkdown(text);
    addCopyButton(bubble);
  } else {
    bubble.textContent = text;
  }
  chatLog.appendChild(bubble);
  updateEmptyState();
  scrollToBottom();
  return bubble;
}

function appendAssistantText(bubble, text) {
  bubble.dataset.raw = (bubble.dataset.raw || "") + text;
  bubble.innerHTML = renderMarkdown(bubble.dataset.raw);
  addCopyButton(bubble);
  scrollToBottom();
}

// --- Thinking indicator -----------------------------------------------------

let thinkingEl = null;

function showThinking() {
  if (thinkingEl) return;
  thinkingEl = document.createElement("div");
  thinkingEl.className = "chat-thinking";
  thinkingEl.setAttribute("aria-label", "Claude is thinking");
  thinkingEl.innerHTML = "<span></span><span></span><span></span>";
  chatLog.appendChild(thinkingEl);
  scrollToBottom();
}

function hideThinking() {
  if (thinkingEl) {
    thinkingEl.remove();
    thinkingEl = null;
  }
}

// --- Tool pills -------------------------------------------------------------

let lastToolPill = null;

function formatToolInput(input) {
  if (!input || typeof input !== "object") return "";
  const parts = [];
  for (const [key, value] of Object.entries(input)) {
    let rendered = typeof value === "string" ? value : JSON.stringify(value);
    if (rendered.length > 40) rendered = `${rendered.slice(0, 39)}…`;
    parts.push(`${key}: ${rendered}`);
  }
  return parts.join(", ");
}

function settleLastTool() {
  if (lastToolPill) {
    lastToolPill.classList.remove("running");
    lastToolPill.classList.add("is-done");
    lastToolPill = null;
  }
}

function addTool(name, input) {
  settleLastTool();
  const pill = document.createElement("div");
  pill.className = "chat-tool running";
  const label = document.createElement("span");
  label.className = "chat-tool-name";
  label.textContent = `\u{1F6E0} ${name}`;
  pill.appendChild(label);
  const args = formatToolInput(input);
  if (args) {
    const argsEl = document.createElement("span");
    argsEl.className = "chat-tool-args";
    argsEl.textContent = args;
    pill.appendChild(argsEl);
  }
  chatLog.appendChild(pill);
  lastToolPill = pill;
  updateEmptyState();
  scrollToBottom();
}

// --- Input enable/disable ---------------------------------------------------

function setEnabled(enabled, note) {
  chatBusy = !enabled;
  chatInput.disabled = !enabled;
  chatSend.disabled = !enabled;
  chatStop.classList.toggle("hidden", enabled);
  if (note) {
    chatNote.textContent = note;
    chatNote.classList.remove("hidden");
  } else {
    chatNote.classList.add("hidden");
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/chat/status", { cache: "no-store" });
    const status = await response.json();
    if (!status.available) {
      setEnabled(false, "Chat is off. Set ANTHROPIC_API_KEY in the environment to enable it.");
      chatStop.classList.add("hidden");
    } else {
      setEnabled(true);
    }
  } catch (_err) {
    setEnabled(false, "Chat status unavailable.");
    chatStop.classList.add("hidden");
  }
}

async function submitMessage(message) {
  addBubble("user", message);
  setEnabled(false);
  showThinking();
  const controller = new AbortController();
  activeController = controller;
  let assistantBubble = null;
  const ensureAssistant = () => {
    hideThinking();
    settleLastTool();
    if (!assistantBubble) assistantBubble = addBubble("assistant", "");
    return assistantBubble;
  };

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      throw new Error(`request failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        let event;
        try {
          event = JSON.parse(line);
        } catch (_err) {
          continue;
        }
        if (event.type === "text") {
          appendAssistantText(ensureAssistant(), event.text);
        } else if (event.type === "tool") {
          hideThinking();
          assistantBubble = null; // start a fresh bubble after a tool call
          addTool(event.name, event.input);
          showThinking();
        } else if (event.type === "error") {
          hideThinking();
          addBubble("error", event.message || "Chat error");
        }
      }
    }
  } catch (err) {
    if (err && err.name === "AbortError") {
      // Clean stop requested by the user — no error bubble.
    } else {
      addBubble("error", String(err && err.message ? err.message : err));
    }
  } finally {
    hideThinking();
    settleLastTool();
    activeController = null;
    setEnabled(true);
    chatInput.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (chatBusy) return;
  const message = chatInput.value.trim();
  if (!message) return;
  chatInput.value = "";
  resetInputHeight();
  submitMessage(message);
});

chatInput.addEventListener("input", autoGrow);

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

chatStop.addEventListener("click", () => {
  if (activeController) activeController.abort();
});

chatReset.addEventListener("click", async () => {
  if (chatBusy) return;
  try {
    await fetch("/api/chat/reset", { method: "POST" });
  } catch (_err) {
    /* ignore */
  }
  chatLog.querySelectorAll(".chat-bubble, .chat-tool, .chat-thinking").forEach((el) => el.remove());
  lastToolPill = null;
  thinkingEl = null;
  updateEmptyState();
  chatInput.focus();
});

if (chatEmpty) {
  chatEmpty.querySelectorAll(".chat-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      chatInput.value = chip.textContent;
      autoGrow();
      chatInput.focus();
    });
  });
}

updateEmptyState();
refreshStatus();
