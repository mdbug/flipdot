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

/**
 * Render Markdown to sanitized HTML.
 * @param {string} raw - Raw Markdown (or plain) text.
 * @returns {string} Sanitized HTML safe to inject.
 */
function renderMarkdown(raw) {
  const html = window.marked ? marked.parse(raw) : raw;
  return window.DOMPurify ? DOMPurify.sanitize(html) : html;
}

/** @returns {boolean} Whether the chat log contains any message, tool, or thinking block. */
function hasMessages() {
  return chatLog.querySelector(".chat-bubble, .chat-tool, .chat-thinking-block") !== null;
}

/** Show or hide the empty-state placeholder based on whether messages exist. */
function updateEmptyState() {
  if (!chatEmpty) return;
  chatEmpty.classList.toggle("hidden", hasMessages());
}

/** Scroll the chat log to the most recent message. */
function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

/** Grow the input textarea to fit its content, capped at 9 lines. */
function autoGrow() {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 9 * 16)}px`;
}

/** Reset the input textarea back to its default single-line height. */
function resetInputHeight() {
  chatInput.style.height = "auto";
}

/**
 * Append a hover "copy to clipboard" button to a message bubble.
 * @param {HTMLElement} bubble - The bubble element to attach the button to.
 */
function addCopyButton(bubble) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "chat-copy";
  btn.title = "Copy message";
  btn.setAttribute("aria-label", "Copy message");
  btn.innerHTML = '<span aria-hidden="true">⧉</span>';
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bubble.dataset.raw || bubble.textContent);
      btn.classList.add("copied");
      btn.innerHTML = '<span aria-hidden="true">✓</span>';
      setTimeout(() => {
        btn.classList.remove("copied");
        btn.innerHTML = '<span aria-hidden="true">⧉</span>';
      }, 1200);
    } catch (_err) {
      /* clipboard unavailable */
    }
  });
  bubble.appendChild(btn);
}

/**
 * Create and append a message bubble to the chat log.
 * @param {string} role - "user", "assistant", or "error".
 * @param {string} [text] - Initial message text.
 * @returns {HTMLElement} The created bubble element.
 */
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

/**
 * Append streamed text to an assistant bubble and re-render its Markdown.
 * @param {HTMLElement} bubble - The assistant bubble being streamed into.
 * @param {string} text - The text chunk to append.
 */
function appendAssistantText(bubble, text) {
  bubble.dataset.raw = (bubble.dataset.raw || "") + text;
  bubble.innerHTML = renderMarkdown(bubble.dataset.raw);
  addCopyButton(bubble);
  scrollToBottom();
}

// --- Thinking indicator -----------------------------------------------------

let thinkingEl = null;

/** Show the animated "Claude is thinking" indicator (idempotent). */
function showThinking() {
  if (thinkingEl) return;
  thinkingEl = document.createElement("div");
  thinkingEl.className = "chat-thinking";
  thinkingEl.setAttribute("aria-label", "Claude is thinking");
  thinkingEl.innerHTML = "<span></span><span></span><span></span>";
  chatLog.appendChild(thinkingEl);
  scrollToBottom();
}

/** Remove the animated "thinking" indicator if present. */
function hideThinking() {
  if (thinkingEl) {
    thinkingEl.remove();
    thinkingEl = null;
  }
}

// --- Streamed reasoning (extended thinking) ---------------------------------

let thinkingBlock = null;

/**
 * Append streamed extended-thinking text to the collapsible reasoning block.
 * @param {string} text - The reasoning chunk to append.
 */
function appendThinking(text) {
  if (!thinkingBlock) {
    thinkingBlock = document.createElement("details");
    thinkingBlock.className = "chat-thinking-block";
    thinkingBlock.open = true;
    const summary = document.createElement("summary");
    summary.textContent = "\u{1F4AD} Thinking…";
    const body = document.createElement("div");
    body.className = "chat-thinking-body";
    body.dataset.raw = "";
    thinkingBlock.append(summary, body);
    chatLog.appendChild(thinkingBlock);
    updateEmptyState();
  }
  const body = thinkingBlock.querySelector(".chat-thinking-body");
  body.dataset.raw += text;
  body.innerHTML = renderMarkdown(body.dataset.raw);
  scrollToBottom();
}

/**
 * Collapse the live reasoning into a "Thought for a moment" toggle once the
 * real answer (or a tool call) begins. Leaves the next thinking segment fresh.
 */
function settleThinking() {
  if (thinkingBlock) {
    thinkingBlock.open = false;
    const summary = thinkingBlock.querySelector("summary");
    if (summary) summary.textContent = "Thought for a moment";
    thinkingBlock = null;
  }
}

// --- Tool pills -------------------------------------------------------------

let lastToolPill = null;

/**
 * Format a tool's input object into a short "key: value" summary string.
 * @param {Object} input - The tool input arguments.
 * @returns {string} A truncated, comma-joined summary.
 */
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

/** Mark the most recent tool pill as finished (stops its running animation). */
function settleLastTool() {
  if (lastToolPill) {
    lastToolPill.classList.remove("running");
    lastToolPill.classList.add("is-done");
    lastToolPill = null;
  }
}

/**
 * Append a running tool-call pill to the chat log.
 * @param {string} name - The tool name.
 * @param {Object} input - The tool input arguments.
 */
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

/**
 * Enable or disable the chat input controls, optionally showing a note.
 * @param {boolean} enabled - Whether input is allowed.
 * @param {string} [note] - Optional status note to display.
 */
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

/** Poll the backend chat status and enable/disable the UI accordingly. */
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

/**
 * Send a user message and stream the NDJSON response, rendering text,
 * reasoning, tool calls, and errors as they arrive.
 * @param {string} message - The user's message text.
 */
async function submitMessage(message) {
  addBubble("user", message);
  setEnabled(false);
  showThinking();
  const controller = new AbortController();
  activeController = controller;
  let assistantBubble = null;
  const ensureAssistant = () => {
    hideThinking();
    settleThinking();
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
        if (event.type === "thinking") {
          hideThinking();
          appendThinking(event.text);
        } else if (event.type === "text") {
          appendAssistantText(ensureAssistant(), event.text);
        } else if (event.type === "tool") {
          hideThinking();
          settleThinking();
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
    settleThinking();
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
  chatLog
    .querySelectorAll(".chat-bubble, .chat-tool, .chat-thinking, .chat-thinking-block")
    .forEach((el) => el.remove());
  lastToolPill = null;
  thinkingEl = null;
  thinkingBlock = null;
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
