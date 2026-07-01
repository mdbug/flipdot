// Standalone Claude chat page. Talks to the same /api/chat backend the rest of
// the console exposes; the backend runs the agentic loop, drives the display
// through the MCP tools, and auto-saves each conversation as a session that can
// be listed, resumed, renamed, or deleted from the sidebar.

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatSend = document.getElementById("chatSend");
const chatStop = document.getElementById("chatStop");
const chatNote = document.getElementById("chatNote");
const chatEmpty = document.getElementById("chatEmpty");
const chatModel = document.getElementById("chatModel");
const chatCount = document.getElementById("chatCount");
const chatTitle = document.getElementById("chatTitle");
const chatUsageTotal = document.getElementById("chatUsageTotal");
const chatNew = document.getElementById("chatNew");
const sessionList = document.getElementById("sessionList");
const sessionEmpty = document.getElementById("sessionEmpty");
const chatSidebar = document.getElementById("chatSidebar");
const chatSidebarToggle = document.getElementById("chatSidebarToggle");
const chatSidebarScrim = document.getElementById("chatSidebarScrim");

let chatBusy = false;
let activeController = null;
let currentSessionId = null;

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

/** Remove every rendered message/tool/thinking node, leaving the empty state. */
function clearLog() {
  chatLog
    .querySelectorAll(
      ".chat-bubble, .chat-tool, .chat-thinking, .chat-thinking-block, .chat-usage",
    )
    .forEach((el) => el.remove());
  lastToolPill = null;
  thinkingEl = null;
  thinkingBlock = null;
  updateEmptyState();
}

/** Scroll the chat log to the most recent message. */
function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

/** Grow the input textarea to fit its content, capped at 9 lines. */
function autoGrow() {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 9 * 22)}px`;
}

/** Reset the input textarea back to its default single-line height. */
function resetInputHeight() {
  chatInput.style.height = "auto";
}

/** Update the live character counter beneath the composer. */
function updateCount() {
  if (chatCount) chatCount.textContent = `${chatInput.value.length} / 2000`;
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

// --- Token usage & cost ------------------------------------------------------

/**
 * Format a token count compactly (e.g. 1234 -> "1.2k", 950 -> "950").
 * @param {number} n - Token count.
 * @returns {string} Compact count.
 */
function formatTokens(n) {
  const value = Number(n) || 0;
  if (value < 1000) return String(value);
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)}k`;
}

/**
 * Build a human summary of a usage record (tokens + optional estimated cost).
 * @param {Object} usage - {input, output, cache_write, cache_read, cost}.
 * @returns {string} e.g. "1.2k in · 340 out · 3.1k cached · ~$0.0210".
 */
function formatUsage(usage) {
  if (!usage) return "";
  const cached = (Number(usage.cache_read) || 0) + (Number(usage.cache_write) || 0);
  const parts = [`${formatTokens(usage.input)} in`, `${formatTokens(usage.output)} out`];
  if (cached > 0) parts.push(`${formatTokens(cached)} cached`);
  if (typeof usage.cost === "number") parts.push(`~$${usage.cost.toFixed(4)}`);
  return parts.join(" · ");
}

/**
 * Append a per-response usage line beneath the latest assistant reply.
 * @param {Object} usage - The turn's usage record.
 */
function addUsageLine(usage) {
  const text = formatUsage(usage);
  if (!text) return;
  const line = document.createElement("div");
  line.className = "chat-usage";
  line.textContent = text;
  chatLog.appendChild(line);
  scrollToBottom();
}

/**
 * Update (or hide) the running session-total badge in the chat bar.
 * @param {Object|null} usage - The cumulative session usage, or null to hide.
 */
function setSessionUsage(usage) {
  if (!chatUsageTotal) return;
  const text = formatUsage(usage);
  if (!text) {
    chatUsageTotal.textContent = "";
    chatUsageTotal.classList.add("hidden");
    return;
  }
  chatUsageTotal.textContent = `Σ ${text}`;
  chatUsageTotal.classList.remove("hidden");
}

// --- Session history rendering ----------------------------------------------

/**
 * Rebuild the chat log from a stored Anthropic message history.
 * @param {Array<Object>} messages - Saved messages (role + content blocks).
 */
function renderHistory(messages) {
  clearLog();
  for (const msg of messages || []) {
    const content = msg.content;
    if (msg.role === "user") {
      // Plain strings are real user turns; list content is a tool result that
      // rode along as a user-role message, which we don't surface.
      if (typeof content === "string") addBubble("user", content);
      continue;
    }
    if (msg.role !== "assistant") continue;
    if (typeof content === "string") {
      addBubble("assistant", content);
      continue;
    }
    for (const block of content || []) {
      if (block.type === "text" && block.text) {
        addBubble("assistant", block.text);
      } else if (block.type === "thinking" && block.thinking) {
        appendThinking(block.thinking);
        settleThinking();
      } else if (block.type === "tool_use") {
        addTool(block.name, block.input);
        settleLastTool();
      }
    }
  }
  updateEmptyState();
  scrollToBottom();
}

// --- Session sidebar --------------------------------------------------------

/**
 * Format an ISO timestamp into a short relative label (e.g. "5m", "2h", "3d").
 * @param {string} iso - ISO-8601 timestamp.
 * @returns {string} Compact relative-time string.
 */
function formatRelativeTime(iso) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (secs < 60) return "just now";
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

/** Update the active highlight in the sidebar to match currentSessionId. */
function markActiveSession() {
  sessionList.querySelectorAll(".session-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.id === currentSessionId);
  });
}

/**
 * Render the saved-session list into the sidebar.
 * @param {Array<Object>} sessions - Session summaries from the backend.
 */
function renderSessionList(sessions) {
  sessionList.innerHTML = "";
  if (!sessions.length) {
    sessionEmpty.classList.remove("hidden");
    return;
  }
  sessionEmpty.classList.add("hidden");
  for (const session of sessions) {
    const item = document.createElement("li");
    item.className = "session-item";
    item.dataset.id = session.id;

    const main = document.createElement("button");
    main.type = "button";
    main.className = "session-open";
    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = session.title || "New conversation";
    const meta = document.createElement("span");
    meta.className = "session-meta";
    meta.textContent = formatRelativeTime(session.updated_at);
    main.append(title, meta);
    main.addEventListener("click", () => selectSession(session.id));

    const rename = document.createElement("button");
    rename.type = "button";
    rename.className = "session-action";
    rename.title = "Rename";
    rename.setAttribute("aria-label", "Rename chat");
    rename.innerHTML = '<span aria-hidden="true">&#x270E;</span>';
    rename.addEventListener("click", (e) => {
      e.stopPropagation();
      renameSession(session.id, session.title);
    });

    const del = document.createElement("button");
    del.type = "button";
    del.className = "session-action session-delete";
    del.title = "Delete";
    del.setAttribute("aria-label", "Delete chat");
    del.innerHTML = '<span aria-hidden="true">×</span>';
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteSession(session.id, session.title);
    });

    item.append(main, rename, del);
    sessionList.appendChild(item);
  }
  markActiveSession();
}

/** Fetch the session list from the backend and render it. */
async function loadSessions() {
  try {
    const response = await fetch("/api/chat/sessions", { cache: "no-store" });
    const data = await response.json();
    if (data.active_id) currentSessionId = data.active_id;
    renderSessionList(data.sessions || []);
  } catch (_err) {
    /* sidebar stays as-is */
  }
}

/** Set the conversation-bar title text. */
function setTitle(title) {
  if (chatTitle) chatTitle.textContent = title || "New conversation";
}

/**
 * Resume a saved session: load its history into the log and make it active.
 * @param {string} id - The session id to open.
 */
async function selectSession(id) {
  if (chatBusy) return;
  try {
    const response = await fetch(`/api/chat/sessions/${id}/resume`, { method: "POST" });
    if (!response.ok) throw new Error(`resume failed: ${response.status}`);
    const record = await response.json();
    currentSessionId = id;
    setTitle(record.title);
    renderHistory(record.messages);
    setSessionUsage(record.usage);
    markActiveSession();
    closeSidebar();
  } catch (_err) {
    /* ignore — leave the current view untouched */
  }
}

/**
 * Rename a saved session via a prompt.
 * @param {string} id - The session id.
 * @param {string} current - The current title (prompt default).
 */
async function renameSession(id, current) {
  const next = window.prompt("Rename chat", current || "");
  if (next === null) return;
  const title = next.trim();
  if (!title) return;
  try {
    await fetch(`/api/chat/sessions/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    if (id === currentSessionId) setTitle(title);
    loadSessions();
  } catch (_err) {
    /* ignore */
  }
}

/**
 * Delete a saved session after confirmation.
 * @param {string} id - The session id.
 * @param {string} title - The title shown in the confirm dialog.
 */
async function deleteSession(id, title) {
  if (!window.confirm(`Delete “${title || "this chat"}”? This can't be undone.`)) return;
  try {
    await fetch(`/api/chat/sessions/${id}`, { method: "DELETE" });
    if (id === currentSessionId) {
      currentSessionId = null;
      clearLog();
      setTitle("New conversation");
    }
    loadSessions();
  } catch (_err) {
    /* ignore */
  }
}

/** Start a fresh conversation, leaving any saved sessions on disk. */
async function newConversation() {
  if (chatBusy) return;
  try {
    await fetch("/api/chat/reset", { method: "POST" });
  } catch (_err) {
    /* ignore */
  }
  currentSessionId = null;
  clearLog();
  setTitle("New conversation");
  setSessionUsage(null);
  markActiveSession();
  closeSidebar();
  chatInput.focus();
}

// --- Sidebar drawer (narrow screens) ----------------------------------------

/** Open the sidebar drawer overlay. */
function openSidebar() {
  chatSidebar.classList.add("open");
  chatSidebarScrim.hidden = false;
  if (chatSidebarToggle) chatSidebarToggle.setAttribute("aria-expanded", "true");
}

/** Close the sidebar drawer overlay. */
function closeSidebar() {
  chatSidebar.classList.remove("open");
  chatSidebarScrim.hidden = true;
  if (chatSidebarToggle) chatSidebarToggle.setAttribute("aria-expanded", "false");
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
  if (note) {
    chatNote.textContent = note;
    chatNote.classList.remove("hidden");
  } else {
    chatNote.classList.add("hidden");
  }
}

/**
 * Toggle the composer button between "send" (idle) and "stop" (streaming),
 * mirroring claude.ai's single-button morph.
 * @param {boolean} streaming - Whether a response is currently streaming.
 */
function setStreaming(streaming) {
  chatSend.classList.toggle("hidden", streaming);
  chatStop.classList.toggle("hidden", !streaming);
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
  setStreaming(true);
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
      body: JSON.stringify({ message, model: chatModel ? chatModel.value : undefined }),
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
        } else if (event.type === "usage") {
          addUsageLine(event);
        } else if (event.type === "session_saved") {
          currentSessionId = event.session.id;
          setTitle(event.session.title);
          setSessionUsage(event.session.usage);
          loadSessions();
        } else if (event.type === "notice") {
          addBubble("notice", event.text || "");
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
    setStreaming(false);
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
  updateCount();
  submitMessage(message);
});

chatInput.addEventListener("input", () => {
  autoGrow();
  updateCount();
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

chatStop.addEventListener("click", () => {
  if (activeController) activeController.abort();
});

chatNew.addEventListener("click", newConversation);

if (chatSidebarToggle) {
  chatSidebarToggle.addEventListener("click", () => {
    if (chatSidebar.classList.contains("open")) closeSidebar();
    else openSidebar();
  });
}
if (chatSidebarScrim) chatSidebarScrim.addEventListener("click", closeSidebar);

if (chatEmpty) {
  chatEmpty.querySelectorAll(".chat-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      chatInput.value = chip.textContent;
      autoGrow();
      updateCount();
      chatInput.focus();
    });
  });
}

// Remember the model choice across reloads. Guard against a stored value that
// no longer matches an available option.
if (chatModel) {
  const MODEL_KEY = "flipdot.chat.model";
  const stored = localStorage.getItem(MODEL_KEY);
  if (stored && [...chatModel.options].some((opt) => opt.value === stored)) {
    chatModel.value = stored;
  }
  chatModel.addEventListener("change", () => {
    localStorage.setItem(MODEL_KEY, chatModel.value);
  });
}

updateEmptyState();
updateCount();
refreshStatus();
loadSessions();
