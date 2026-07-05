import { test, expect } from "@playwright/test";

const SESSIONS_RESPONSE = {
  sessions: [
    { id: "abc123", title: "Draw a circle", preview: "Draw a circle", created_at: 0 },
    { id: "def456", title: "Write HELLO", preview: "Write HELLO", created_at: 1 },
  ],
  active_id: null,
};

function mockSessions(page, data = SESSIONS_RESPONSE) {
  page.route("/api/chat/sessions", (route) => route.fulfill({ json: data }));
}

function mockChatStatus(page) {
  page.route("/api/chat/status", (route) =>
    route.fulfill({ json: { available: true, busy: false, session_id: null } })
  );
}

test("shows empty-state when no sessions exist", async ({ page }) => {
  mockSessions(page, { sessions: [], active_id: null });
  mockChatStatus(page);
  await page.goto("/chat");

  await expect(page.locator("#sessionEmpty")).toBeVisible();
  await expect(page.locator("#sessionList li")).toHaveCount(0);
});

test("populates session list from API", async ({ page }) => {
  mockSessions(page);
  mockChatStatus(page);
  await page.goto("/chat");

  const items = page.locator("#sessionList li");
  await expect(items).toHaveCount(2);
  await expect(items.nth(0)).toContainText("Draw a circle");
  await expect(items.nth(1)).toContainText("Write HELLO");
  await expect(page.locator("#sessionEmpty")).not.toBeVisible();
});

test("character counter updates as user types", async ({ page }) => {
  mockSessions(page, { sessions: [], active_id: null });
  mockChatStatus(page);
  await page.goto("/chat");

  await expect(page.locator("#chatCount")).toHaveText("0 / 2000");
  await page.locator("#chatInput").fill("Hello world");
  await expect(page.locator("#chatCount")).toHaveText("11 / 2000");
});

test("clicking a chip fills the textarea with its text", async ({ page }) => {
  mockSessions(page, { sessions: [], active_id: null });
  mockChatStatus(page);
  await page.goto("/chat");

  const chip = page.locator(".chat-chip").first();
  const chipText = await chip.textContent();
  await chip.click();

  await expect(page.locator("#chatInput")).toHaveValue(chipText.trim());
});

test("submitting the form appends a user bubble and sends to /api/chat", async ({ page }) => {
  mockSessions(page, { sessions: [], active_id: null });
  mockChatStatus(page);

  const streamLine = JSON.stringify({ type: "text", text: "Done!" });
  page.route("/api/chat", (route) =>
    route.fulfill({
      status: 200,
      headers: { "Content-Type": "text/plain" },
      body: streamLine + "\n",
    })
  );

  await page.goto("/chat");
  await page.locator("#chatInput").fill("Draw a star");

  const [request] = await Promise.all([
    page.waitForRequest("/api/chat"),
    page.locator("#chatSend").click(),
  ]);

  expect(request.method()).toBe("POST");
  const body = JSON.parse(request.postData());
  expect(body.message).toBe("Draw a star");

  await expect(page.locator(".chat-bubble.chat-user")).toHaveText("Draw a star");
});

test("session item click POSTs to resume endpoint and updates title", async ({ page }) => {
  mockSessions(page);
  mockChatStatus(page);
  page.route("/api/chat/sessions/abc123/resume", (route) =>
    route.fulfill({
      json: { id: "abc123", title: "Draw a circle", messages: [] },
    })
  );

  await page.goto("/chat");

  const [request] = await Promise.all([
    page.waitForRequest("/api/chat/sessions/abc123/resume"),
    page.locator("#sessionList .session-open").first().click(),
  ]);

  expect(request.method()).toBe("POST");
  await expect(page.locator("#chatTitle")).toHaveText("Draw a circle");
});

const MODELS_RESPONSE = {
  default: "claude-opus-4-8",
  locked: null,
  models: [
    { id: "claude-opus-4-8", label: "Claude Opus 4.8", provider: "anthropic", available: true },
    { id: "claude-haiku-4-5", label: "Claude Haiku 4.5", provider: "anthropic", available: true },
    { id: "gpt-5.4-mini", label: "GPT-5.4 mini", provider: "openai", available: false },
    { id: "z-ai/glm-5.2", label: "GLM-5.2", provider: "openrouter", available: true },
  ],
};

function mockChatModels(page, data = MODELS_RESPONSE) {
  page.route("/api/chat/models", (route) => route.fulfill({ json: data }));
}

test("model selector populates grouped by provider, disabling keyless models", async ({ page }) => {
  mockSessions(page, { sessions: [], active_id: null });
  mockChatStatus(page);
  mockChatModels(page);
  await page.goto("/chat");

  const select = page.locator("#chatModel");
  await expect(select.locator("optgroup")).toHaveCount(3);
  await expect(select.locator('optgroup[label="Anthropic"] option')).toHaveCount(2);
  const openaiOption = select.locator('option[value="gpt-5.4-mini"]');
  await expect(openaiOption).toBeDisabled();
  await expect(openaiOption).toHaveText("GPT-5.4 mini (no API key)");
  await expect(select.locator('option[value="z-ai/glm-5.2"]')).toBeEnabled();
});

test("model selector locks after the first message and unlocks on new chat", async ({ page }) => {
  mockSessions(page, { sessions: [], active_id: null });
  mockChatStatus(page);
  mockChatModels(page);
  page.route("/api/chat", (route) =>
    route.fulfill({
      status: 200,
      headers: { "Content-Type": "text/plain" },
      body: JSON.stringify({ type: "text", text: "ok" }) + "\n",
    })
  );
  page.route("/api/chat/reset", (route) => route.fulfill({ json: { status: "ok" } }));

  await page.goto("/chat");
  const select = page.locator("#chatModel");
  await expect(select).toBeEnabled();

  await page.locator("#chatInput").fill("hello");
  await page.locator("#chatSend").click();
  await expect(select).toBeDisabled();

  await page.locator("#chatNew").click();
  await expect(select).toBeEnabled();
});

test("resuming a session locks the selector to the session's model", async ({ page }) => {
  mockSessions(page);
  mockChatStatus(page);
  mockChatModels(page);
  page.route("/api/chat/sessions/abc123/resume", (route) =>
    route.fulfill({
      json: { id: "abc123", title: "Draw a circle", messages: [], model: "claude-haiku-4-5" },
    })
  );

  await page.goto("/chat");
  await page.locator("#sessionList .session-open").first().click();

  const select = page.locator("#chatModel");
  await expect(select).toBeDisabled();
  await expect(select).toHaveValue("claude-haiku-4-5");
});
