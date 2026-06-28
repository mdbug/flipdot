import { test, expect } from "@playwright/test";

const SCRIPTS_RESPONSE = {
  scripts: ["wave.py", "bounce.py"],
  active: "wave.py",
};

const CODE_RESPONSE = { code: "import time\nprint('hello')" };

function mockScriptsApi(page, { scripts = SCRIPTS_RESPONSE } = {}) {
  page.route("/api/scripts", (route) =>
    route.fulfill({ json: scripts }),
  );
  page.route("/api/scripts/*/code", (route) =>
    route.fulfill({ json: CODE_RESPONSE }),
  );
  page.route("/api/scripts/*/play", (route) =>
    route.fulfill({ json: { ok: true } }),
  );
  page.route("/api/scripts/*", (route) => {
    if (route.request().method() === "DELETE") {
      route.fulfill({ json: { ok: true } });
    } else {
      route.continue();
    }
  });
}

test("shows empty state when no scripts exist", async ({ page }) => {
  page.route("/api/scripts", (route) =>
    route.fulfill({ json: { scripts: [], active: "" } }),
  );
  await page.goto("/scripts");

  await expect(page.locator("#scriptListEmpty")).toBeVisible();
  await expect(page.locator("#scriptList li")).toHaveCount(0);
});

test("lists scripts returned by the API", async ({ page }) => {
  mockScriptsApi(page);
  await page.goto("/scripts");

  const items = page.locator("#scriptList .script-name");
  await expect(items).toHaveCount(2);
  await expect(items.nth(0)).toHaveText("wave.py");
  await expect(items.nth(1)).toHaveText("bounce.py");
});

test("shows active badge on the currently playing script", async ({ page }) => {
  mockScriptsApi(page);
  await page.goto("/scripts");

  const waveBadge = page
    .locator("#scriptList .script-item")
    .filter({ hasText: "wave.py" })
    .locator(".script-active-badge");
  await expect(waveBadge).toBeVisible();
  await expect(waveBadge).toHaveText("playing");

  const bounceBadge = page
    .locator("#scriptList .script-item")
    .filter({ hasText: "bounce.py" })
    .locator(".script-active-badge");
  await expect(bounceBadge).not.toBeVisible();
});

test("shows error message when the API request fails", async ({ page }) => {
  page.route("/api/scripts", (route) =>
    route.fulfill({ status: 500, body: "Internal Server Error" }),
  );
  await page.goto("/scripts");

  await expect(page.locator("#scriptListError")).toBeVisible();
  await expect(page.locator("#scriptListError")).toContainText("Failed to load");
});

test("clicking a script shows detail panel and loads its code", async ({ page }) => {
  mockScriptsApi(page);
  await page.goto("/scripts");

  await page.locator("#scriptList .script-item").filter({ hasText: "bounce.py" }).click();

  await expect(page.locator("#scriptsDetail")).toBeVisible();
  await expect(page.locator("#scriptsPlaceholder")).not.toBeVisible();
  await expect(page.locator("#scriptsDetailName")).toHaveText("bounce.py");
  await expect(page.locator("#scriptsCode code")).toContainText("print('hello')");
});

test("detail status reflects playing vs idle", async ({ page }) => {
  mockScriptsApi(page);
  await page.goto("/scripts");

  // Select the currently active script
  await page.locator("#scriptList .script-item").filter({ hasText: "wave.py" }).click();
  await expect(page.locator("#scriptsDetailStatus")).toHaveText("playing");

  // Select an idle script
  await page.locator("#scriptList .script-item").filter({ hasText: "bounce.py" }).click();
  await expect(page.locator("#scriptsDetailStatus")).toHaveText("idle");
});

test("play button POSTs to the correct endpoint and shows status", async ({ page }) => {
  mockScriptsApi(page);
  await page.goto("/scripts");
  await page.locator("#scriptList .script-item").filter({ hasText: "bounce.py" }).click();

  const [request] = await Promise.all([
    page.waitForRequest((req) => req.url().includes("/api/scripts/bounce.py/play")),
    page.locator("#scriptsPlay").click(),
  ]);

  expect(request.method()).toBe("POST");
  await expect(page.locator("#scriptsActionStatus")).toHaveText("Playing.");
  await expect(page.locator("#scriptsActionStatus")).toBeVisible();
});

test("delete button removes the script after confirmation", async ({ page }) => {
  let deleteRequested = false;
  page.route("/api/scripts", (route) => {
    if (deleteRequested) {
      route.fulfill({ json: { scripts: ["wave.py"], active: "wave.py" } });
    } else {
      route.fulfill({ json: SCRIPTS_RESPONSE });
    }
  });
  page.route("/api/scripts/*/code", (route) => route.fulfill({ json: CODE_RESPONSE }));
  page.route("/api/scripts/bounce.py", (route) => {
    deleteRequested = true;
    route.fulfill({ json: { ok: true } });
  });

  await page.goto("/scripts");
  await page.locator("#scriptList .script-item").filter({ hasText: "bounce.py" }).click();

  page.on("dialog", (dialog) => dialog.accept());
  await page.locator("#scriptsDelete").click();

  // Detail panel hides and list reloads without bounce.py
  await expect(page.locator("#scriptsDetail")).not.toBeVisible();
  await expect(page.locator("#scriptsPlaceholder")).toBeVisible();
  await expect(page.locator("#scriptList .script-name")).toHaveCount(1);
  await expect(page.locator("#scriptList .script-name")).toHaveText("wave.py");
});
