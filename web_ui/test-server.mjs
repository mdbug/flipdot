import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { join, extname } from "node:path";
import { fileURLToPath } from "node:url";

const DIR = fileURLToPath(new URL(".", import.meta.url));
const PORT = parseInt(process.env.PORT || "3001");

const PAGES = {
  "/": "index.html",
  "/chat": "chat.html",
  "/scripts": "scripts.html",
  "/controller-metrics": "controller_metrics.html",
  "/font-grid": "font_grid.html",
};

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".woff2": "font/woff2",
  ".json": "application/json",
};

createServer((req, res) => {
  const url = req.url.split("?")[0];

  if (PAGES[url]) {
    const content = readFileSync(join(DIR, PAGES[url]));
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(content);
    return;
  }

  if (url.startsWith("/static/")) {
    const filePath = join(DIR, url.slice("/static/".length));
    if (existsSync(filePath)) {
      const mime = MIME[extname(filePath)] ?? "application/octet-stream";
      res.writeHead(200, { "Content-Type": mime });
      res.end(readFileSync(filePath));
      return;
    }
  }

  res.writeHead(404, { "Content-Type": "text/plain" });
  res.end("Not found");
}).listen(PORT, () => {
  console.log(`Test server listening on http://localhost:${PORT}`);
});
