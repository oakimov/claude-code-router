import { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import { existsSync, mkdirSync, writeFileSync, readFileSync, unlinkSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { Buffer } from "buffer";
import { RATE_LIMIT_CONFIG } from "@caeliq/ccr-shared";

const QWEN_AUTH_FILE = join(homedir(), ".claude-code-router", "qwen_auth.json");
const QWEN_TARGET = "https://qwen.aikit.club";
const UPSTREAM_TIMEOUT_MS = 10_000;

interface QwenTokens {
  token: string;
  expiresAt: number | null;
  updatedAt: number;
}

function loadTokens(): QwenTokens | null {
  try {
    if (!existsSync(QWEN_AUTH_FILE)) return null;
    const tokens = JSON.parse(readFileSync(QWEN_AUTH_FILE, "utf-8"));
    if (!tokens.token) return null;
    return tokens as QwenTokens;
  } catch {
    return null;
  }
}

function extractExpFromJwt(token: string): number | null {
  try {
    const payload = JSON.parse(
      Buffer.from(token.split(".")[1], "base64url").toString()
    );
    if (payload.exp) return payload.exp * 1000;
  } catch {
    // Fall through
  }
  return null;
}

async function validateToken(token: string): Promise<string | null> {
  try {
    const res = await fetch(`${QWEN_TARGET}/v1/validate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.access_token || token;
  } catch {
    return null;
  }
}

// Bookmarklet: when clicked on a Qwen page, reads localStorage.getItem('token'),
// copies it to the clipboard (falls back to a prompt if clipboard is blocked),
// and redirects the browser to the CCR /qwen/auth page with the token in the
// URL fragment. The receiving page picks it up and POSTs it as JSON
// automatically. Mirrors qwen-proxy.mjs:126.
//
// The CCR address is baked into the bookmarklet because the JS runs in the
// Qwen page's context (no knowledge of CCR's origin at click time). If you
// run CCR on a non-default host/port, set QWEN_AUTH_REDIRECT env var to the
// full URL prefix (e.g. "http://192.168.1.10:8080").
const QWEN_AUTH_REDIRECT =
  process.env.QWEN_AUTH_REDIRECT || "http://127.0.0.1:3456";
const BOOKMARKLET_CODE =
  `javascript:(function(){` +
  `var t=localStorage.getItem('token');` +
  `if(!t||t.length<40){alert('No Qwen token found. Make sure you are signed in at chat.qwen.ai.');return;}` +
  `var dest='${QWEN_AUTH_REDIRECT}/qwen/auth#/token/'+encodeURIComponent(t);` +
  `navigator.clipboard.writeText(t).then(function(){window.location.href=dest;})` +
  `.catch(function(){` +
  `var p=prompt('Token found. Press Ctrl+C to copy, then click OK to continue.',t);` +
  `if(p)window.location.href=dest;` +
  `});` +
  `})()`;

const HTML_PAGE = (status: "ok" | "warn" | "err", message: string) => `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Qwen Auth</title>
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:640px;margin:40px auto;padding:0 20px;line-height:1.5}
h1{margin:0 0 16px}
.status{padding:12px 16px;border-radius:8px;margin:16px 0}
.ok{background:#d4edda;color:#155724;border:1px solid #c3e6cb}
.warn{background:#fff3cd;color:#856404;border:1px solid #ffeeba}
.err{background:#f8d7da;color:#721c24;border:1px solid #f5c6cb}
pre{background:#f4f4f4;padding:12px;border-radius:6px;overflow-x:auto;font-size:13px;word-break:break-all}
code{background:#f4f4f4;padding:2px 5px;border-radius:3px;font-size:13px}
input[type=text],input[type=password]{width:100%;padding:10px;font-size:14px;border:1px solid #ccc;border-radius:6px;margin:8px 0;font-family:monospace}
button{padding:10px 24px;background:#06f;color:#fff;border:none;border-radius:6px;font-size:14px;cursor:pointer}
button:hover{background:#05e}
a{color:#06f}
.step{background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;padding:16px;margin:12px 0}
.bookmarklet{background:#28a745;color:#fff;padding:10px 24px;border-radius:6px;font-size:14px;cursor:grab;display:inline-block;text-decoration:none;font-weight:bold}
.bookmarklet:hover{background:#218838}
</style>
</head>
<body>
<h1>Qwen Authentication</h1>
<div class="status ${status}">${message}</div>
${status === "ok"
  ? `<p>Your Qwen token is saved and the transformer will refresh it automatically when it nears expiry.</p>
     <p>You can <a href="/qwen/forget">forget the token</a> or submit a new one below.</p>`
  : ""}
<div class="step">
  <strong>Option 1 — Bookmarklet (recommended)</strong>
  <ol>
    <li>Drag this button to your bookmarks bar:
      <p><a class="bookmarklet" href="${BOOKMARKLET_CODE}">Get Qwen Token</a></p>
    </li>
    <li>Open <a href="https://chat.qwen.ai" target="_blank">chat.qwen.ai</a> in another tab and sign in.</li>
    <li>Click the bookmarklet on the Qwen page. Your token will be sent here automatically.</li>
  </ol>
</div>
<div class="step">
  <strong>Option 2 — Manual paste</strong>
  <ol>
    <li>Open <a href="https://chat.qwen.ai" target="_blank">chat.qwen.ai</a> in another tab and sign in.</li>
    <li>Open the browser dev tools (F12) and go to the Console tab.</li>
    <li>Run: <pre>copy(localStorage.getItem('token'))</pre></li>
    <li>Come back here and paste the token into the form below.</li>
  </ol>
</div>
<form id="tokenForm">
  <label for="token"><strong>Qwen JWT</strong></label>
  <input type="password" id="token" name="token" placeholder="eyJ..." required>
  <button type="submit">Save Token</button>
</form>
<script>
// Plain <form> submission sends application/x-www-form-urlencoded, which
// Fastify rejects (no parser registered). Intercept submit and POST as JSON
// instead — that's what the route expects.
document.getElementById('tokenForm').addEventListener('submit', function(e) {
  e.preventDefault();
  var token = document.getElementById('token').value;
  fetch('/qwen/auth', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token: token })
  }).then(function(r) { return r.text(); })
    .then(function(html) { document.open(); document.write(html); document.close(); })
    .catch(function(err) { alert('Save failed: ' + err.message); });
});

// If the page was opened from the bookmarklet, the URL will have
// #/token/<jwt>. POST it as JSON automatically so the user only has to
// click the bookmarklet — no copy/paste dance.
(function(){
  var hash = window.location.hash;
  if (hash && hash.indexOf('#/token/') > -1) {
    var token = decodeURIComponent(hash.split('#/token/')[1]);
    var input = document.getElementById('token');
    if (token && token.length > 40 && input) {
      input.value = token;
      // Replace the form with a "saving…" message so the user sees that
      // something is happening while the upstream validate call runs.
      var form = document.getElementById('tokenForm');
      if (form) form.style.display = 'none';
      var status = document.createElement('div');
      status.className = 'status warn';
      status.textContent = 'Token received. Validating with Qwen and saving...';
      document.body.insertBefore(status, document.body.firstChild);
      fetch('/qwen/auth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: token })
      }).then(function(r) { return r.text(); })
        .then(function(html) { document.open(); document.write(html); document.close(); })
        .catch(function(err) { alert('Save failed: ' + err.message); });
    }
  }
})();
</script>
<hr>
<p><a href="/qwen/status">View token status (JSON)</a></p>
</body>
</html>
`;

function formatExp(expiresAt: number | null): string {
  return expiresAt
    ? new Date(expiresAt).toISOString().slice(0, 19).replace("T", " ")
    : "unknown";
}

export async function registerQwenAuthRoutes(
  app: FastifyInstance
): Promise<void> {
  app.get(
    "/qwen/auth",
    { config: { rateLimit: { ...RATE_LIMIT_CONFIG } } },
    async (req: FastifyRequest, reply: FastifyReply) => {
    const tokens = loadTokens();
    const status = tokens?.token ? "ok" : "warn";
    const message = tokens?.token
      ? `Authenticated. Token expires: ${formatExp(tokens.expiresAt)}.`
      : "Not authenticated. Paste a token below or use the bookmarklet on the Qwen page.";
    reply.type("text/html; charset=utf-8").send(HTML_PAGE(status, message));
  });

  app.post(
    "/qwen/auth",
    { config: { rateLimit: { ...RATE_LIMIT_CONFIG } } },
    async (req: FastifyRequest, reply: FastifyReply) => {
    const body = (req.body || {}) as { token?: string };
    const raw = (body.token || "").trim();

    if (!raw || !raw.includes(".") || raw.length < 40) {
      reply
        .type("text/html; charset=utf-8")
        .status(400)
        .send(
          HTML_PAGE(
            "err",
            "Invalid token. Make sure you copied the full JWT from localStorage."
          )
        );
      return;
    }

    const validated = await validateToken(raw);
    if (!validated) {
      reply
        .type("text/html; charset=utf-8")
        .status(400)
        .send(
          HTML_PAGE(
            "err",
            "Token rejected by Qwen. Make sure you are logged in at chat.qwen.ai and the token is fresh."
          )
        );
      return;
    }

    const expiresAt = extractExpFromJwt(validated);
    const dir = join(homedir(), ".claude-code-router");
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
    writeFileSync(
      QWEN_AUTH_FILE,
      JSON.stringify({ token: validated, expiresAt, updatedAt: Date.now() }, null, 2),
      { mode: 0o600, encoding: "utf-8" }
    );

    app.log.info({ expiresAt, path: QWEN_AUTH_FILE }, "Qwen token saved");
    reply
      .type("text/html; charset=utf-8")
      .send(
        HTML_PAGE(
          "ok",
          `Token saved! Expires: ${formatExp(expiresAt)}. The Qwen transformer will use it for the next request.`
        )
      );
  });

  app.get(
    "/qwen/forget",
    { config: { rateLimit: { ...RATE_LIMIT_CONFIG } } },
    async (req: FastifyRequest, reply: FastifyReply) => {
    try {
      unlinkSync(QWEN_AUTH_FILE);
    } catch {
      // Ignore: file may not exist
    }
    app.log.info("Qwen token cleared");
    reply
      .type("text/html; charset=utf-8")
      .send(HTML_PAGE("warn", "Token cleared. Paste a new token below."));
  });

  app.get(
    "/qwen/status",
    { config: { rateLimit: { ...RATE_LIMIT_CONFIG } } },
    async (req: FastifyRequest, reply: FastifyReply) => {
    const tokens = loadTokens();
    reply.type("application/json").send({
      ok: !!tokens?.token,
      expiresAt: tokens?.expiresAt ?? null,
      path: QWEN_AUTH_FILE,
    });
  });
}
