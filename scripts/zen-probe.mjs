#!/usr/bin/env node
/**
 * Minimal Zen TTFT probe. Works under Node and Bun.
 *
 * Env:
 *   OPENCODE_API_KEY   required
 *   ZEN_BASE           default https://opencode.ai/zen/v1
 *   ZEN_MODEL          default muse-spark-1.2-contributor-free
 *   ZEN_MODE           models | responses  (default responses)
 *   ZEN_HTTP_PROTOCOL  default | http1.1 | http2 | http3
 *   ZEN_RUNTIME_LABEL  optional label printed in JSON
 */
import https from "node:https";
import http2 from "node:http2";
import { URL } from "node:url";

const key = process.env.OPENCODE_API_KEY || process.env.OPENCODE_ZEN_API_KEY;
if (!key) {
  console.error(JSON.stringify({ ok: false, error: "OPENCODE_API_KEY unset" }));
  process.exit(2);
}

const base = (process.env.ZEN_BASE || "https://opencode.ai/zen/v1").replace(/\/$/, "");
const model = process.env.ZEN_MODEL || "muse-spark-1.2-contributor-free";
const mode = process.env.ZEN_MODE || "responses";
const label = process.env.ZEN_RUNTIME_LABEL || "unknown";
const protocolReq = (process.env.ZEN_HTTP_PROTOCOL || "default").toLowerCase();
const isBun = typeof Bun !== "undefined";
const runtime = isBun ? `bun/${Bun.version}` : `node/${process.version}`;

const OPENCODE_VERSION = process.env.OPENCODE_VERSION || "1.18.25";
const sessionId =
  process.env.ZEN_SESSION_ID ||
  `ses_${Date.now().toString(16).padStart(12, "0").slice(-12)}probeSESSION00`;
const requestId =
  process.env.ZEN_REQUEST_ID ||
  `msg_${Date.now().toString(16).padStart(12, "0").slice(-12)}probeREQUEST00`;
// Match OpenCode CLI → Zen (request.ts + CCR opencode-headers). No Authorization.
const commonHeaders = {
  "content-type": "application/json",
  "x-api-key": key,
  "x-opencode-project": "global",
  "x-opencode-session": sessionId,
  "x-opencode-request": requestId,
  "x-opencode-client": "cli",
  "user-agent": `opencode/${OPENCODE_VERSION}`,
};

function ms(t0) {
  return Math.round((performance.now() - t0) * 1000) / 1000;
}

function normalizeProtocol(p) {
  if (!p || p === "default" || p === "auto") return "default";
  if (p === "h1" || p === "http1" || p === "http1.1") return "http1.1";
  if (p === "h2" || p === "http2") return "http2";
  if (p === "h3" || p === "http3") return "http3";
  throw new Error(`unsupported ZEN_HTTP_PROTOCOL=${p}`);
}

const protocol = normalizeProtocol(protocolReq);

/** Bun: pin via fetch protocol option (http2/http3 experimental). */
async function bunFetch(url, init) {
  const opts = { ...init };
  if (protocol === "http1.1") opts.protocol = "http1.1";
  else if (protocol === "http2") opts.protocol = "http2";
  else if (protocol === "http3") opts.protocol = "http3";
  // default: leave unset (Bun HTTP/1.1 path)
  return fetch(url, opts);
}

/**
 * Node: force protocol with https (1.1) or http2. HTTP/3 is not available in
 * Node's stock fetch/undici here — report unsupported.
 */
function nodeRequest(urlStr, { method, headers, body }) {
  if (protocol === "http3") {
    return Promise.reject(
      new Error("node does not support HTTP/3 client in this probe (use bun)")
    );
  }
  const url = new URL(urlStr);
  const t0 = performance.now();

  if (protocol === "http2") {
    return new Promise((resolve, reject) => {
      const client = http2.connect(url.origin, {
        // ALPN negotiates h2
        servername: url.hostname,
      });
      client.on("error", (err) => {
        client.close();
        reject(err);
      });
      const path = url.pathname + url.search;
      const req = client.request({
        ":method": method || "GET",
        ":path": path,
        ...headers,
      });
      const chunks = [];
      let status = 0;
      let hdrs = {};
      let headersMs = null;
      req.on("response", (h) => {
        headersMs = ms(t0);
        status = h[":status"] || 0;
        hdrs = h;
      });
      req.on("data", (c) => chunks.push(c));
      req.on("end", () => {
        const buf = Buffer.concat(chunks);
        client.close();
        resolve({
          status,
          headersMs,
          headers: {
            get(name) {
              const key = name.toLowerCase();
              const v = hdrs[key] ?? hdrs[name];
              return Array.isArray(v) ? v[0] : v ?? null;
            },
          },
          httpVersion: "2",
          async text() {
            return buf.toString("utf8");
          },
          async arrayBuffer() {
            return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          },
          body: null,
          _raw: buf,
          ok: status >= 200 && status < 300,
        });
      });
      req.on("error", (err) => {
        client.close();
        reject(err);
      });
      if (body) req.write(body);
      req.end();
    });
  }

  // http1.1 or default → HTTPS/1.1
  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        protocol: "https:",
        hostname: url.hostname,
        port: url.port || 443,
        path: url.pathname + url.search,
        method: method || "GET",
        headers,
        // Force HTTP/1.1 (Node https defaults to 1.1 anyway)
        ALPNProtocols: ["http/1.1"],
      },
      (res) => {
        const headersMs = ms(t0);
        const chunks = [];
        res.on("data", (c) => chunks.push(c));
        res.on("end", () => {
          const buf = Buffer.concat(chunks);
          resolve({
            status: res.statusCode || 0,
            headersMs,
            headers: {
              get(name) {
                const v = res.headers[name.toLowerCase()];
                return Array.isArray(v) ? v[0] : v ?? null;
              },
            },
            httpVersion: res.httpVersion,
            async text() {
              return buf.toString("utf8");
            },
            body: null,
            _raw: buf,
            ok: (res.statusCode || 0) >= 200 && (res.statusCode || 0) < 300,
          });
        });
      }
    );
    req.on("error", reject);
    if (body) req.write(body);
    req.end();
  });
}

async function doFetch(url, init) {
  if (isBun) {
    const res = await bunFetch(url, init);
    return {
      status: res.status,
      ok: res.ok,
      headers: res.headers,
      headersMs: null, // filled by caller around await
      httpVersion: protocol === "default" ? "bun-default(~1.1)" : protocol,
      text: () => res.text(),
      body: res.body,
      _fetchResponse: res,
    };
  }
  return nodeRequest(url, init);
}

async function readSseMetrics(res, t0, headersMs) {
  // Node forced paths buffer the whole body (http2/https). Parse SSE offline.
  if (res._raw) {
    const text = res._raw.toString("utf8");
    let firstEventMs = null;
    let firstEventType = null;
    let firstDeltaMs = null;
    let firstDeltaType = null;
    for (const line of text.split(/\r?\n/)) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      let ev;
      try {
        ev = JSON.parse(data);
      } catch {
        continue;
      }
      // Buffered responses: timestamps collapse to end-of-body; still report types.
      const now = ms(t0);
      if (firstEventMs == null) {
        firstEventMs = headersMs ?? now;
        firstEventType = ev.type || null;
      }
      const isDelta =
        typeof ev.type === "string" &&
        (ev.type.endsWith(".delta") ||
          ev.type === "response.output_text.delta" ||
          ev.type === "response.reasoning_summary_text.delta");
      if (isDelta && firstDeltaMs == null) {
        firstDeltaMs = headersMs ?? now;
        firstDeltaType = ev.type;
      }
    }
    return {
      msFirstEvent: firstEventMs,
      firstEventType,
      msFirstDelta: firstDeltaMs,
      firstDeltaType,
      msComplete: ms(t0),
      note: "node-forced-protocol buffers body; firstEvent/Delta times ≈ headers",
    };
  }

  // Streaming path (Bun fetch / Node default fetch)
  if (!res.body) {
    return { msComplete: ms(t0) };
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let firstEventMs = null;
  let firstDeltaMs = null;
  let firstEventType = null;
  let firstDeltaType = null;
  let done = false;
  while (!done) {
    const { value, done: streamDone } = await reader.read();
    if (streamDone) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n");
    buf = parts.pop() || "";
    for (const line of parts) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (!data || data === "[DONE]") {
        done = true;
        break;
      }
      let ev;
      try {
        ev = JSON.parse(data);
      } catch {
        continue;
      }
      const now = ms(t0);
      if (firstEventMs == null) {
        firstEventMs = now;
        firstEventType = ev.type || null;
      }
      const isDelta =
        typeof ev.type === "string" &&
        (ev.type.endsWith(".delta") ||
          ev.type === "response.output_text.delta" ||
          ev.type === "response.reasoning_summary_text.delta");
      if (isDelta && firstDeltaMs == null) {
        firstDeltaMs = now;
        firstDeltaType = ev.type;
      }
      if (ev.type === "response.completed" || ev.type === "response.failed") {
        done = true;
        break;
      }
    }
  }
  return {
    msFirstEvent: firstEventMs,
    firstEventType,
    msFirstDelta: firstDeltaMs,
    firstDeltaType,
    msComplete: ms(t0),
  };
}

async function probeModels() {
  const t0 = performance.now();
  const res = await doFetch(`${base}/models`, {
    method: "GET",
    headers: commonHeaders,
  });
  const tHeaders = res.headersMs ?? ms(t0);
  const text = await res.text();
  return {
    ok: res.ok,
    status: res.status,
    mode: "models",
    protocolRequested: protocol,
    httpVersion: res.httpVersion || null,
    msHeaders: tHeaders,
    msBody: ms(t0),
    bytes: text.length,
    altSvc: res.headers.get("alt-svc"),
    cfRay: res.headers.get("cf-ray"),
  };
}

async function probeResponses() {
  const t0 = performance.now();
  const body = JSON.stringify({
    model,
    stream: true,
    store: false,
    input: [
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "Reply with exactly: ok" }],
      },
    ],
    max_output_tokens: 16,
  });

  let res;
  try {
    res = await doFetch(`${base}/responses`, {
      method: "POST",
      headers: commonHeaders,
      body,
    });
  } catch (err) {
    return {
      ok: false,
      status: 0,
      mode: "responses",
      protocolRequested: protocol,
      msHeaders: ms(t0),
      error: String(err?.message || err).slice(0, 300),
    };
  }

  const tHeaders = res.headersMs ?? ms(t0);
  if (!res.ok) {
    const errText = res._raw
      ? res._raw.toString("utf8")
      : await res.text().catch(() => "");
    return {
      ok: false,
      status: res.status,
      mode: "responses",
      protocolRequested: protocol,
      httpVersion: res.httpVersion || null,
      msHeaders: tHeaders,
      error: errText.slice(0, 300),
      altSvc: res.headers.get("alt-svc"),
      cfRay: res.headers.get("cf-ray"),
    };
  }

  const streamMetrics = await readSseMetrics(res, t0, tHeaders);
  return {
    ok: true,
    status: res.status,
    mode: "responses",
    protocolRequested: protocol,
    httpVersion: res.httpVersion || null,
    msHeaders: tHeaders,
    ...streamMetrics,
    altSvc: res.headers.get("alt-svc"),
    cfRay: res.headers.get("cf-ray"),
  };
}

try {
  const result = {
    label,
    runtime,
    base,
    model: mode === "responses" ? model : undefined,
    ...(mode === "models" ? await probeModels() : await probeResponses()),
  };
  console.log(JSON.stringify(result));
  process.exit(result.ok ? 0 : 1);
} catch (err) {
  console.log(
    JSON.stringify({
      ok: false,
      label,
      runtime,
      protocolRequested: protocol,
      error: String(err?.message || err).slice(0, 400),
    })
  );
  process.exit(1);
}
