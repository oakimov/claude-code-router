#!/usr/bin/env node
/**
 * inject-statsig.mjs
 *
 * Patch the ChatGPT/Codex desktop renderer's Statsig client so the model
 * picker shows every catalog model instead of only the server-delivered
 * allowlist (Statsig dynamic config 107580212 -> use_hidden_models).
 *
 * The renderer reads the picker setting via
 *   u7r(n.getDynamicConfig('107580212').value)
 * and shows a model only when `!model.hidden` OR (when use_hidden_models)
 * its slug is in the allowlist. We patch the client's getDynamicConfig to
 * return use_hidden_models=false, then force a re-evaluation.
 *
 * Usage:
 *   node inject-statsig.mjs [--port 9222] [--config <statsig-id>] [--reload]
 *
 *   default            : patch the already-loaded page in place (no reload)
 *   --reload           : install on next navigation + reload the page
 *   --config <id>      : override the Statsig dynamic-config id (default 107580212)
 *
 * Requires the app to be running with --remote-debugging-port=<port> and
 * Node >= 22 (global fetch + WebSocket).
 */

function parseArgs(argv) {
  const options = {
    port: Number(process.env.CDP_PORT || 9222),
    reload: false,
    configId: "107580212",
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--reload") {
      options.reload = true;
      continue;
    }
    if (arg === "--port" || arg === "--config") {
      const value = argv[++i];
      if (!value || value.startsWith("--")) {
        throw new Error(`${arg} requires a value`);
      }
      if (arg === "--port") options.port = Number(value);
      else options.configId = value;
      continue;
    }
    throw new Error(`Unknown option: ${arg}`);
  }
  if (!Number.isInteger(options.port) || options.port < 1 || options.port > 65535) {
    throw new Error(`Invalid CDP port: ${options.port}`);
  }
  return options;
}

const OPTIONS = parseArgs(process.argv.slice(2));
const PORT = OPTIONS.port;
const RELOAD = OPTIONS.reload;
const CONFIG_ID = OPTIONS.configId;

const SOURCE = `
(() => {
  const CONFIG_ID = __CONFIG_ID__;
  const patched = new WeakSet();

  const patchInstance = (inst) => {
    if (!inst || typeof inst.getDynamicConfig !== 'function' || patched.has(inst)) return false;
    patched.add(inst);
    try {
      const orig = inst.getDynamicConfig.bind(inst);
      inst.getDynamicConfig = function (name) {
        const dc = orig(name);
        if (String(name) === CONFIG_ID && dc && dc.value) {
          // Mutate in place (covers the shared-reference case)...
          try { Object.assign(dc.value, { use_hidden_models: false, available_models: [] }); } catch (e) {}
          // ...and shadow the getter with a fresh object (covers a per-call copy).
          try {
            const v = dc.value;
            Object.defineProperty(dc, 'value', {
              configurable: true,
              enumerable: true,
              get: () => ({ ...v, use_hidden_models: false, available_models: [] }),
            });
          } catch (e) {}
        }
        return dc;
      };
      // Force a values_updated so the picker store re-reads the patched config.
      try { inst.updateUserAsync(inst.getContext().user); } catch (e) {}
      return true;
    } catch (e) {
      return false;
    }
  };

  // Keep polling: the app creates two clients (pre-login async + post-login
  // bootstrap), and the post-login one appears only after login.
  setInterval(() => {
    try {
      const s = window.__STATSIG__;
      if (!s) return;
      const all = [];
      if (s.instances) for (const k of Object.keys(s.instances)) all.push(s.instances[k]);
      if (s.firstInstance) all.push(s.firstInstance);
      for (const inst of all) patchInstance(inst);
    } catch (e) {}
  }, 300);
})();
`.replace("__CONFIG_ID__", JSON.stringify(CONFIG_ID));

async function getTargets() {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 5000);
  try {
    const res = await fetch(`http://127.0.0.1:${PORT}/json/list`, { signal: ctl.signal });
    if (!res.ok) throw new Error(`CDP list failed: HTTP ${res.status}`);
    const body = await res.json();
    return Array.isArray(body) ? body : [];
  } finally {
    clearTimeout(timer);
  }
}

function pickRenderers(targets) {
  const pages = targets.filter((t) => t.type === "page");
  if (pages.length) return pages;
  // Fallback: any target with a debugger URL that isn't DevTools/browser-only.
  return targets.filter(
    (t) => t.webSocketDebuggerUrl && !/^(devtools|chrome|chrome-extension|about:)/.test(String(t.url || ""))
  );
}

function connect(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    const timer = setTimeout(() => {
      ws.close();
      reject(new Error("CDP websocket connect timed out"));
    }, 8000);
    ws.addEventListener("open", () => {
      clearTimeout(timer);
      resolve(ws);
    });
    ws.addEventListener("error", () => {
      clearTimeout(timer);
      reject(new Error("CDP websocket connect failed (add --remote-allow-origins=* to the launch args if refused)"));
    });
  });
}

function rpc(ws, id, method, params, timeoutMs = 10_000) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      ws.removeEventListener("message", onMessage);
      ws.removeEventListener("close", onClose);
      ws.removeEventListener("error", onError);
    };
    const fail = (error) => {
      cleanup();
      reject(error);
    };
    const onMessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.id === id) {
        cleanup();
        if (msg.error) reject(new Error(`${method}: ${JSON.stringify(msg.error)}`));
        else resolve(msg.result);
      }
    };
    const onClose = () => fail(new Error(`${method}: CDP websocket closed`));
    const onError = () => fail(new Error(`${method}: CDP websocket error`));
    const timer = setTimeout(
      () => fail(new Error(`${method}: CDP command timed out`)),
      timeoutMs
    );
    ws.addEventListener("message", onMessage);
    ws.addEventListener("close", onClose);
    ws.addEventListener("error", onError);
    ws.send(JSON.stringify({ id, method, params: params || {} }));
  });
}

async function patchTarget(target) {
  const ws = await connect(target.webSocketDebuggerUrl);
  try {
    if (RELOAD) {
      await rpc(ws, 1, "Page.addScriptToEvaluateOnNewDocument", { source: SOURCE });
      await rpc(ws, 2, "Page.reload", { ignoreCache: true });
    } else {
      await rpc(ws, 1, "Runtime.evaluate", { expression: SOURCE, awaitPromise: true });
    }
  } finally {
    ws.close();
  }
  return target.url;
}

async function waitForCdp() {
  let last = [];
  for (let i = 0; i < 45; i++) {
    try {
      last = await getTargets();
    } catch {
      // endpoint not up yet
    }
    const pages = pickRenderers(last);
    if (pages.length) return pages;
    if (i % 5 === 4 && last.length) {
      console.log(
        `[statsig-unlock] waiting for renderer target (seen: ${last
          .map((t) => t.type || "?")
          .join(", ")})...`
      );
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  const kinds = {};
  for (const t of last) kinds[t.type || "?"] = (kinds[t.type || "?"] || 0) + 1;
  const dump = last.map((t) => ({ type: t.type, url: t.url, title: t.title })).slice(0, 25);
  throw new Error(
    `No renderer targets on 127.0.0.1:${PORT} after ~45s. Target types: ${JSON.stringify(kinds)}. ` +
      `Targets: ${JSON.stringify(dump, null, 2)}. ` +
      `Confirm the app window is open and --remote-debugging-port=${PORT} was used.`
  );
}

async function main() {
  const pages = await waitForCdp();
  for (const t of pages) {
    const url = await patchTarget(t);
    console.log(`[statsig-unlock] ${RELOAD ? "installed + reloaded" : "patched live"}  ${url}`);
  }
  console.log(`[statsig-unlock] Done. If models still don't show, fully quit and re-run launch.sh (a stale picker store may need a fresh window).`);
}

main().catch((err) => {
  console.error(`[statsig-unlock] ${err.message}`);
  process.exit(1);
});
