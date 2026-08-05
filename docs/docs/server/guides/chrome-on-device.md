---
sidebar_position: 3
---

# Chrome On-Device (Gemini Nano)

Claude Code Router can use **Chrome's built-in Gemini Nano** model — a ~4GB local LLM that runs entirely on your device with zero API cost and zero latency to external providers. This is accessed through Chrome's Prompt API via the Chrome DevTools Protocol (CDP).

## How It Works

1. `ccr chrome-bridge` starts a bridge process that connects to Chrome's Prompt API
2. The `chrome-on-device` transformer translates requests between OpenAI Chat Completions format and Chrome's Prompt API format
3. Responses are streamed back through the router as standard SSE
4. The model runs entirely locally in Chrome — no data leaves your machine

## Prerequisites

- **Google Chrome** (Canary or Dev channel) with Gemini Nano enabled
- **Node.js** installed on the **host** (the bridge runs on the host, not inside Docker)
- Claude Code Router running

## Setup

### 1. Enable Gemini Nano in Chrome

1. Install [Chrome Canary](https://www.google.com/chrome/canary/) or Chrome Dev
2. Go to `chrome://flags/#prompt-api-for-gemini-nano` and enable **Prompt API for Gemini Nano**
3. Go to `chrome://flags/#optimization-guide-on-device-model` and enable **Enables optimization guide on device**
4. Restart Chrome
5. Wait for the model to download (check `chrome://components/` for **Optimization Guide On Device Model**)
6. Verify the model is ready: open DevTools console and run:
   ```js
   (await ai.languageModel.capabilities()).available
   ```
   It should return `"readily"`

### 2. Start the Bridge

Run the bridge process on your host machine (not in Docker):

```bash
ccr chrome-bridge
```

The bridge listens on `http://127.0.0.1:9229` by default.

### 3. Configure Provider

Add the Chrome provider to your `~/.claude-code-router/config.json`:

```json
{
  "Providers": [
    {
      "name": "chrome",
      "baseUrl": "http://127.0.0.1:9229",
      "apiKey": "dummy",
      "models": ["gemini-nano"],
      "transformer": {
        "use": ["chrome-on-device"]
      }
    }
  ],
  "Router": {
    "background": "chrome,gemini-nano"
  }
}
```

### 4. Restart

```bash
docker compose restart ccr
```

## Features

- **Zero API cost** — Fully local inference
- **Zero external latency** — No network requests to providers
- **Streaming and non-streaming** — Full SSE streaming support
- **Structured output** — Uses `responseConstraint` for reliable JSON tool calls
- **Automatic stall recovery** — If the model stalls (whitespace-only output), the bridge retries with higher temperature

## Session Management

The bridge maintains persistent `LanguageModel` sessions — conversation history is carried forward naturally within each session, not rebuilt per turn.

- **Multi-session support**: Requests are fingerprinted by a `User-Agent + IP` hash into separate sessions, allowing multiple concurrent Claude Code instances without context contamination. A built-in web dashboard (served on the bridge port) shows real-time stats for all sessions, including turn count, idle time, and context usage.
- **Idle session eviction**: Sessions idle for more than 5 minutes are automatically destroyed to free resources. The `cli` session (dashboard default) is never evicted. Sessions can also be manually evicted via the dashboard's Evict button.
- **Auto-compaction**: Triggers at 85% context usage, resetting the session while preserving the system prompt.
- The bridge connects to Chrome over CDP with a 5-minute protocol timeout to handle slow model inference.

## Limitations

- **Model quality** — Gemini Nano is a small on-device model, best suited for simple tasks and background work
- **Chrome dependency** — Requires Chrome with Gemini Nano enabled; not available in all browsers
- **Node.js on host** — The bridge process must run on the host, not in Docker
- **No external tools** — Restricted to the model's built-in capabilities

## Use Cases

- **Background tasks** — Route `background`-scenario requests to Gemini Nano
- **Simple queries** — Quick answers, text summarization, formatting
- **Offline-capable** — Works without internet access once the model is downloaded

## Troubleshooting

**Bridge won't connect**: Ensure Chrome is running with `--remote-debugging-port=9229` or the bridge can't connect to the Prompt API.

**Model not available**: Check `chrome://components/` for the Optimization Guide model status. If it's not downloaded, wait and restart Chrome.

**Slow responses**: The first request may be slow as the model loads. Subsequent requests should be faster.

## Architecture

```
┌──────────┐     ┌──────────────┐     ┌──────────────────────┐
│  Claude  │────▶│ CCR Server   │────▶│ chrome-on-device     │
│  Code    │     │ (Docker)     │     │ transformer           │
└──────────┘     └──────────────┘     └──────────┬───────────┘
                                                 │
                                                 ▼
                                        ┌──────────────────┐
                                        │ ccr chrome-bridge │
                                        │ (host process)    │
                                        └────────┬─────────┘
                                                 │
                                                 ▼
                                        ┌──────────────────┐
                                        │ Chrome Prompt API│
                                        │ (Gemini Nano)    │
                                        └──────────────────┘
```
