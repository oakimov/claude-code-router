---
sidebar_position: 8
---

# ccr chrome-bridge

Start the Chrome On-Device model bridge for Gemini Nano.

```bash
ccr chrome-bridge
```

## Description

Starts a bridge process that connects to Chrome's Prompt API via the Chrome DevTools Protocol (CDP), allowing the router to use Chrome's built-in Gemini Nano model (~4GB local LLM).

**Must be run on the host machine** (not inside Docker), as it needs direct access to Chrome's debugging port.

## How It Works

1. The bridge connects to Chrome's remote debugging port (`--remote-debugging-port=9229`)
2. It communicates with the Prompt API to send prompts and receive responses
3. The router's `chrome-on-device` transformer translates between OpenAI Chat Completions format and the Prompt API
4. Responses are streamed back through the router as standard SSE

## Features

- **Zero API cost** — Fully local inference on your device
- **Streaming support** — Full SSE streaming
- **Structured output** — Uses `responseConstraint` for reliable JSON tool calls
- **Automatic stall recovery** — Retries with higher temperature if the model stalls

## Prerequisites

- Google Chrome (Canary or Dev) with Gemini Nano enabled
- Node.js installed on the host

For detailed setup instructions, see the [Chrome On-Device Integration Guide](/docs/server/guides/chrome-on-device).
