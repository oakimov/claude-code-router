---
title: Fork Changelog — What We've Added
date: 2026-06-01
tags: [changelog, fork, features]
---

# Fork Changelog

This fork of claude-code-router adds several provider integrations, UI improvements, and infrastructure changes. Here's the timeline of what was added and when.

## January 2026 — Mistral Integration

Added the **Mistral AI transformer** with direct API support, including reasoning parameter handling and dedicated Docker Compose configuration. Decoupled Mistral transformation logic into shared utilities.

- `feat: add Mistral transformer for direct API support`
- `fix: correctly handle reasoning parameter for Mistral API`

## April 2026 — Gemini Stability, Code Quality & Build Pipeline

Major improvements to Gemini/Gemma streaming reliability. Fixed Gemini 500 errors caused by malformed `thoughtSignature` placement in request bodies and tool use failures in multi-turn conversations. The project was fully localized (all comments translated to English). The **UI package was integrated into the Docker build** process so the management interface is served directly from the container.

- `build: include UI package in Docker build process`
- `refactor: localize codebase by translating all comments to English`
- `fix: resolve Gemini 500 errors and tool use failures`
- `feat: add ThinkingSequencer for ordered Gemini streaming`

## May 7 — DeepSeek Reasoning Replay

Implemented **mandatory reasoning replay** for DeepSeek models. DeepSeek requires previous assistant reasoning content to be included in subsequent requests — without this, the model loses coherence across multi-turn conversations. The `reasoning` transformer captures reasoning output from responses and replays it automatically.

- `feat: implement mandatory reasoning replay for DeepSeek provider`

## May 7 — Codex (ChatGPT) Integration

Added the **Codex transformer** supporting OpenAI's Responses API, with OAuth authentication via PKCE flow (`ccr codex-auth`). Supports SSE streaming, reasoning/thinking content, tool calls, web search, and image handling. Requires a ChatGPT Plus or Pro subscription.

- `feat: add Codex transformer and support for Codex authentication`

## May 8 — Model Discovery

Added **non-interactive model discovery** (`ccr model get`). Fetches remote models from any provider's API, parses custom JSON structures, and appends missing models to the local configuration without modifying existing entries.

- `feat: add support for custom provider model discovery`

## May 10 — Chrome On-Device (Gemini Nano)

Added the **Chrome On-Device bridge and transformer** for Chrome's built-in Gemini Nano model (~4GB local LLM). Communicates via the Chrome DevTools Protocol (CDP) and provides zero-cost, zero-latency local inference. Features include stall recovery with dynamic temperature scaling, structured JSON output via `responseConstraint`, and full SSE streaming.

- `feat: add Chrome on-device bridge and transformer for Gemini Nano`

## May 12 — Bridge Stability Improvements

Enhanced the Chrome bridge with a multi-session dashboard, idle eviction for stale sessions, improved tool-loop prevention via reflexive detection, and optimized system prompts for tool interaction consistency.

- `feat(bridge): add multi-session dashboard, idle eviction`
- `refactor: migrate bridge to explicit <tool_result> XML tags`

## May 13 — Terminal Forge UI Redesign

Complete visual redesign of the Web UI ("Terminal Forge" design system) with a dark theme, refreshed layouts for the shell, header, dashboard, login, providers, transformers, and router panels. Added a Codex Control Room hero component.

- `feat(ui): introduce Terminal Forge design tokens and force dark theme`
- `feat(ui): redesign App shell, header and dashboard layout`
- `feat(ui): rewrite CodexControlRoom as the routing hero`

## May 18 — Passthrough Mode & DeepSeek Thinking Fix

Added per-provider passthrough mode for direct Anthropic-style auth handling, and fixed DeepSeek thinking without requiring the client to send a reasoning parameter.

- `feat: add per-provider passthrough mode`
- `fix: enable DeepSeek thinking without client reasoning param`

## May 25 — Codex Fixes & Anthropic Effort Passthrough

Fixed Codex transformer arguments for tool calls, restored streaming usage calculation, and passed Claude Code's effort parameter directly instead of mapping through budget_tokens.

- `fix(anthropic): pass through Claude Code effort directly`

## June 1 — Qwen Chat Integration

Added the **Qwen Chat auth transformer** with JWT-based authentication (`ccr qwen-auth`) and a browser-based auth page at `/qwen/auth` supporting a bookmarklet for easy token extraction from `chat.qwen.ai`. Automatic token rotation and stripping of Qwen's trailing `<details>...</details>` metadata block.

- `feat(qwen): add Qwen Chat auth (ccr qwen-auth) and provider transformer`

## Documentation

All features have dedicated integration guides in the docs:

- [Codex Integration](/docs/server/guides/codex)
- [Qwen Chat Integration](/docs/server/guides/qwen)
- [Chrome On-Device](/docs/server/guides/chrome-on-device)
- [DeepSeek Reasoning Replay](/docs/server/guides/deepseek-reasoning)
- [Model Discovery](/docs/server/guides/model-discovery)
