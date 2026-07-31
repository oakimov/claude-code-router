---
title: New Providers and Prompt Caching Fixes
date: 2026-07-26
tags: [changelog, fork, providers, caching]
---

# New Providers and Prompt Caching Fixes

Since the [June fork changelog](/blog/2026/06/01/fork-changelog), this fork added several subscription-backed providers and fixed prompt-caching reliability across the transformer stack. Exact introduction dates from the git history:

{/* truncate */}

## June 18, 2026 — Claude Auth (Pro/Max Subscription)

Added the **claude-auth OAuth PKCE flow** (`ccr claude-auth`) so Claude Pro/Max subscription tokens can be routed through the gateway like other subscription-backed providers.

- `feat: add claude-auth OAuth PKCE flow for Claude Pro/Max subscription routing`

## July 19, 2026 — Cursor SDK Provider

Added the **Cursor SDK provider bridge** with docs, enabling Cursor-backed models through the router with streaming and thinking transport support.

- `feat(cursor): add Cursor SDK provider bridge with docs`

## July 20, 2026 — Claude Auth Cache-Control Round-Trip

Hardened claude-auth with multi-client `anthropic-beta` header resolution and **cache-control round-trip preservation**, so prompt-cache breakpoints survive transformer passes instead of being stripped or rewritten incorrectly.

- `feat(claude-auth): multi-client anthropic-beta resolution and cache-control round-trip preservation`

## July 24, 2026 — Provider-Native Prompt Caching

Shipped **provider-native prompt caching** across Anthropic, OpenAI/Codex, OpenRouter, Vercel, Gemini, and related paths, plus OpenCode Zen session self-heal. This is the main caching reliability fix for long sessions and multi-provider routing.

- `feat(caching+opencode): provider-native prompt caching and OpenCode Zen session self-heal`
- `fix(codex): make PAT and OAuth auth reliable`

## July 26, 2026 — Antigravity OAuth & Cursor/Gemini Hardening

Added **Antigravity OAuth** (`ccr antigravity-auth`) for Google Antigravity-backed Claude/Gemini access, and hardened Gemini/Cursor streaming, thinking transport, and interrupted SDK stream recovery.

- `feat(antigravity+cursor): add Antigravity OAuth and harden Gemini/Cursor paths`
- `fix(cursor): harden streaming and thinking transport`
- `fix(cursor): recover cleanly after interrupted SDK streams`

## Documentation

- [Claude Auth](/docs/server/guides/claude-auth)
- [Cursor SDK](/docs/server/guides/cursor)
- [Codex Integration](/docs/server/guides/codex)
- [Earlier fork timeline](/blog/2026/06/01/fork-changelog)
