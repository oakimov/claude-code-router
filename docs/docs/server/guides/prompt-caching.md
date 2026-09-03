---
sidebar_position: 9
---

# Prompt Cache Lifetimes

How long a warm prefix cache survives, per provider path through CCR. The short version: **every path CCR routes to uses a ~5-minute idle TTL unless the client explicitly requests otherwise** — including opencode Zen. There is no 1-hour caching anywhere in the default chain, despite what the billing dashboard's Cache Read column may suggest.

## The tiers that exist

**Anthropic.** Two ephemeral tiers ([Claude docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)): default 5-minute, and opt-in 1-hour via `cache_control: { "type": "ephemeral", "ttl": "1h" }` at 2× write cost (vs 1.25× for 5-minute). Each reuse refreshes the timer, so active traffic stays hot indefinitely — only *idle gaps* kill the cache. Usage splits creation tokens into `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` when both are in play.

**OpenAI-compatible prefix cache.** Automatic, no markers. `in_memory` entries typically survive **5–10 minutes idle** (up to an hour); the `24h` extended tier holds ~30 minutes typically. Same refresh-on-reuse rule. Caches are machine-local, so overflow routing can miss even inside the TTL.

## Claude Code: 5 minutes by default, 1 hour conditionally

Claude Code's default is the 5-minute tier: its breakpoints carry bare `cache_control: { type: "ephemeral" }`, and a request to expose the knob ([anthropics/claude-code#60316](https://github.com/anthropics/claude-code/issues/60316)) was closed as not planned. However, reverse-engineering ([anthropics/claude-code#43566](https://github.com/anthropics/claude-code/issues/43566)) shows the client *does* request `ttl: "1h"` for eligible Claude.ai subscriber traffic on included plan usage — gated behind subscriber status, non-overage state, and an internal allowlist. On API keys or Extra Usage it silently falls back to 5 minutes. So "Claude Code keeps cache an hour" is true only for that subscriber slice, and CCR cannot change which tier the client asks for — it can only preserve it.

## opencode Zen: 5 minutes, durable routing

Verified against the opencode console codebase (`packages/console/app/src/routes/zen`):

- **Anthropic downstream**: Zen injects `{ cache_control: { type: "ephemeral" } }` itself (`util/provider/anthropic.ts`) — no `ttl`, so the 5-minute tier.
- **OpenAI downstream**: the usage mapper tracks `cacheWrite5mTokens` and hardcodes `cacheWrite1hTokens: undefined` (`util/provider/openai.ts`) — Zen never requests 1-hour retention.
- **No `ttl` string exists anywhere** under the Zen routes, and `prompt_cache_key` is passed through untouched, never interpreted for retention.
- **Routing is durable, cache is not**: Zen pins a backend per `modelId/sessionId` in a database table (`ModelStickyProviderTable`, no expiry; falls back to workspace ID or IP when no session is sent). So a stable `x-opencode-session` always reaches the *same backend* — but that backend's KV cache still expires after minutes idle. Durable routing ≠ durable cache.

## Requesting longer retention yourself

`cacheWrite1hTokens` (and Anthropic's `ephemeral_1h_input_tokens`) are *reporting* fields — you don't set them, you earn them by requesting the longer tier:

- **Anthropic downstream**: `cache_control: { "type": "ephemeral", "ttl": "1h" }` on breakpoints. Rules: max 4 breakpoints, 1-hour entries must precede 5-minute ones (else 400), 2× write cost. TTL is not part of the cache key — 5m and 1h requests read each other's entries both ways.
- **OpenAI downstream (≤5.5 families)**: top-level `prompt_cache_retention: "24h"` (same price either tier, needs ≥1024 tokens). On 5.6+ use `prompt_cache_options: { "ttl": "30m" }` instead — the retention field is deprecated/invalid there.

**Through Zen, neither knob survives.** Zen rebuilds both body shapes from allowlists: the Anthropic builder injects bare `ephemeral` itself (dropping client markers), and the OpenAI builder emits a fixed field list with no retention key. Only a Zen-side change could enable 1-hour through that path. On direct (non-Zen) providers CCR could promote breakpoints / inject the retention param itself — currently it only preserves a client-supplied `ttl`, never upgrades. That promotion would have to be opt-in per provider: 2× write cost and strict model-gating (unsupported models 400) make a global default unsafe.

CCR is TTL-neutral: it preserves a client-supplied `ttl` on cache breakpoints (`packages/core/src/utils/cacheControl.ts` propagates the latest explicit marker's `ttl` to the automatic breakpoint) but never injects `1h` on its own. Since Claude Code never asks, the whole Claude Code → CCR → Zen chain runs on the 5-minute tier end to end.

## What CCR itself does (besides retention)

CCR *does* control the other half of the equation — session stability. A stable `x-opencode-session` (persisted in `ccr-sessions.json`, same value across restarts) keeps Zen routing you to the same backend, which is a prerequisite for any hit. Without it, even a 30-second gap can miss on a different machine.

## Practical guidance

- **Diagnose with gaps, not vibes.** Every CCR cache-outcome log line carries `msSinceLastTurn`: a miss after a >5-minute gap is ordinary TTL expiry, not a bug. A miss after seconds is worth investigating.
- **Interactive use is unaffected.** Turns seconds apart refresh the timer continuously.
- **Sparse workflows pay re-warm per gap.** If pauses over 5 minutes are routine, expect one cold turn per resumption (our logs show follow-ups hitting 0.99 immediately after). The only mitigations are keeping the conversation warm with periodic cheap turns, or a client that requests the 1-hour tier — which Claude Code currently cannot do.
- **Cost math reminder.** 1-hour writes cost 2× input vs 1.25× for 5-minute; reads are ~10% either way. The longer tier only wins when reuse within the hour exceeds ~24% — true for most agentic sessions, which is why the missing knob stings.
