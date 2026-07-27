# Provider Bridges: Future Optimizations

This document tracks potential improvements and known unknowns for the provider bridges. Most of it concerns `chrome-device-bridge.ts` (Gemini Nano); Cursor SDK items are collected at the end.

## 📖 Core Concepts & Mechanics

### Whitespace Stalling (`MAX_WS_STALL`)
Gemini Nano can enter deterministic loops when emitting highly structured content (e.g., deeply indented code), producing an endless stream of whitespace.
- **Handling**: The bridge monitors `nonWsChars`. If `stallChars` exceeds `MAX_WS_STALL` (currently 1000) without any non-whitespace content, the bridge calls `controller.abort()` to kill the session.
- **Recovery**: This triggers a `truncated: true` signal to the server, which initiates a fallback retry without `responseConstraint` and with a dynamic temperature increase.

## 🛠 Potential Optimizations

### 1. Assistant History Management
- [x] Implement selective inclusion of assistant messages in `buildTurnPrompt` to preserve conversation memory.
- **Done**: Assistant messages are included on the first turn (`processedMsgCount === 0`) so the model sees its own prior tool calls when a conversation starts with pre-existing history. Subsequent turns skip them since the Prompt API session already has them in context.

### 2. Time-Based Stall Detection
- [ ] Implement hybrid stall detection (time + char count) to prevent premature aborts on indented files.
- **Problem**: The current `MAX_WS_STALL` is based purely on the number of whitespace characters (1000). Highly indented files (e.g., deeply nested JSON or Python) might trigger this abort prematurely.
- **Potential Fix**: Implement a hybrid stall detection mechanism that combines the character count with a time-based check (e.g., if no non-whitespace content is produced for 15-20 seconds, regardless of the char count).

### 3. Dynamic Top-K Scaling
- [ ] Implement Top-K increase during fallback retries to break deterministic loops.
- **Problem**: We currently only increase temperature during fallback retries. In some deterministic loops, the model may be stuck between a few high-probability tokens.
- **Potential Fix**: Increase `DEFAULT_TOPK` (e.g., from 40 to 60) during the fallback retry (alongside the temperature increase) to provide the model with a wider selection of tokens to break the loop.

## ✅ Current Status (Baseline)
- [x] Operational Override for hallucination prevention.
- [x] Dynamic temperature scaling for stall recovery.
- [x] JSON-robust extraction (`extractJson`).
- [x] Persistent session management with parameter clamping.
- [x] Tool Result labeling and result-checking guidelines.

## 🖱 Cursor SDK Bridge

### 1. When does Cursor read workspace `AGENTS.md`? — resolved
- [x] Answer: **once per session**, when the agent's rules service is constructed.
- **Evidence**: `@cursor/sdk` `LocalCursorRulesService` sets `this._rules = this.load(...)` in its constructor and only re-reads through an *optional* file watcher (`.cursor/rules/**/*.mdc`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `.cursorignore`). Confirmed empirically: a canary line appended to a live session's `AGENTS.md` was not visible to the model on the next turn, so no watcher is active in the headless local runtime.
- **Consequence**: `refreshWorkspaceGuidance` serves the next agent created against that directory, not the live session. That is fine — a live turn already receives changed host facts through the prompt head/tail, and the in-memory `session.hostEnv` update (used by the scratch-path correction) happens regardless. Documented at the function.

### 2. Model-specific grounding strength — closed, not worth doing
- [x] Decision: keep grounding uniform across models; rely on `scratchPathViolations` for monitoring.
- **Measurement**: guidance + tail reminder is 2,002 chars ≈ **501 tokens**. Real Claude Code turns on a live `glm-5.2` session measured 39.8k–53.5k prompt tokens, so the cost is **~1%**, and it sits at the head of the prompt where Cursor's cache absorbs it (one turn reported 583,872 of 651,111 raw prompt tokens as cached).
- **Why not adaptive**: escalating only after a violation means shipping weaker defaults and waiting for a user-visible failure to trigger the fix, in exchange for ~1%.
