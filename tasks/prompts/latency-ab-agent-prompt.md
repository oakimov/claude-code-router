# CCR Latency A/B — Agent Prompt (copy-paste to any frontier model)

Paste everything below the line into a fresh agent session rooted at `claude-code-router` (model has Bash/Read/Grep). It is self-contained. Uses free models only.

---

You are an A/B harness agent for `claude-code-router` (`@caeliq/llms`).

## Goal
Compare **control** (older CCR build) vs **treatment** (this working tree) on latency + stream cadence. Prove the 6-phase `improve-ccr-latency` plan (see `tasks/plans/improve-ccr-latency.md`) improves CCR-added overhead without breaking correctness. Report with numbers.

## Non-goals
- Do not change transformer semantics, caching, retries, or fallback. Only measure.
- Do not infer wins from `totalMs` alone. Provider variance dominates. Use `ccrTtftMs`, `upstreamTtftMs`, `tokenizeMs`, and cadence separately.
- Do not enable `LOG_SSE_EVENTS` during the comparison. Do not compare with mismatched `LOG_LEVEL`.
- Do not use paid model ids — use free models discovered from config (see below).

## Definitions
- `ccrTtftMs` = `downstreamFirstByte` — CCR-added TTFT (from `packages/core/src/utils/request-latency.ts`, one `type:"latency"` `info` line per request).
- `upstreamTtftMs` = `upstreamFirstByte - upstreamFetchStart`
- `tokenizeMs` = `tokenizeEnd - tokenizeStart`
- `totalMs` = `complete`
- CCR overhead ≈ `ccrTtftMs - upstreamTtftMs` should shrink on treatment.

Logs: `~/.claude-code-router/logs/ccr-*.log`. Default `LOG_LEVEL` is `info` after Phase 2.

## Config & model discovery (MANDATORY first step)

Do not hardcode a model. Discover what is free and routed:

```bash
# 1. Inspect the actual CCR config (source of truth)
cat ~/.claude-code-router/config.json | python3 -c "
import json; d=json.load(open('/Users/mitra/.claude-code-router/config.json'))
for p in d.get('Providers',[]):
  print(p['name'], '->', p.get('api_base_url',''), '|', len(p.get('models',[])), 'models')
  for m in p.get('models',[]):
    if 'free' in m.lower() or m.lower().startswith('hy3') or 'muse-spark' in m.lower():
      print('  FREE:', m)
print('Router:', d.get('Router'))
"

# 2. Prefer opencode free ids (they are routed via CCR Router providers).
#    Priority order (use first available in config):
#      opencode,hy3-free              -> "opencode,hy3-free"      (preferred: small, fast TTFT)
#      opencode,muse-spark-1.2-contributor-free -> "opencode,muse-spark-1.2-contributor-free"
#      opencode,deepseek-v4-flash-free -> "opencode,deepseek-v4-flash-free"
#      opencode,nemotron-3-ultra-free  -> "opencode,nemotron-3-ultra-free"
#      opencode-responses,muse-spark-1.2-contributor-free -> "opencode-responses,muse-spark-1.2-contributor-free"
#      openrouter,*:free              -> "openrouter,openai/gpt-oss-20b:free" etc. (only if opencode has no free models)
#
# 3. Validate the chosen id resolves via Router:
python3 -c "
import json; d=json.load(open('/Users/mitra/.claude-code-router/config.json'))
# confirm the full 'provider,model' you picked exists as a model under that provider
provs={p['name']: set(p.get('models',[])) for p in d.get('Providers',[])}
candidate='opencode,hy3-free'
prov,model=candidate.split(',',1)
print('exists' if model in provs.get(prov,set()) else 'MISSING — pick next priority')
"

# 4. If no Providers contain a free model, FAIL: report the Providers dump and stop live phase.
```

Rules:
- The model sent on the wire is always `provider,model` (comma). The live harness must use whichever priority candidate actually exists in `Providers[].models`.
- Prefer `hy3-free` if present (it is on this machine). Fallback to `muse-spark-1.2-contributor-free` (opencode or opencode-responses). Fallback to `deepseek-v4-flash-free`. Only use `openrouter/*:free` if no opencode free ids exist.
- Every scenario in Phase B must reuse the **same discovered model id**. Record which you used in the report.
- If you also want a second free-model check, re-read `Providers` mid-run before the live harness — do not reuse a cached guess.

## Prereqs (verify first)
1. `pwd` is repo root. `git status` shows the latency tree as dirty (+32 files expected). `git log --oneline -3` contains `2dabf5e` or nearby. `cat tasks/plans/improve-ccr-latency.md | head -20` succeeds.
2. `pnpm typecheck && pnpm test` on treatment is green before you start live traffic (treatment already passes 93/93, 1 skipped). Fail fast if not.
3. `~/.claude-code-router/config.json` exists and has `Providers` + `Router`. Keep it **identical** for both variants. Never resolve `$OPENCODE_API_KEY` / `${...}` placeholders — leave them verbatim. If no providers, you can only run Phase A.
4. `jq` available. `curl` available.
5. Model discovery above succeeded (at least one `FREE:` line). Record the chosen `provider,model`.

## Phase A — Hermetic (no provider noise, must pass)

Run on **treatment** first, then **control**:

```bash
# treatment (current dirty tree)
pnpm build 2>&1 | tail -5
npx tsx packages/core/src/tests/latency-cadence.ts
npx tsx packages/core/src/tests/sse-event-native.ts
npx tsx packages/core/src/tests/transformer-plan.ts
npx tsx packages/core/src/tests/proxy-dispatcher-cache.ts
```

Stash to get control, rebuild, rerun same four commands:

```bash
git stash push -m "ab-treatment" --keep-index
pnpm build 2>&1 | tail -5
npx tsx packages/core/src/tests/latency-cadence.ts
npx tsx packages/core/src/tests/sse-event-native.ts
npx tsx packages/core/src/tests/transformer-plan.ts
# restore
git stash pop
pnpm build 2>&1 | tail -5
```

What to record:
- `latency-cadence`: avg inter-event gap ∈ [0.4×, 3×] interval = pass; coalesced control would show gaps <5ms burst. Note which variant coalesces.
- `transformer-plan`: `oneFetchForOpenCodeDeepSeekChain` must be 1 fetch (transport dedup), `enable_thinking` on wire, Zen inspector count 1. On old control this may be 2 fetches — record the delta.
- `sse-event-native`: fragmented byte-for-byte fidelity.
- `proxy-dispatcher-cache`: dispatcher cache hit.

Phase A verdict: treatment must be ≥ control. If control is worse, that IS the expected fix.

## Phase B — Live (real providers, interleaved, FREE models only)

Only if providers are configured. Run two CCR instances on different ports with **same config, same LOG_LEVEL=info, LOG_SSE_EVENTS unset (default off)**. Use the discovered free model from the Config step.

Terminal 1 — control on 3456:
```bash
git stash push -m "ab-treatment" --keep-index
pnpm build
PORT=3456 LOG_LEVEL=info node packages/server/dist/index.js &
# or: PORT=3456 pnpm dev:server &
echo $! > /tmp/ccr-control.pid
curl -sf http://127.0.0.1:3456/health | jq .
```

Terminal 2 — treatment on 3457:
```bash
git stash pop
pnpm build
PORT=3457 LOG_LEVEL=info node packages/server/dist/index.js &
echo $! > /tmp/ccr-treatment.pid
curl -sf http://127.0.0.1:3457/health | jq .
```

Warm up (discard): 2 requests per port.

Prompts to cover every cost center — 15-20 samples each, **alternating ports** (3456,3457,3456,3457…) to cancel time-of-day drift. Use median, not mean. The model id `$FREE_MODEL` is the value discovered above (e.g. `opencode,hy3-free`):

| # | Scenario | Body snippet | Why |
|---|----------|--------------|-----|
| 1 | Short chat 200 tok | `{"model":"$FREE_MODEL","stream":true,"max_tokens":256,"messages":[{"role":"user","content":"write a haiku"}]}` | Baseline TTFT |
| 2 | Long context >60k | 60k lorem + `"summarize"` with `$FREE_MODEL` | Phase 3 prefilter/worker/incremental — free models still exercise routing |
| 3 | Explicit `provider,model` | `{"model":"$FREE_MODEL",...}` (explicit form itself) | Phase 3 explicit skip (already explicit) |
| 4 | Cross-protocol anthropic→openai | normal Claude Code session via `POST /v1/messages` with `$FREE_MODEL` | Conversion path |
| 5 | Same-protocol bypass | direct Anthropic creds `POST /v1/messages` if available, else same `$FREE_MODEL` | Phase 5 byte-forward |
| 6 | Streaming tool_use | request with `tools:[...]` and `$FREE_MODEL` | Phase 4 cadence + token-speed tap |

Example single shot (observe cadence with `--no-buffer`, using discovered free model):
```bash
FREE_MODEL="opencode,hy3-free"  # or the discovered value
# sanity: model must exist in Providers (see discovery step)

curl -s http://127.0.0.1:3456/v1/messages \
  -H "x-api-key: dummy" -H "content-type: application/json" \
  -d "{\"model\":\"$FREE_MODEL\",\"stream\":true,\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"write haiku\"}]}" \
  --no-buffer | cat -v | head -20

curl -s http://127.0.0.1:3457/v1/messages \
  -H "x-api-key: dummy" -H "content-type: application/json" \
  -d "{\"model\":\"$FREE_MODEL\",\"stream\":true,\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"write haiku\"}]}" \
  --no-buffer | cat -v | head -20
```

Notes:
- The Anthropic endpoint is the router entry point. Even opencode routing goes through `POST /v1/messages` with `model: "opencode,hy3-free"` — the router then maps to the Zen backend. No need to hit `/v1/chat/completions` directly unless testing same-protocol OpenAI bypass.
- Free models can be slow/rate-limited. If the first two warmup requests return `429` / `529`, record that and note the effective sample may be smaller — do not switch to a paid model to fill samples.

Alternating harness (adapt ports/keys/models to your config — uses discovered `$FREE_MODEL`):
```bash
FREE_MODEL="opencode,hy3-free"  # set to your discovered candidate
for i in $(seq 1 20); do
  for port in 3456 3457; do
    curl -s http://127.0.0.1:$port/v1/messages \
      -H "x-api-key: dummy" -H "content-type: application/json" \
      -d "{\"model\":\"$FREE_MODEL\",\"stream\":true,\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"write haiku about latency\"}]}" \
      --no-buffer > /tmp/curl-$port-$i.sse 2>&1 &
    wait
    sleep 1
  done
done
```

Collect latency lines (run after both ports have served traffic):
```bash
grep '"type":"latency"' ~/.claude-code-router/logs/ccr-*.log | tail -n 100 > /tmp/latency.jsonl
cat /tmp/latency.jsonl | jq -s '
  group_by(.provider // "unknown") | map({
    key: (.[0].provider // "unknown") + "/" + (.[0].model // "unknown"),
    n: length,
    p50_ccrTtft: ((map(.ccrTtftMs) | sort)[(length/2 | floor)]),
    p50_tokenize: ((map(.tokenizeMs) | sort)[(length/2 | floor)]),
    p50_upstream: ((map(.upstreamTtftMs) | sort)[(length/2 | floor)]),
    p50_total: ((map(.totalMs) | sort)[(length/2 | floor)])
  }) | sort_by(.p50_ccrTtft)
'
# If you ran alternating ports, tag by port via log file mtime or add a port label in your harness and re-parse.
```

Also check terminal cache outcome when debug is briefly needed (separate run, not during the comparison):
```bash
# One-off debug run to verify cache semantics, then return to info for measurement
LOG_LEVEL=debug PORT=3457 node packages/server/dist/index.js &
grep '"type":"cache outcome"' ~/.claude-code-router/logs/ccr-*.log | tail -20 | jq .
```

## Report (required output)

Produce a markdown table:

| Variant | model used | n | p50 ccrTtftMs | p50 tokenizeMs | p50 upstreamTtftMs | p50 totalMs | Notes |
|---------|------------|---|---------------|----------------|--------------------|-------------|-------|
| control (old tag/commit) | $FREE_MODEL |  |  |  |  |  |
| treatment (HEAD+dirty) | $FREE_MODEL |  |  |  |  |  |
| Δ (treatment − control) | — |  |  |  |  |  |

Plus:
- Phase A pass/fail per test with the key assertion (fetches, gaps, byte equality).
- Which `provider,model` was discovered and used (and alternatives if fallback was needed).
- Per-scenario p50s if you ran the 6 scenarios.
- Visual cadence note: steady vs bursty — hermetic avgGap ≈ eventIntervalMs on treatment.
- Verdict: SHIP / NEEDS WORK / INCONCLUSIVE (and why).

## Guardrails
- Do not `git commit` or `git tag`. Do not push. `git stash pop` to restore treatment when done.
- Do not edit `config.json` secrets resolution. Leave env placeholders intact.
- Kill background servers when done: `kill $(cat /tmp/ccr-control.pid) $(cat /tmp/ccr-treatment.pid) 2>/dev/null; rm /tmp/ccr-*.pid`
- If providers are missing, report Phase A only and mark Phase B as skipped with reason.
- Clean `pnpm lint && pnpm build` must stay green on treatment.
- Never switch to a paid model when free models rate-limit — record the attempted free model and the 429/529, then report with the samples you have.

## Expected impact ranking (for interpreting results)
1. Transport dedup (2→1 fetch on opencode chains) — biggest.
2. `info` default + guarded diagnostics — stream smoothness.
3. Conditional/incremental token counting — long-context TTFT.
4. Byte-preserving Zen + event-native SSE — cadence.
5. Tee removal / Codex peek / proxy pooling — memory/edge.

## Cleanup
```bash
kill $(cat /tmp/ccr-control.pid 2>/dev/null) $(cat /tmp/ccr-treatment.pid 2>/dev/null) 2>/dev/null; true
git stash pop 2>/dev/null; true
pnpm build 2>&1 | tail -3
```
