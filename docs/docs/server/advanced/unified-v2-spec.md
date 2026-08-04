---
sidebar_position: 2
---

# Unified v2 Format Specification

## Status and conformance language

This document specifies CCR's planned provider-neutral conversation and inference representation, **Unified v2**. It is an internal data contract, not a client-facing HTTP protocol.

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**, and **MAY** are normative.

Unified v2 is a clean-sheet format. The current `UnifiedChatRequest` and any earlier endpoint plan are not authoritative for this specification.

## Goals

Unified v2 MUST represent the information required by:

- OpenAI Chat Completions
- OpenAI Responses, including conversations, previous responses, background operations, hosted tools, and typed stream events
- Anthropic Messages, including thinking/signatures, citations, documents, cache control, server tools, containers, and context management
- Gemini GenerateContent v1/v1beta
- Vertex Gemini variants
- Antigravity envelopes and Gemini dialect variants
- Gemini Interactions, including continuation, background execution, stateful resources, and events
- Cursor SDK agents, sessions, runs, parked tools, progress/thinking, and resumable output

Any source format MAY be projected to any target format. A projection MUST report unrepresentable semantics and MUST NOT silently discard, weaken, or fabricate required behavior.

Unified v2 guarantees **canonical observable history**. A conversation MAY begin with OpenAI Responses, switch to Gemini, switch to Anthropic, and later switch back to Gemini without losing observable history. Provider-private state MAY be retained for its owner, but MUST NOT be treated as portable.

## Non-goals

Unified v2 does not specify:

- Provider wire mappings or transformation algorithms
- HTTP endpoints or SSE framing
- Provider capability discovery
- A UI state format such as Vercel `UIMessage`
- Multi-instance CCR clustering

## Replacement policy

Unified v2 completely replaces Unified v1.

- Implementations MUST NOT retain a v1 compatibility façade, dual-run mode, feature flag, adapter, deprecated v1 export, or fallback path.
- Existing v1 conversation/session/runtime state MUST NOT be imported into the v2 journal or checkpoint store.
- The initial v2 database starts empty. No historical v1 state is preserved, reconstructed, or migrated.
- Cutover is repository-wide and atomic: all internal consumers MUST use v2 before the replacement is considered complete, and the v1 types and code paths MUST then be deleted.
- This no-migration rule applies to old Unified records and internal conversation/provider state. It does not permit deletion of unrelated user configuration, credentials, presets, or logs.

## Architecture

Unified v2 consists of three layers:

1. **Canonical ordered item journal** — immutable typed semantic items representing observable conversation history.
2. **Typed event grammar** — request, response, operation, and stream lifecycle events that fold into journal items and can be replayed.
3. **Runtime and transport envelope** — non-model-visible execution state such as headers, credentials, abort signals, sockets, routing, retries, and provider handles.

A complete role-based message is a derived convenience projection over ordered items. It is not a second canonical store. Stream deltas are operation events and do not become canonical conversation history individually.

## Version and protocol profiles

Every serializable root MUST carry a schema version:

```ts
type UnifiedSchemaVersion = "2.0";
type UnifiedId = string;

interface UnifiedProtocolProfile {
  family:
    | "openai-chat"
    | "openai-responses"
    | "anthropic-messages"
    | "gemini-generate-content"
    | "gemini-interactions"
    | "vertex-gemini"
    | "antigravity"
    | "cursor-sdk"
    | string;
  version?: string;
  revision?: string;
}
```

Protocol profiles version independently from the core schema.

## Root records

```ts
interface UnifiedRequestV2 {
  schemaVersion: "2.0";
  requestId: UnifiedId;
  conversation: UnifiedConversationInput;
  model: UnifiedModelIntent;
  generation: UnifiedGenerationIntent;
  tools: UnifiedToolDefinition[];
  toolChoice?: UnifiedToolChoice;
  cache?: UnifiedCacheIntent;
  state?: UnifiedStateInput;
  output?: UnifiedOutputIntent;
  metadata?: UnifiedSemanticMetadata;
  source: UnifiedSourceProvenance;
}

interface UnifiedResultV2 {
  schemaVersion: "2.0";
  responseId: UnifiedId;
  conversationId?: UnifiedId;
  operationId?: UnifiedId;
  model: UnifiedResolvedModel;
  status: UnifiedResponseStatus;
  output: UnifiedItem[];
  usage?: UnifiedUsage;
  error?: UnifiedError;
  state?: UnifiedStateOutput;
  provenance: UnifiedResponseProvenance;
  loss?: UnifiedLossReport;
}
```

A root record MUST NOT also be an upstream provider body.

## Conversation identity and completeness

```ts
interface UnifiedConversationInput {
  conversationId?: UnifiedId;
  branchId?: UnifiedId;
  headItemId?: UnifiedId;
  completeness: "complete" | "partial" | "reference-only";
  items: UnifiedItem[];
  continuations?: UnifiedProviderResourceRef[];
}
```

- `complete` means all canonical observable history needed for provider switching is present.
- `partial` means a declared history range is unavailable.
- `reference-only` means execution depends on provider-private state and cannot be portably reconstructed.

Cross-provider continuity is guaranteed only for complete history. Partial and reference-only forms remain representable and MUST produce an explicit portability decision.

## Canonical items

Every canonical item contains:

```ts
interface UnifiedItemBase {
  id: UnifiedId;
  conversationId: UnifiedId;
  branchId: UnifiedId;
  sequence: number;
  parentId?: UnifiedId;
  createdAt: string;
  role?: "system" | "developer" | "user" | "assistant" | "tool";
  visibility: "model" | "client" | "internal";
  status: "in-progress" | "completed" | "incomplete" | "failed" | "cancelled";
  provenance: UnifiedItemProvenance;
  cache?: UnifiedCacheDirective;
  providerData?: UnifiedProviderDataRef[];
}
```

`UnifiedItem` is a closed discriminated union containing at least:

- `message`
- `reasoning`
- `tool-call`, `tool-result`
- `approval-request`, `approval-response`
- `server-tool-call`, `server-tool-result`
- `computer-action`, `computer-action-result`
- `code-execution`, `code-execution-result`
- `search-result`, `retrieval-result`
- `artifact`, `reference`
- `compaction-summary`
- `provider-switch`
- `interruption`
- `error`

Items MUST be append-only. Content or status transitions MUST be expressed as events or revisions. A compaction summary MAY replace only a contiguous prefix and MUST identify its covered range and audit hash. Item and part order is semantic and MUST be preserved.

## Content parts

Every content part has a stable part ID. `UnifiedContentPart` includes:

- `text`, with language, style, and annotations
- `reasoning`, with disclosed summary/text/state and optional private-state reference
- `refusal`
- `image`, `audio`, `video`, `document`, `file`, and `binary`
- `json`
- `citation` and `grounding-source`
- `artifact-reference`

```ts
type UnifiedDataSource =
  | {
      kind: "inline";
      mediaType: string;
      data: Uint8Array | string;
      encoding: "bytes" | "base64" | "utf8";
    }
  | { kind: "url"; url: string; headersRef?: UnifiedId }
  | { kind: "artifact"; artifactId: UnifiedId; sha256: string }
  | { kind: "provider-resource"; resource: UnifiedProviderResourceRef };
```

Media metadata MUST be able to express names, media types, sizes, checksums, page/range/temporal selection, image detail or media resolution, video offsets/FPS, transcripts, document context/title, and provider constraints.

Citation and grounding data MUST retain source identity, URI/title, quoted text, spans plus indexing unit, confidence, retrieval/search queries, retrieval time, and the supported content relationship. It MUST NOT be reduced to URL-only annotations.

Large inline content SHOULD be externalized to the artifact store at persistence boundaries.

## Reasoning and private latent state

Observable reasoning text or summaries belong in canonical history. A source-visible redaction marker also belongs in canonical history.

Encrypted reasoning, Anthropic signatures, Gemini thought signatures, OpenAI private continuation data, and Cursor-private agent state MUST be represented as encrypted owner-private resources referenced from items or checkpoints. Synthetic signatures MUST be marked as synthetic and MUST NOT be confused with provider-issued state.

## Tools and agent lifecycle

### Definitions

The format MUST support function tools with JSON Schema, strictness, description, deferred loading, allowed callers, execution locality, and cache directives.

It MUST also support typed hosted tools including web search/fetch, file search, retrieval, URL context, maps, code execution, computer use, MCP, containers, skills, memory, and registered future types.

Every tool declares:

```ts
execution: "client" | "ccr" | "provider";
```

and a portability classification.

### Calls and results

Tool lifecycle states include:

- `arguments-streaming`
- `arguments-ready`
- `approval-required`
- `approval-approved`
- `approval-denied`
- `executing`
- `output-ready`
- `output-error`
- `cancelled`

Calls retain the stable call ID, name/type, parsed input, raw argument text, incremental fragments, caller, execution owner, provider-executed status, and private linkage. Results retain call ID, structured or multimodal output, error details, and execution ownership.

## Generation and output intent

`UnifiedGenerationIntent` represents concepts rather than provider field names. It includes maximum output tokens, sampling controls, stop sequences, candidate count, seed, penalties, log probabilities, reasoning effort/budget/summary, parallel tool policy, service tier/priority, prediction/prefill, safety settings, and output modalities/media controls.

`UnifiedOutputIntent` includes unconstrained text, JSON object, JSON Schema, strict structured output, text grammar, media output, and requested includes or annotations.

Each intent carries requirement strength:

```ts
type UnifiedRequirement = "required" | "preferred" | "optional";
```

A projection MUST NOT silently weaken a required intent.

## Canonical continuity and provider-private checkpoints

### Canonical observable history

All committed observable items are persisted independently of provider. A provider switch reconstructs target input from the canonical branch head. This includes user/assistant content, disclosed reasoning summaries, calls/results, citations, artifacts, errors, interruptions, and provenance.

### Provider-private checkpoints

Provider-private checkpoints are encrypted and stored separately from canonical items:

```ts
interface UnifiedProviderCheckpoint {
  checkpointId: UnifiedId;
  conversationId: UnifiedId;
  branchId: UnifiedId;
  journalHeadItemId: UnifiedId;
  owner: UnifiedResourceOwner;
  resourceType:
    | "response"
    | "conversation"
    | "interaction"
    | "session"
    | "run"
    | "container"
    | "cached-content"
    | "file"
    | "reasoning"
    | string;
  encryptedPayload: Uint8Array;
  status: "active" | "stale" | "expired" | "revoked" | "invalid";
  createdAt: string;
  expiresAt?: string;
}
```

The owner scope includes protocol profile, provider identity, credential fingerprint—not the credential—endpoint fingerprint, project/region/tenant, model/family, and API revision.

A checkpoint is directly resumable only when its journal head equals the current branch head and its owner is compatible. After another provider appends turns, the older checkpoint is stale for direct continuation but MAY be retained. Returning to that provider reconstructs from canonical history and creates a new checkpoint at the current head.

Private state MUST NOT cross providers, credentials, endpoints, projects, regions, tenants, or incompatible models.

### Operations and background work

`UnifiedOperation` represents queued, running, requires-action, completed, failed, cancelled, and expired work. It includes owner, source branch/head, provider job reference, polling or webhook state, result/error, cancellation capability, timestamps, and cleanup lease.

A committed operation lifecycle or result mutation MUST atomically update the journal and conversation mutation time.

## Cache representation

```ts
interface UnifiedCacheIntent {
  mode: "automatic" | "explicit" | "disabled";
  affinityKey?: string;
  breakpoints?: Array<{
    itemId: UnifiedId;
    partId?: UnifiedId;
    ttl?: "5m" | "1h" | number;
  }>;
  retention?: "request" | "conversation";
}
```

Portable cache intent is separate from provider-owned cache resources. A Gemini cached-content name, OpenAI response/conversation ID, Anthropic container, or Cursor session is not a portable cache key.

Cache usage MUST distinguish uncached input, cache reads, and cache writes/creation where reported. Unknown counters MUST remain unknown rather than becoming zero.

## Provider data and projection loss

Canonical items MUST NOT contain an unrestricted provider-options object.

```ts
interface UnifiedProviderDataRef {
  profile: UnifiedProtocolProfile;
  scope: "portable-annotation" | "owner-private";
  schema: string;
  resourceId: UnifiedId;
}
```

Portable annotations require a registered validated JSON schema and MUST contain no credentials. Owner-private payloads are encrypted outside canonical items. Raw request/response bodies are not correctness data and MAY appear only in an opt-in encrypted diagnostic store.

```ts
interface UnifiedLossReport {
  disposition: "lossless" | "degraded" | "unsupported";
  entries: Array<{
    path: string;
    feature: string;
    severity: "info" | "warning" | "error";
    reason: string;
    sourceItemId?: UnifiedId;
    targetCapability?: string;
  }>;
}
```

An error-level loss for required semantics MUST prevent execution. Preferred or optional degradation MAY proceed only with an explicit report.

## Usage, status, safety, and errors

`UnifiedUsage` includes optional counters for input/output/total, cache read/write, reasoning, media, accepted/rejected prediction, tool prompt, modality breakdowns, server-tool calls, and reported cost or credits.

Every counter declares:

- Scope: `request`, `operation`, `session`, or `cumulative`
- Source: `reported`, `estimated`, or `derived`

This prevents a cumulative Cursor session counter from being interpreted as per-turn usage.

Status and finish reasons include stop, length, tool calls, content filtering/safety, refusal, max context, malformed tool call, requires action, paused, cancelled, interrupted, failed, incomplete, and unknown provider reason. Original reasons remain available through registered annotations.

`UnifiedError` carries a stable CCR category/code, HTTP/gRPC/provider status, retryability, retry-after, sanitized message, field path, operation/item relation, provider request ID, and optional private diagnostic reference. It MUST NOT contain credentials or unsanitized provider content.

## Typed event grammar

Every `UnifiedEventV2` includes schema version, operation ID, event ID, monotonically increasing sequence, timestamp, and applicable item/part/call IDs.

The event union includes:

- Operation created/queued/running/requires-action/completed/failed/cancelled
- Response started/metadata
- Item started/delta/completed/failed
- Text and reasoning deltas
- Signature-ready
- Tool arguments delta/ready
- Tool lifecycle transitions
- Citation/source/artifact ready
- Usage update
- Checkpoint ready/invalidated
- Warning/loss update
- Heartbeat
- Error
- Stream completed

Events are idempotent by `(operationId, sequence)` and totally ordered within one operation. Stable IDs relate deltas to completed items. A deterministic fold MUST produce the same `UnifiedResultV2` and canonical items regardless of transport chunk boundaries.

Durable stream events support replay but do not become canonical journal entries individually.

## Transport and runtime separation

### Serializable transport metadata

`UnifiedTransportMetadata` MAY contain source profile, endpoint/action, sanitized method/path template, request/trace/idempotency keys, accepted response mode, stream preference, content encoding/type, allowlisted beta or revision semantics, and safe client-family identification.

It MUST NOT contain credentials, cookies, arbitrary raw headers, or secrets.

### Runtime-only context

`UnifiedExecutionContext` is non-serializable and non-persisted. It includes request/reply/socket, abort signals, logger, resolved credentials, raw headers, provider clients/SDK handles, database and writer handles, active publishers/subscribers, and retry/fallback state.

CCR consumes client credentials and generates provider authentication independently. Header preservation uses profile allowlists; a forward-all-headers mode is forbidden.

## Durable storage

### Deployment assumption

One CCR process owns storage and upstream connections. Multiple independent clients may connect from any machine, subscribe, disconnect, and reconnect. SQLite is sufficient for this deployment. Multi-instance CCR requires another store implementation but does not change Unified v2.

### Storage interface

`UnifiedStateStore` MUST be defined independently of SQLite. It supports transactional append/fold, branches and heads, checkpoints/resources, artifacts, operations, event batches, leases, replay, compaction, and retention cleanup.

The store and its SQLite implementation belong to `@caeliq/llms` (the sdk), not to the CCR HTTP server. Store construction MUST accept an explicit configuration — DB path, artifact directory, retention days, encryption key material — supplied by the caller. It MUST NOT resolve `~/.claude-code-router` or read `config.json` itself. The host application (the CCR server) is responsible for resolving that configuration and paths and passing them in; the server otherwise remains limited to inbound transport, auth, and routing into the sdk.

### SQLite requirements

The initial implementation uses built-in `node:sqlite` `DatabaseSync` behind a serialized asynchronous writer queue.

The database MUST use:

- WAL mode
- Foreign keys
- Busy timeout
- `STRICT` tables
- Transactional schema migrations with an independent schema version
- Prepared statements
- Bounded event batches
- One serialized writer queue
- Startup integrity checks and a backup hook

Minimum tables:

- `conversations`, `branches`
- `journal_items`, `journal_revisions`
- `operations`, `stream_events`
- `provider_checkpoints`, `provider_resources`
- `artifacts`, `artifact_refs`, `pending_artifact_gc`
- `leases`, `schema_migrations`

Provider-private data MUST be encrypted at rest with a versioned CCR-local key. Credentials MUST never be stored. Large artifacts use a SHA-256 content-addressed directory with SQLite metadata and refcounts.

## Resumable live streams

- Each operation has one producer and multiple independent subscribers.
- Each durable event has an operation-local sequence.
- Resume cursors identify `{ conversationId, operationId, afterSequence }` and MUST be authorization-bound and tamper-resistant.
- Reconnection replays later SQLite events and atomically joins the in-memory publisher without a replay/live gap.
- Subscribers MUST NOT compete for the producer.
- Client disconnect detaches only that subscriber. Explicit cancellation is separate.
- Process restart preserves committed events. An ordinary provider HTTP stream becomes interrupted; provider-owned resumable/background work MAY continue from its checkpoint.
- Terminal fold, journal append, usage/status, operation state, and checkpoint updates MUST commit atomically.

## Retention and cleanup

One configuration setting controls the complete conversation scope:

```jsonc
{
  "conversationRetentionDays": 7
}
```

The default is seven days. `conversations.last_updated_at` is the only retention clock. **Touched means meaningful committed mutation, not access.**

The following update it transactionally:

- Canonical item append or revision
- Tool or approval lifecycle change
- Provider switch
- Operation lifecycle or output change
- Committed stream output/event
- Checkpoint/resource creation, replacement, invalidation, or execution-relevant expiry observation
- Artifact reference mutation
- Background result, error, or cancellation

The following do not update it:

- Reading, listing, or exporting
- Reconstructing target input
- Replaying events
- Attaching or detaching subscribers
- Cleanup or integrity scans
- Checkpoint validity reads
- Failed requests that commit nothing

Updating an old conversation resets its full retention period. Active operations hold cleanup leases, but a lease alone does not touch the timestamp.

Cleanup runs on startup and periodically in bounded transactions. It deletes the complete eligible conversation scope and only unshared artifacts. Completed stream events use the same retention period; there is no separate stream TTL.

## Security and privacy invariants

- Conversation and resume IDs are opaque but are not authorization credentials.
- No credential, cookie, API key, bearer header, or SDK token object may enter Unified v2, SQLite, artifacts, or logs.
- Owner-private state is encrypted and key-versioned.
- Resource reuse requires exact compatible owner scope.
- Data is sanitized before persistence.
- Item, event, payload, artifact, and subscriber-lag sizes are bounded.
- Persisted provider and tool data is untrusted data, not instructions to CCR.

## Validation and canonicalization

Validators MUST enforce schema version, discriminated unions, unique IDs and references, item/event ordering, legal state transitions, tool/result correlation, owner scope, branch/head consistency, serializability, numeric validity, secret exclusion, and size/depth limits.

Canonical serialization preserves item/part order, sorts semantically unordered object keys, externalizes persisted bytes, and produces SHA-256 fingerprints. It MUST preserve unknown versus zero, empty versus absent, and profile revision.

## Versioning

Unified v2 starts at `2.0`.

- Minor versions are additive.
- Major versions change required fields, invariants, or existing meanings.
- Provider profiles version independently.
- SQLite schema version is independent.
- Unknown optional registered data may be retained but not executed.
- Unknown required semantics produce an `unsupported` loss report.

## Conformance scenarios

An implementation conforms when it demonstrates:

1. OpenAI-origin turn → Gemini → Anthropic → Gemini without losing canonical observable history.
2. A provider checkpoint at head A cannot directly resume after another provider advances to head B; returning to its owner reconstructs history and creates a checkpoint at B.
3. Tool argument streaming, approval, execution, interruption, reconnect, and result remain coherent.
4. Background operations survive client disconnect and update retention only on committed mutation.
5. Multiple clients consume and reconnect independently without stealing events.
6. Restart preserves committed replay events and distinguishes interrupted HTTP streams from provider-resumable work.
7. A six-day-old conversation updated today survives a new seven-day period, while a seven-day mutation-inactive scope is removed.
8. Required incompatible semantics yield `unsupported`; optional degradation is explicit.
9. SQLite contains neither credentials nor plaintext owner-private state.
10. Identical journals and event streams yield identical results and fingerprints.

## Design comparison: Vercel AI SDK

Vercel AI SDK 6 persists neutral `UIMessage[]` snapshots and derives `ModelMessage`; applications provide storage, and resumable live SSE commonly uses Redis plus an `activeStreamId`. It preserves typed tool/reasoning/provider metadata but does not define durable provider-continuation checkpoints or retention.

Unified v2 adopts neutral persistence, typed lifecycles, stable IDs, and disconnect/cancel separation. It deliberately uses an append-only canonical journal, separates observable metadata from encrypted owner-private checkpoints, and uses SQLite for one CCR process serving many clients. Redis becomes necessary only if multiple CCR processes must share producers and state.
