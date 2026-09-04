---
title: FIM Completions API
---

# FIM Completions API

一等入站 FIM Completions 协议（`openai_fim_completions`）。协议所有者：`Fim`。
与聊天路径 `/v1/messages`、`/v1/chat/completions`、`/v1/responses` **独立的流水线**。

典型客户端（例如 [Zed](https://zed.dev) Edit Prediction 的 Codestral 提供商）使用
**Codestral** 请求/响应线格式。CCR 保持该客户端契约，并可路由到 Codestral 本身，
或本地 LM Studio 中的 Qwen 模型。

## 端点

| 路径 | 方法 |
|------|------|
| `/v1/fim/completions` | POST |
| `/fim/completions` | POST（别名） |

## 客户端请求（v1）

仅 Codestral/Mistral 形态 — `prompt` + 可选 `suffix`。不做入站模板自动检测。

```json
{
  "model": "codestral-latest",
  "prompt": "def add(a, b):\n",
  "suffix": "\n    return result",
  "max_tokens": 128,
  "temperature": 0.2,
  "stream": false
}
```

裸 `model` 经 `Router.fim`（再回退 `Router.default`）解析。推荐显式 `provider,model`。

## 客户端响应（v1）

响应线格式匹配**入站** kind。v1 入站为 Codestral/Mistral，因此客户端收到
Codestral 形态 JSON — 即使上游是 LM Studio/Qwen（跨族编码）。同族
Codestral→Codestral 为响应透传（不重塑）。

```json
{
  "id": "…",
  "object": "chat.completion",
  "model": "…",
  "created": 1718659669,
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  },
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "    result = a + b" },
      "finish_reason": "stop"
    }
  ]
}
```

| 路径 | 请求 | 响应 |
|------|------|------|
| Codestral（`fim.mistral`） | 同族 body 透传 | 同族透传（上游 Codestral 线格式不变） |
| LM Studio Qwen（`fim.qwen`） | 编码为 `/v1/completions` + FIM tokens | 上游 `text_completion` → **入站**（v1: Codestral）线格式 |

流式 SSE 同样按入站 kind 编码。Zed 等客户端当前以 `"stream": false` 调用 FIM。

日后加入 Qwen/DeepSeek **入站**时，客户端响应将跟随该入站线格式，而不再固定为 Codestral。

---

## 配置：Codestral

编辑器说 Codestral FIM、希望 CCR 代理 Mistral API（鉴权重写、路由、可选 fallback）时使用。

### 1. 提供商

专用 FIM 条目 — **不要**在同一提供商上叠加聊天 `mistral`：

```json
{
  "name": "codestral-fim",
  "api_base_url": "https://codestral.mistral.ai/v1/fim/completions",
  "api_key": "$CODESTRAL_API_KEY",
  "models": ["codestral-latest"],
  "transformer": { "use": ["fim.mistral"] }
}
```

`fim.mistral` 仅设置 URL + `Authorization: Bearer …`。JSON body 原样透传（同族透传）。

### 2. Router

```json
"Router": {
  "fim": "codestral-fim,codestral-latest"
}
```

### 3. 将客户端指向 CCR

示例（Zed Edit Prediction → Codestral 提供商）：

- **API URL**：仅 CCR 基址，例如 `http://127.0.0.1:3456`（Zed 会追加 `/v1/fim/completions`）。
- **API key**：必须匹配 CCR `APIKEY`（不是你的 Mistral key）。调用 Mistral 时 CCR 会替换 `$CODESTRAL_API_KEY`。
- **Model**：`codestral-latest`（或默认）；裸名经 `Router.fim` 解析。

---

## 配置：LM Studio 中的 Qwen

**客户端仍说 Codestral FIM**，但推理在本地（Qwen2.5-Coder 等）经 LM Studio 的
OpenAI 兼容服务运行时使用。

### 1. LM Studio

1. 加载支持 FIM 的 coder 模型（例如 `qwen/qwen2.5-coder-14b` 或 7B 变体）。
2. 启动本地服务（默认 `http://127.0.0.1:1234`）。

### 2. 提供商（`fim.qwen`）

```json
{
  "name": "lmstudio-qwen-fim",
  "api_base_url": "http://127.0.0.1:1234/v1/completions",
  "api_key": "lm-studio",
  "models": ["qwen/qwen2.5-coder-14b"],
  "transformer": { "use": ["fim.qwen"] }
}
```

`models[0]`（以及 `Router.fim`）须与 LM Studio 暴露的**精确**模型 id 一致。

**Docker**：若 CCR 在 Compose 中、LM Studio 在宿主机，请用宿主机网关代替 `127.0.0.1`：

```json
"api_base_url": "http://host.docker.internal:1234/v1/completions"
```

### 3. `fim.qwen` 做什么

| 方向 | 行为 |
|------|------|
| 请求 | 客户端 `{ prompt, suffix }` → 单个带 `<\|fim_prefix\|>…<\|fim_suffix\|>…<\|fim_middle\|>` 的 `prompt`；去掉 `suffix`；URL 强制为 `/v1/completions` |
| 响应 | LM Studio `text_completion` → **入站**线格式（v1 mistral：`chat.completion` + `message.content`） |

### 4. Router

```json
"Router": {
  "fim": "lmstudio-qwen-fim,qwen/qwen2.5-coder-14b"
}
```

可同时保留两个提供商，只切换 `Router.fim`。

### 5. 延迟与编辑器

本地中/大型模型做完整非流式补全往往比 Codestral **慢得多**。每次击键就取消并重请求的编辑器（Zed 使用 `"stream": false`）可能在 LM Studio 完成前中止 — CCR 会记录 `client disconnected` / `AbortError`。这是客户端关闭连接，不是路由缺失。请改用更小/更快的本地模型，或在需要 Codestral 级延迟时使用 Codestral。

---

## 同时配置两个提供商

```json
{
  "Providers": [
    {
      "name": "codestral-fim",
      "api_base_url": "https://codestral.mistral.ai/v1/fim/completions",
      "api_key": "$CODESTRAL_API_KEY",
      "models": ["codestral-latest"],
      "transformer": { "use": ["fim.mistral"] }
    },
    {
      "name": "lmstudio-qwen-fim",
      "api_base_url": "http://127.0.0.1:1234/v1/completions",
      "api_key": "lm-studio",
      "models": ["qwen/qwen2.5-coder-14b"],
      "transformer": { "use": ["fim.qwen"] }
    }
  ],
  "Router": {
    "fim": "codestral-fim,codestral-latest"
  },
  "fallback": {
    "fim": ["lmstudio-qwen-fim,qwen/qwen2.5-coder-14b"]
  }
}
```

需要本地推理且不想改编辑器时，将 `Router.fim` 切到 LM Studio 目标即可。

---

## 其他提供商

### DeepSeek（`fim.deepseek`）

```json
{
  "name": "deepseek-fim",
  "api_base_url": "https://api.deepseek.com/beta/completions",
  "api_key": "$DEEPSEEK_API_KEY",
  "models": ["deepseek-chat"],
  "transformer": { "use": ["fim.deepseek"] }
}
```

### DashScope 上的 Qwen（`fim.qwen`）

与 LM Studio 相同转换器；不同 base URL：

```json
{
  "name": "dashscope-qwen-fim",
  "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/completions",
  "api_key": "$DASHSCOPE_API_KEY",
  "models": ["qwen-coder-turbo"],
  "transformer": { "use": ["fim.qwen"] }
}
```

---

## 架构说明

- 入站 kind 定义**客户端契约**（请求解析 + 响应编码）。v1 入站仅为 Codestral/Mistral。
- 流程：入站 → Unified FIM → 出站 `fim.*`。
- **同族**（`inbound === outbound`）：请求与响应 body 透传（仅鉴权/URL）。
- **跨族**：出站转换器按提供商编码请求；流水线将上游响应编码回**入站**线格式（并非永远固定 Codestral）。
- 未来的 Qwen/DeepSeek 入站适配器接入同一接缝；届时客户端响应将匹配那些入站线格式。
- 配置**专用** FIM 提供商；不要将聊天转换器（`mistral`、`deepseek` 等）与 `fim.*` 叠在同一提供商上。
- 遗留客户端 `/v1/completions` 仍不路由（404）。

另见：[路由](/docs/server/config/routing)（`Router.fim`）、[提供商](/docs/server/config/providers)、[转换器](/docs/server/config/transformers)（`fim.*`）。
