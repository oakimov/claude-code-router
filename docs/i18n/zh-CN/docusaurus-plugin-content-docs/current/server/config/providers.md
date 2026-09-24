---
title: 提供商配置
sidebar_position: 2
---

# 提供商配置

配置 LLM 提供商的详细指南。

## 支持的提供商

### DeepSeek

```json
{
  "name": "deepseek",
  "api_base_url": "https://api.deepseek.com/chat/completions",
  "api_key": "your-api-key",
  "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
  "transformer": {
    "use": ["deepseek"]
  }
}
```

### Groq

```json
{
  "name": "groq",
  "api_base_url": "https://api.groq.com/openai/v1/chat/completions",
  "api_key": "your-api-key",
  "models": ["llama-3.3-70b-versatile"]
}
```

### Gemini

```json
{
  "name": "gemini",
  "api_base_url": "https://generativelanguage.googleapis.com/v1beta/models/",
  "api_key": "your-api-key",
  "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
  "transformer": {
    "use": ["gemini"]
  }
}
```

### OpenRouter

```json
{
  "name": "openrouter",
  "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
  "api_key": "your-api-key",
  "models": [
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.5-pro-preview"
  ],
  "transformer": {
    "use": ["openrouter"]
  }
}
```

### Mistral

```json
{
  "name": "mistral",
  "api_base_url": "https://api.mistral.ai/v1/chat/completions",
  "api_key": "your-api-key",
  "models": ["mistral-large-latest", "mistral-small-latest"],
  "transformer": {
    "use": ["mistral"]
  }
}
```

### Cerebras

```json
{
  "name": "cerebras",
  "api_base_url": "https://api.cerebras.ai/v1/chat/completions",
  "api_key": "your-api-key",
  "models": ["cerebras-gpt"],
  "transformer": {
    "use": ["cerebras"]
  }
}
```

### Codex（ChatGPT）

需要通过 `ccr codex-auth` 完成 OAuth 认证。

```json
{
  "name": "codex",
  "baseUrl": "https://chatgpt.com/backend-api/codex",
  "apiKey": "oauth_dummy_key",
  "models": ["gpt-5", "gpt-5-high", "gpt-5-mini"],
  "transformer": {
    "use": ["openai-responses", "codex"]
  }
}
```

### Cursor（SDK）

通过 `@cursor/sdk` 路由到 Cursor 模型。认证使用以 `crsr_` 开头的控制台密钥，或环境变量 `CURSOR_API_KEY`。

```json
{
  "name": "cursor",
  "api_base_url": "https://cursor.com",
  "api_key": "$CURSOR_API_KEY",
  "models": ["composer-2", "claude-opus-4-8", "gpt-5.4"],
  "transformer": {
    "use": [
      [
        "cursor-sdk",
        {
          "cursorMode": "bridge"
        }
      ]
    ]
  }
}
```

使用 `ccr model get cursor` 发现模型。详见 [Cursor SDK 集成指南](/docs/server/guides/cursor)。

### Antigravity

通过 Google Antigravity 网关路由，使用 `ccr antigravity-auth` 完成 OAuth。将 `gemini` 链接在 `antigravity-auth` **之前**。

```json
{
  "name": "antigravity",
  "api_base_url": "https://daily-cloudcode-pa.sandbox.googleapis.com",
  "api_key": "oauth",
  "project_id": "$ANTIGRAVITY_PROJECT_ID",
  "models": [
    "gemini-3-pro-high",
    "gemini-3-flash",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking"
  ],
  "transformer": {
    "use": [
      ["gemini", { "cachedContent": false, "thoughtSignatureFallback": "skip" }],
      "antigravity-auth"
    ]
  }
}
```

- **`cachedContent: false`** — 必需。Antigravity 没有 `cachedContents` 资源；Gemini 默认（`true`）会导致 404。
- **`thoughtSignatureFallback: "skip"`** — 保持默认。当缺少工具调用的 thought signature 时，盖印 Google 的 `skip_thought_signature_validator` 哨兵，避免网关 400。仅在端点拒绝该哨兵时设为 `"none"`。

完整选项说明：[转换器 → gemini](/docs/server/config/transformers#options-cachedcontent-and-thoughtsignaturefallback)。另见 [CLI 认证命令](/docs/cli/commands/auth)。

### Qwen Chat

需要通过 `ccr qwen-auth` 完成 JWT 认证。

```json
{
  "name": "qwen",
  "baseUrl": "https://qwen.aikit.club/v1/chat/completions",
  "apiKey": "oauth_dummy_key",
  "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
  "transformer": {
    "use": ["qwen-auth", "reasoning", "OpenAI"]
  }
}
```

### Chrome 内置模型（Gemini Nano）

需要运行 `ccr chrome-bridge` 桥接进程。

```json
{
  "name": "chrome",
  "baseUrl": "http://127.0.0.1:9229",
  "apiKey": "dummy",
  "models": ["gemini-nano"],
  "transformer": {
    "use": ["chrome-on-device"]
  }
}
```

### Ollama（本地模型）

```json
{
  "name": "ollama",
  "api_base_url": "http://localhost:11434/v1/chat/completions",
  "api_key": "ollama",
  "models": ["qwen2.5-coder:latest"]
}
```

### 火山引擎

```json
{
  "name": "volcengine",
  "api_base_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
  "api_key": "your-api-key",
  "models": ["deepseek-v3-250324", "deepseek-r1-250528"],
  "transformer": {
    "use": ["deepseek"]
  }
}
```

### ModelScope

```json
{
  "name": "modelscope",
  "api_base_url": "https://api-inference.modelscope.cn/v1/chat/completions",
  "api_key": "",
  "models": [
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "Qwen/Qwen3-235B-A22B-Thinking-2507"
  ],
  "transformer": {
    "use": [
      ["maxtoken", { "max_tokens": 65536 }],
      "enhancetool"
    ],
    "Qwen/Qwen3-235B-A22B-Thinking-2507": {
      "use": ["reasoning"]
    }
  }
}
```

### DashScope（阿里云）

```json
{
  "name": "dashscope",
  "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
  "api_key": "your-api-key",
  "models": ["qwen3-coder-plus"],
  "transformer": {
    "use": [
      ["maxtoken", { "max_tokens": 65536 }],
      "enhancetool"
    ]
  }
}
```

### OpenCode Zen

`opencode-headers` 以 OpenCode CLI 的身份发送请求（会话、请求和项目请求头），并由它自己完成对 Zen 的上游调用。请在提供商级别、协议转换器之后使用它。

```json
{
  "name": "opencode",
  "api_base_url": "https://opencode.ai/zen/v1/responses",
  "api_key": "$OPENCODE_API_KEY",
  "models": ["muse-spark-1.3-contributor-free"],
  "transformer": {
    "use": ["openai-responses", "opencode-headers"]
  }
}
```

**会话与重试。** 每个对话都有一个持久化的 `x-opencode-session`（重启后不变）：客户端自带会话 ID 时（Claude Code metadata 或 `x-session-id` 等请求头）以它为键，否则以首条消息指纹为键。Zen 按该会话路由。瞬时失败（408/409/425/429/5xx、网络错误）使用同一会话重试；Zen 的确定性坏分桶错误（`401 No provider available`、`400 … Upstream request failed`）会重新生成会话。重试耗尽后，坏分桶错误以 `503` 返回，以便 CCR 的常规回退机制切换到其他模型。

**免费模型**（`opencode.ai/zen/` 端点上以 `-free` 结尾的模型）只有在请求看起来来自 OpenCode 客户端时才能通过 Zen 的免费额度检查。对这些模型，CCR 会：

- 上游始终使用流式请求；非流式客户端仍会收到由流汇总而成的 JSON 响应。
- 在客户端缺少时添加名为 `read` 和 `shell` 的函数工具。每个桩工具会复制客户端对应工具的 schema（`Read` / `read_file`、`Bash` / `pwsh` / `exec`，或 `run_code` 类工具），响应中对桩工具的调用会改回该客户端工具名。没有对应工具的桩会提示模型不要调用。
- 将 `prompt_cache_key` 设为 Zen 会话 ID（Responses 协议）；使用其他 key 时 Zen 会以 `response.incomplete` 停止。
- 当客户端启用了推理但未指定摘要级别时，请求 `reasoning.summary: "detailed"`。
- 对停滞的流执行故障转移：CCR 会暂存响应，直到 Zen 发送实际输出。若 30 秒内没有完整事件，或 60 秒内没有输出，则返回 `504`（可触发回退）。不输出摘要的推理项最多可静默 5 分钟。输出开始后，60 秒无数据（推理期间为 5 分钟）会以错误结束流。

付费 Zen 模型的请求保持不变。

## 提供商配置选项

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 提供商的唯一标识符 |
| `api_base_url` | string | 是 | API 基础 URL |
| `api_key` | string | 是 | API 认证密钥 |
| `models` | string[] | 否 | 可用模型列表 |
| `transformer` | object | 否 | 应用的转换器配置 |

## 模型选择

在路由中选择模型时，使用以下格式：

```
{provider-name},{model-name}
```

例如：

```
deepseek,deepseek-chat
codex,gpt-5
cursor,composer-2
claude-subscription,claude-sonnet-4-6
chrome-nano,gemini-nano
```

## FIM 提供商

FIM 使用**独立**端点（`POST /v1/fim/completions`）与专用 `fim.*` 转换器。建议在
聊天提供商旁再写一条 FIM 条目（可共用 API key）。Codestral 与本地 Qwen（LM Studio）
完整步骤见 [FIM Completions API](/docs/server/api/fim-completions-api)。

| 转换器 | 典型上游 | 说明 |
|--------|----------|------|
| `fim.mistral` | Codestral `/v1/fim/completions` | 入站为 mistral 时同族请求/响应透传 |
| `fim.qwen` | LM Studio 或 DashScope `/v1/completions` + FIM tokens | 跨族：响应编码回**入站**线格式（v1 → Codestral 形态） |
| `fim.deepseek` | DeepSeek `/beta/completions` | 跨族：同样编码回入站线格式 |

客户端响应跟随入站 kind，而不是永远固定为 Codestral 形态。

## 使用环境变量

您可以在配置中使用环境变量来保护 API 密钥：

```json
{
  "Providers": [
    {
      "name": "deepseek",
      "api_base_url": "https://api.deepseek.com/chat/completions",
      "api_key": "$DEEPSEEK_API_KEY",
      "models": ["deepseek-chat"]
    }
  ]
}
```

支持 `$VAR_NAME` 和 `${VAR_NAME}` 两种语法。

## 转换器配置

转换器用于适配不同提供商的 API 差异。您可以在提供商级别或模型级别配置转换器：

### 提供商级别转换器

应用于提供商的所有模型：

```json
{
  "name": "openrouter",
  "transformer": {
    "use": ["openrouter"]
  }
}
```

### 模型级别转换器

应用于特定模型：

```json
{
  "name": "deepseek",
  "transformer": {
    "use": ["deepseek"],
    "deepseek-chat": {
      "use": ["tooluse"]
    }
  }
}
```

## 下一步

- [路由配置](/docs/server/config/routing) - 配置请求如何路由
- [转换器](/docs/server/config/transformers) - 对请求应用转换
