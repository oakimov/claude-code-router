---
title: Models API
---

# Models API

CCR 暴露 OpenAI 兼容的模型列表，供 SDK 客户端与工具发现路由器可达的模型：

- `GET /v1/models`
- `GET /models`
- `GET /v1/models/{id}`

## 模型 ID

`MODEL_ID_OUTPUT` 控制 API 返回的表示：

- `"literal"`（默认）发出 CCR 原生 `provider,model` ID。
- `"masked"` 将不以 `claude` 或 `anthropic` 开头的 ID 发为
  `claude-<小写 UTF-8 hex>`，以免被 Claude 客户端过滤。

两种形式均可在任一模式下发往 `/v1/responses`、`/v1/chat/completions` 或
`/v1/messages`。例如以下二者解析相同：

```text
codex,gpt-5.6-sol
claude-636f6465782c6770742d352e362d736f6c
```

```bash
curl -H "x-api-key: your-router-api-key" http://localhost:3456/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "deepseek,deepseek-chat",
      "object": "model",
      "created": 0,
      "owned_by": "deepseek"
    }
  ]
}
```

`owned_by` 是 CCR 提供商名。`created` 恒为 `0` — CCR 没有逐模型创建时间，
且按启动时间赋值会使重启前后本应相同的响应发生变化。

## 单个模型

```bash
curl -H "x-api-key: your-router-api-key" \
  "http://localhost:3456/v1/models/deepseek,deepseek-chat"
```

id 含 `/` 时需 URL 编码。例如
`openrouter,anthropic/claude-3.5-sonnet` 在路径中变为
`openrouter,anthropic%2Fclaude-3.5-sonnet`。

未知 id 返回 `404` 与 OpenAI 形态错误（`code: "model_not_found"`）。
单模型查询接受任一表示，并按 `MODEL_ID_OUTPUT` 返回所选 ID 格式。

## 行为

列表来自运行中服务器启动时的 `Providers` 条目 — 每个提供商/模型一对一条，
去重。无 `models` 数组的提供商会被跳过；无提供商的配置返回空 `data` 而非错误。

因列表反映运行中服务器，编辑 `config.json` 后请重启：

```bash
ccr restart
```

## 认证

列表受与其余面相同的 API key 检查保护。配置了 `APIKEY` 时发送
`Authorization: Bearer <key>` 或 `x-api-key: <key>`。

## Codex

Codex **不会**从此端点填充模型选择器 — 它在启动时读取本地目录文件。使用
[`ccr codex-config`](/docs/cli/commands/codex-config) 生成该目录并将 Codex
指向 CCR。
