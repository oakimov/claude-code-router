---
title: Responses API
---

# Responses API

一等入站 OpenAI Responses 协议（`openai_responses`）。协议所有者：
`openai-responses`。Messages、Chat Completions 与 FIM 为独立入站路由 — 见
[API 概览](/docs/server/api/overview)。

CCR 接受 OpenAI Responses 请求于：

- `POST /v1/responses`
- `POST /responses`

配置 OpenAI SDK：

```shell
export OPENAI_BASE_URL=http://127.0.0.1:3456/v1
export OPENAI_API_KEY=your-router-api-key
```

该路由支持字符串与消息数组 `input`、instructions、URL/data URL 文本与图片、
function 调用/结果、function 工具、reasoning effort/summary 意图、token 限制、
usage、JSON 响应，以及有序 `response.*` SSE 事件。Function-call ID 会规范化到
Responses 的长度与字符限制，并在同一轮内保持关联。

CCR **不**持有 OpenAI 对话状态。会拒绝 `store: true`、`previous_response_id`、
`conversation` 与后台执行。提供商文件 ID 也会被拒绝。每个入站 Responses 工具 —
`function`、`custom`（Codex MCP / 插件工具）、`web_search` 及其他托管类型 —
都会投影为 Unified Chat Completions 的 `function` 工具，使 Chat Completions
后端看不到 Responses 专用托管工具变体。Responses `custom` 工具接受自由文本而非
JSON 参数，因此 CCR 在与 Chat Completions 提供商通信时用合成的必填字符串参数
承载输入，再将客户端响应还原为 `custom_tool_call` / `custom_tool_call_output`。
客户端因此看到原始工具协议，而模型调用的是 Unified function 表示。流式成功以
`response.completed` 结束；失败发出 `response.failed`。Chat Completions 的
`[DONE]` 标记不属于客户端 Responses 流。
