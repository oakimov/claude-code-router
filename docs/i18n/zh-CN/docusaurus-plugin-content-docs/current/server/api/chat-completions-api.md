---
title: Chat Completions API
---

# Chat Completions API

一等入站 OpenAI Chat Completions 协议（`openai_chat_completions`）。协议所有者：
`OpenAI`。Responses、Messages 与 FIM 为独立入站路由 — 见
[API 概览](/docs/server/api/overview)。

CCR 接受 OpenAI Chat Completions 请求于：

- `POST /v1/chat/completions`
- `POST /chat/completions`

两条路径在 preset 命名空间下同样可用。认证使用
`Authorization: Bearer <APIKEY>` 或 `x-api-key: <APIKEY>`。

```bash
curl http://127.0.0.1:3456/v1/chat/completions \
  -H "Authorization: Bearer $CCR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai,gpt-5-mini",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

使用 `provider,model` 显式指定目标。裸模型经 `Router.default` 与场景规则路由；
若无法解析到提供商，CCR 返回 400。

兼容层支持单文本输出、system/developer/user/assistant/tool 消息、文本与
`image_url` 输入、function 工具与结果、采样字段、token 限制、JSON 回复与 SSE。
流式以 `data: [DONE]` 结束。不支持的状态、音频、结构化输出、多候选、logprobs
等语义返回 OpenAI 形态的 400 错误。
