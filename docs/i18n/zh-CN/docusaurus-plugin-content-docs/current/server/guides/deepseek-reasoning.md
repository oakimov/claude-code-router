---
sidebar_position: 4
---

# DeepSeek 推理重放

某些基于 DeepSeek 的提供商（如通过 OpenCode、ZenGo）要求在后续请求中包含之前助手的**推理内容**。这是因为 DeepSeek 将其输出分为可见内容和内部推理（思维链），在多轮对话中必须将推理内容回传给模型以保持连贯性。

`reasoning` 转换器自动处理这一过程：捕获响应中的推理输出，并在下一次请求中重放。

## 工作原理

1. 在每个响应中，`reasoning` 转换器拦截 `reasoning_content` 字段（例如 DeepSeek 的 `<reasoning_content>...</reasoning_content>` 块）
2. 它将推理文本存储在对话上下文中
3. 在下一个请求中，它将存储的推理作为 `reasoning` 块注入消息数组
4. 确保模型能接收到自己的先前推理作为上下文

## 配置

`reasoning` 转换器通常在模型级别配置：

```json
{
  "Providers": [
    {
      "name": "opencode",
      "baseUrl": "https://api.opencode.com/v1/chat/completions",
      "apiKey": "$OPENCODE_API_KEY",
      "models": ["deepseek-v4"],
      "transformer": {
        "use": ["opencode-headers"],
        "deepseek-v4": {
          "use": ["reasoning"]
        }
      }
    }
  ]
}
```

## 需要推理重放的提供商

| 提供商 | 需要推理重放 | 说明 |
|--------|-------------|------|
| DeepSeek（官方） | ✅ | 必须在多轮中重放 `reasoning_content` |
| OpenCode | ✅ | 使用 DeepSeek 模型 |
| ZenGo | ✅ | 基于 DeepSeek |
| 其他 OpenAI 兼容 | ❌ | 标准 Chat Completions，无 reasoning 字段 |

## 与其他转换器组合

`reasoning` 转换器应放在**其他请求修改器之后**。`transformer.use` 中的顺序很重要：

```json
{
  "use": ["opencode-headers", "reasoning"]
}
```

推理重放需要在最终的消息列表中正确注入推理块。

## 故障排除

**模型在多轮对话中丢失上下文**：如果模型重复自己或丢失上下文，reasoning 转换器可能未激活。检查它是否在模型级别的 `use` 数组中。

**推理内容对用户可见**：转换器仅在内部重放推理 — 不应出现在 Claude Code 的输出中。如果出现，请检查 `transformResponseOut` 逻辑。
