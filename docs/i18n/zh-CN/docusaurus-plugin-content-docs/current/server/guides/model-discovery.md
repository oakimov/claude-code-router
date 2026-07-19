---
sidebar_position: 5
---

# 模型发现

Claude Code Router 可以从任何公开模型列表端点的 API 提供商**自动发现可用模型**。这对于模型列表频繁变化的提供商，或探索新提供商上可用的模型非常有用。

## 使用方法

使用提供商名称运行发现命令：

```bash
ccr model get <provider-name>
```

提供商必须在 `config.json` 中配置，至少包含 `baseUrl` 和 `apiKey`。

## 工作原理

1. `ccr model get` 向提供商的模型列表端点发送请求
2. 它使用可配置的路径解析 JSON 响应以提取模型 ID
3. 将发现的模型与当前配置进行比较
4. 追加尚未列出的新模型
5. 保留您的现有配置

## 配置

模型发现使用提供商配置中的约定：

```json
{
  "Providers": [
    {
      "name": "openai",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "$OPENAI_API_KEY",
      "models": ["gpt-4"]
    }
  ]
}
```

运行 `ccr model get openai` 将发现所有可用的 GPT 模型并追加。

## 支持的提供商

模型发现适用于任何公开 `GET /models` 端点并返回 JSON 数组或包含模型 ID 的对象的提供商。包括：

- OpenAI
- DeepSeek
- Groq
- OpenRouter
- 自定义具有 RESTful 模型 API 的提供商

**Cursor** 是特例：当提供商使用 `cursor-sdk`（或名为 `cursor`）时，`ccr model get` 会调用 `@cursor/sdk` 的 `Cursor.models.list`，而不是 HTTP。使用 `crsr_` 密钥或 `CURSOR_API_KEY` 认证。详见 [Cursor SDK 集成](/docs/server/guides/cursor)。

## 故障排除

**未找到模型**：提供商可能未公开模型列表端点，或响应格式与预期不同。请查看提供商的 API 文档以获取正确的模型列表路径。

**重复模型**：该工具仅追加配置中尚未存在的模型 — 重复项会自动跳过。
