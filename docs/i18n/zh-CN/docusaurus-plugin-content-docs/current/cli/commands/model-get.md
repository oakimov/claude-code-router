---
sidebar_position: 3
---

# ccr model get

非交互式发现提供商的可用模型。

## 用法

```bash
ccr model get <provider名称>
```

## 说明

从指定提供商的 API 获取可用模型列表，使用可配置的路径解析响应，并将新模型追加到本地配置中。现有模型会被保留 — 仅添加缺失的模型。

提供商必须已在 `config.json` 中配置，至少包含 `baseUrl` 和 `apiKey`。

## 示例

### 发现 OpenAI 模型

```bash
ccr model get openai
```

### 发现 DeepSeek 模型

```bash
ccr model get deepseek
```

### 发现 Groq 模型

```bash
ccr model get groq
```

### 发现 Cursor 模型

当提供商名为 `cursor` 或 `transformer.use` 包含 `cursor-sdk` 时，会识别为 Cursor 提供商。模型通过 `@cursor/sdk`（`Cursor.models.list`）列出，而不是 REST `/models`。

```bash
ccr model get cursor
```

认证使用以 `crsr_` 开头的 `api_key`，或 `CURSOR_API_KEY`。详见 [Cursor SDK 集成指南](/docs/server/guides/cursor)。
