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
