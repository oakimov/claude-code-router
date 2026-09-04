---
title: API 概览
---

# API 概览

Claude Code Router Server 提供完整的 HTTP API，支持：

- **Messages API**：Anthropic Claude Messages（`POST /v1/messages`）
- **Chat Completions API**：OpenAI Chat Completions（`POST /v1/chat/completions`）
- **Responses API**：OpenAI Responses（`POST /v1/responses`）
- **FIM Completions API**：Fill-in-the-middle（`POST /v1/fim/completions`；独立流水线）
- **Models API**：OpenAI 兼容的可路由模型列表
- **配置 API**：读取和更新服务器配置
- **日志 API**：查看和管理服务日志
- **工具 API**：计算 Token 数量

以上四个 LLM POST 协议均为一等入站路由。聊天协议共享 Unified 聊天流水线；
FIM 使用独立流水线。响应始终匹配入站客户端协议。

## 基础信息

**Base URL**: `http://localhost:3456`

**认证方式**: API Key（`Authorization: Bearer` 或 `x-api-key`）

```bash
curl -H "x-api-key: your-api-key" http://localhost:3456/api/config
```

## API 端点列表

### 消息 / 推理

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/messages` | POST | 发送消息（兼容 Anthropic API） |
| `/v1/messages/count_tokens` | POST | 计算消息的 Token 数量 |
| `/v1/chat/completions` | POST | OpenAI Chat Completions（别名：`/chat/completions`） |
| `/v1/responses` | POST | OpenAI Responses（别名：`/responses`） |
| `/v1/fim/completions` | POST | FIM Completions（别名：`/fim/completions`） |
| `/v1/models` | GET | 列出可路由模型（别名：`/models`） |
| `/v1/models/{id}` | GET | 单个模型信息 |

### 配置管理

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/config` | GET | 获取当前配置 |
| `/api/config` | POST | 更新配置 |
| `/api/transformers` | GET | 获取可用的转换器列表 |

### 日志管理

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/logs/files` | GET | 获取日志文件列表 |
| `/api/logs` | GET | 获取日志内容 |
| `/api/logs` | DELETE | 清除日志 |

### 服务管理

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/restart` | POST | 重启服务 |
| `/ui` | GET | Web 管理界面 |
| `/ui/` | GET | Web 管理界面（重定向） |

## 认证

### API Key 认证

在请求头中添加 API Key：

```bash
curl -X POST http://localhost:3456/v1/messages \
  -H "x-api-key: your-api-key" \
  -H "content-type: application/json" \
  -d '...'
```

## 流式响应

Messages、Chat Completions、Responses 与 FIM 在 `stream: true` 时支持
Server-Sent Events。每条路由按**入站**客户端协议输出：

- Messages → Anthropic SSE（`message_start` / `content_block_delta` / …）
- Chat Completions → `chat.completion.chunk`，以 `[DONE]` 结束
- Responses → 有序 `response.*` 事件，以 `response.completed` / `response.failed` 结束
- FIM → 入站 FIM 线格式（v1：Codestral/Mistral 形态分片）

```bash
curl -X POST http://localhost:3456/v1/messages \
  -H "x-api-key: your-api-key" \
  -H "content-type: application/json" \
  -d '{"stream": true, ...}'
```

Anthropic Messages 流示例：

```
event: message_start
data: {"type":"message_start","message":{...}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}

event: message_stop
data: {"type":"message_stop"}
```
