---
sidebar_position: 1
---

# Codex（ChatGPT）集成

Claude Code Router 可以使用 **ChatGPT/Codex 订阅**，将 Claude Code 请求路由到 OpenAI 的模型。Codex 后端为 OpenAI 的 ChatGPT 产品提供支持，本集成让你能在 Claude Code 中使用该订阅。

Codex 支持 **两种认证模式**：

- **OAuth**，通过 `ccr codex-auth` — 当你希望 CCR 为你管理 OpenAI token 时推荐使用
- **PAT**（Personal Access Token），通过 `api_key: "at-..."` — 当你已有 Codex 兼容的 PAT 时推荐使用

仍需要 ChatGPT Plus 或 Pro 订阅。

## 认证模式

### 通过 `ccr codex-auth` 使用 OAuth

1. `ccr codex-auth` 会打印授权 URL，并在端口 1455 上启动本地回调服务器
2. 你在浏览器中打开该 URL，登录 OpenAI / ChatGPT 账户
3. OpenAI 重定向到 `http://localhost:1455/auth/callback`，CCR 服务器在此用授权码换取 token（PKCE 流程）
4. Token 保存到 `~/.claude-code-router/codex_auth.json`
5. 返回终端并按 Enter — CLI 确认 token 已保存
6. `codex` 转换器读取 access token，并用它认证 API 请求
7. CLI 与服务器会在过期前五分钟独立刷新 token

ID token 提供所选的 `chatgpt_account_id` 与 FedRAMP 状态。
CCR 会在推理与模型发现请求中发送这些路由头。Token
更新是原子的，并通过文件系统锁协调，因为 CLI 与
服务器是独立进程。若推理请求收到 401，
服务器会重新加载或刷新同一 OAuth 账户并重试一次。

### 通过 `api_key` 使用 PAT

若提供商 `api_key` 以 `at-` 开头，Codex 转换器会将其视为 Personal Access Token，而不使用 OAuth token。

1. 你将 PAT 直接放在提供商的 `api_key` 字段中
2. 在用于后端之前，CCR 会调用 OpenAI 的 whoami 端点，解析账户、
   用户、套餐与 FedRAMP 元数据
3. 服务器会去重并发查询，并短暂缓存结果
4. 运行时与模型发现请求会发送 PAT，以及已解析的账户
   与 FedRAMP 头

若 `api_key` 缺失、只是占位符，或不是以 `at-` 开头，
CCR 会选择来自
`~/.claude-code-router/codex_auth.json` 的 OAuth token 流程。`at-` 值始终保持在 PAT
模式：已吊销或无效的 PAT 会直接失败，不会静默改用 OAuth。

## 前置要求

- [ChatGPT Plus 或 Pro](https://chat.openai.com) 订阅
- Claude Code Router 正在运行（Docker Compose 或本地）

## 设置步骤

### 方案 A：OAuth 设置

#### 1. 认证

运行 OAuth 流程：

```bash
ccr codex-auth
```

CLI 会打印授权 URL。在浏览器中打开它，使用 OpenAI / ChatGPT 账户登录并授权应用。浏览器显示 “Authentication Successful” 后，返回终端并按 Enter。Token 会自动保存。

#### 2. 配置提供商

将 Codex 提供商添加到 `~/.claude-code-router/config.json`：

```json
{
  "Providers": [
    {
      "name": "codex",
      "api_base_url": "https://chatgpt.com/backend-api/codex",
      "api_key": "oauth_dummy_key",
      "models": ["gpt-5", "gpt-5-high", "gpt-5-mini"],
      "transformer": {
        "use": ["openai-responses", "codex"]
      }
    }
  ],
  "Router": {
    "default": "codex,gpt-5"
  }
}
```

### 方案 B：PAT 设置

若你已有 Codex 兼容的 PAT，可以跳过 `ccr codex-auth`，直接把 token 放进 `api_key`。

```json
{
  "Providers": [
    {
      "name": "codex",
      "api_base_url": "https://chatgpt.com/backend-api/codex",
      "api_key": "at-your-personal-access-token",
      "models": ["gpt-5", "gpt-5-high", "gpt-5-mini"],
      "transformer": {
        "use": ["openai-responses", "codex"]
      }
    }
  ],
  "Router": {
    "default": "codex,gpt-5"
  }
}
```

PAT 检测刻意保持简单：若 `api_key` 以 `at-` 开头，CCR 使用 PAT 认证；否则回退到 OAuth。

### 最后一步：重启

```bash
docker compose restart ccr
```

## 认证回退顺序

Codex 转换器按以下顺序选择认证：

1. 若 `api_key` 以 `at-` 开头 → 使用 PAT 认证
2. 否则 → 使用 `~/.claude-code-router/codex_auth.json` 中的 OAuth token
3. 若两者都不可用 → 认证失败

需要基于浏览器的登录与自动 token 刷新时使用 OAuth。希望在提供商配置中使用明确静态凭据时使用 PAT。

## Docker 运行

OAuth 回调使用端口 `1455`，该端口在 `docker-compose.yml` 中映射到 CCR 服务器端口（`"1455:3456"`）。在 Docker 中运行并使用 OAuth 时：

```bash
docker exec -it claude-code-router ccr codex-auth
```

CLI 会打印一个可在主机浏览器中打开的 URL。登录后，浏览器重定向到 `http://localhost:1455/auth/callback`，Docker 会将其转发到容器。Token 通过挂载的 `./ccr-config` 目录在容器重启后仍然保留。

PAT 认证不需要浏览器流程，但仍使用容器内相同的提供商配置。

## 提供商配置说明

- 在 `config.json` 中使用 `api_base_url`，而不是 `baseUrl`
- 在 `config.json` 中使用 `api_key`，而不是 `apiKey`
- `api_key` 的值可以是以下之一：
  - `oauth_dummy_key`（或其他占位符），用于 OAuth 模式
  - 以 `at-` 开头的真实 PAT，用于 PAT 模式
- 两种模式都仍使用 `codex` 转换器
- `ccr model get codex` 适用于任一认证模式
- 模型发现会发送当前的 Codex CLI `client_version`，因为 ChatGPT 后端可能按客户端版本对最新发布的 Codex 模型 slug 进行限制。CCR 默认使用发布时已知的最新稳定版本；可在提供商上通过 `codex_client_version` 覆盖，或在测试更新版本 Codex CLI 时通过 `CCR_CODEX_CLIENT_VERSION` 覆盖。运行时 Codex 请求由核心 Codex 转换器单独处理，它会按 Codex CLI 的方式模拟请求版本和身份头，而不依赖 CCR 的 CLI 包。

## 转换器行为

`codex` 转换器：

- 将统一请求转换为 ChatGPT 后端格式
- 使用 OAuth token 或 PAT 进行认证
- 自动解析并发送 `ChatGPT-Account-ID`
- 在认证账户要求时添加 `X-OpenAI-Fedramp: true`
- 将流式 Responses 风格事件转换回 Claude Code 兼容输出

## 何时使用 `ccr codex-auth`

在以下情况运行 `ccr codex-auth`：

- 你希望使用 OAuth 而不是 PAT
- OAuth token 已过期或被吊销
- 你从配置中移除了 PAT，并希望再次回退到 OAuth

当 `api_key` 已包含以 `at-` 开头的有效 PAT 时，你**不**需要 `ccr codex-auth`。

## 功能特性

- **SSE 流式传输** — 完整支持实时响应流
- **推理/思考内容** — 支持具备推理能力的模型
- **工具调用** — 支持多工具的 function calling
- **网页搜索** — 通过 `{ type: "web_search" }` 内置网页搜索
- **图像处理** — 支持图像输入的视觉能力

## 使用示例

将 Codex 用作默认模型，或为特定场景路由：

```json
{
  "Router": {
    "default": "codex,gpt-5",
    "webSearch": "codex,gpt-5-high",
    "think": "codex,gpt-5-high",
    "background": "codex,gpt-5-mini"
  }
}
```

## 模型参考

| 模型 | 说明 |
|------|------|
| `gpt-5` | 标准 GPT-5 模型 |
| `gpt-5-high` | 高性能变体（推理任务） |
| `gpt-5-mini` | 轻量变体（后台任务） |

## 故障排除

**OAuth token 过期或无效**：重新运行 `ccr codex-auth` 以刷新 token。

**PAT 被拒绝**：确认 `api_key` 包含完整 PAT，且以 `at-` 开头。

**找不到提供商**：确认配置中的提供商名称与 `body.model` 匹配（例如 `codex,gpt-5`）。

**配置字段错误**：在 `config.json` 中使用 `api_base_url` 和 `api_key`，而不是 `baseUrl` / `apiKey`。

**意外回退到 OAuth**：若 PAT 模式未激活，请确认修剪空白后的 `api_key` 以 `at-` 开头。

**没有任何可用认证**：在 `api_key` 中配置 PAT，或通过 `ccr codex-auth` 配置 OAuth token。

## 相关文档

- [CLI 认证命令](/docs/cli/commands/auth)
- [Claude 订阅指南](/docs/server/guides/claude-auth)
- [提供商配置](/docs/server/config/providers)
- [转换器配置](/docs/server/config/transformers)
