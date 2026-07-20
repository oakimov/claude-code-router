---
sidebar_position: 2
---

# Claude 订阅集成

Claude Code Router 可通过 OAuth 认证，使用你**现有的 Claude 订阅**（Pro 或 Max）转发请求。这样可以直接利用 Claude.ai 订阅，无需单独的 API Key。

## 工作原理

1. `ccr claude-auth` 生成 PKCE 挑战，并输出来自 `claude.ai` 的授权 URL
2. 在浏览器中打开该 URL，登录你的 Claude 账户
3. Claude 重定向到 `http://localhost:1455/callback`，由 CCR 服务器用授权码交换令牌
4. 令牌保存到 `~/.claude-code-router/claude_auth.json`
5. 回到终端并按 Enter — CLI 会确认令牌已保存
6. `claude-auth` 转换器读取 access token，并在每次请求中注入为 `Bearer` 令牌
7. 令牌即将过期时，会自动使用 refresh token 刷新

## 前置要求

- [Claude Pro 或 Max](https://claude.ai) 订阅
- Claude Code Router 正在运行（Docker Compose 或本地）

## 设置

### 1. 认证

运行 OAuth 流程：

```bash
ccr claude-auth
```

CLI 会打印授权 URL。在浏览器中打开、登录 Claude 账户并授权应用。浏览器显示 “Authentication Successful” 后，回到终端按 Enter。令牌会自动保存。

### 2. 配置提供商

将提供商添加到 `~/.claude-code-router/config.json`：

```json
{
  "Providers": [
    {
      "name": "claude-subscription",
      "api_base_url": "https://api.anthropic.com",
      "api_key": "no-key",
      "models": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
      "transformer": {
        "use": ["claude-auth", "Anthropic"]
      }
    }
  ],
  "Router": {
    "default": "claude-subscription,claude-sonnet-4-6"
  }
}
```

### 3. 重启

```bash
docker compose restart ccr
```

## 在 Docker 中运行

OAuth 回调使用端口 `1455`，该端口已在 `docker-compose.yml` 中映射到 CCR 服务器（`"1455:3456"`）。在 Docker 中运行时：

```bash
docker exec -it claude-code-router ccr claude-auth
```

CLI 会打印一个可在宿主机浏览器中打开的 URL。登录后，浏览器会重定向到 `http://localhost:1455/callback`，再由 Docker 转发到容器。令牌通过挂载的 `./ccr-config` 目录在容器重启后保留。

## 转换器链

需要两个转换器：

- `claude-auth` — 将请求从 Unified（OpenAI）格式转换为 Anthropic 格式，注入 `Authorization: Bearer <token>`（从 `~/.claude-code-router/claude_auth.json` 加载/刷新令牌），并将 Anthropic SSE 响应转换回 Unified 格式
- `Anthropic` — 注册 `POST /v1/messages` 路由；在提供商链中自身不做 body 转换，仅作为端点桩

### 出站请求头

`claude-auth` 转换器按以下规则构建出站请求头。"始终设置"表示无论客户端如何都会携带该头部；"条件设置"表示仅在客户端发送时才转发。

| 请求头 | 条件 | 值 |
|---|---|---|
| `Authorization` | 始终设置 | `Bearer <access_token>`，来自 `claude_auth.json`；过期时自动刷新 |
| `Content-Type` | 始终设置 | `application/json` |
| `anthropic-beta` | 始终设置 | 见下方 [Beta 请求头逻辑](#anthropic-beta-请求头逻辑) |
| `anthropic-version` | 始终设置 | 优先转发客户端值；若客户端未发送则回退为 `2023-06-01` |
| `User-Agent` | 始终设置 | 优先转发客户端值；若客户端未发送则回退为 `claude-cli/2.1.195 (external, cli)` |
| `x-app` | 条件设置 | 原样转发（例如 `cli`） |
| `x-claude-code-session-id` | 条件设置 | 原样转发；Anthropic 用于会话归因 |
| `anthropic-dangerous-direct-browser-access` | 条件设置 | 客户端设置时原样转发 |

逐跳头部（`connection`、`host`、`accept-encoding`、`content-length`）以及客户端发送的 SDK 内部头部（`x-stainless-*`）**不会**被转发。

#### `anthropic-beta` 请求头逻辑

`oauth-2025-04-20` 始终包含 — Anthropic 要求订阅 OAuth Bearer 认证使用该 beta。其余值取决于客户端类型：

**Claude Code / Anthropic 原生客户端**（客户端发送了 `anthropic-beta`）：  
客户端的 beta token 原样保留，若其中不含 `oauth-2025-04-20` 则追加。不增删任何其他 token。示例 — 客户端发送：

```
claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27,effort-2025-11-24
```

出站：

```
claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27,effort-2025-11-24,oauth-2025-04-20
```

**OpenAI 兼容客户端**（客户端未发送 `anthropic-beta`）：  
从重建后的 Anthropic 请求体推导功能 beta：
- 请求使用扩展思考 → 追加 `interleaved-thinking-2025-05-14,effort-2025-11-24,prompt-caching-scope-2026-01-05`
- 任意消息或工具携带 `cache_control` 块 → 追加上述同一组 beta
- 否则 → 仅发送 `oauth-2025-04-20`

URL 还会附加 `?beta=true`。以上均无需手动配置 — 只要提供商使用 `claude-auth` 即自动生效。

## 令牌存储

令牌保存在 `~/.claude-code-router/claude_auth.json`（权限 0600）：

```json
{
  "access_token": "sk-ant-oat01-...",
  "refresh_token": "...",
  "token_type": "Bearer",
  "scope": "user:profile user:inference user:sessions:claude_code user:mcp_servers",
  "expires_at": 1760000000,
  "last_refresh": 1759996400
}
```

## 故障排查

**令牌过期或无效**：重新运行 `ccr claude-auth` 进行认证。

**"Redirect URI not supported"**：确保浏览器使用 `localhost`（而不是 `127.0.0.1`），且 CCR 服务器在端口 1455 上运行。

**找不到提供商**：确保配置中的提供商名称与模型字符串匹配（例如 `claude-subscription,claude-sonnet-4-6`）。
