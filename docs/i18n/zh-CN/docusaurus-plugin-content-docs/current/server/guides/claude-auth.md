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

需要两个转换器，且顺序固定：

- `claude-auth` — 加载/刷新 OAuth access token，对调用方客户端做身份分类，并构建下文所述的身份/账单请求头。请求体与 URL 的构建交给 `Anthropic` 完成。
- `Anthropic` — 注册客户端路由 `POST /v1/messages`，构建实际的 Anthropic Messages wire 请求体（`transformRequestIn`），并将 SSE/JSON 响应转换回 Unified 格式（`transformResponseOut`）。它会检测同一提供商链中前面是否存在 `claude-auth`，如果存在则跳过设置自己的 `Authorization`/`x-api-key` 请求头，确保不会覆盖 `claude-auth` 注入的 Bearer 令牌。

请求可以来自 CCR 支持的任意入站协议 —— Anthropic Messages（`/v1/messages`）、OpenAI Chat Completions（`/v1/chat/completions`）或 OpenAI Responses（`/v1/responses`）。所有入站请求在路由前都会被归一化为内部 Unified 格式，因此非 Anthropic 形态的客户端（例如某个 OpenAI 形态的工具）同样可以被路由到 `claude-auth` 提供商，处理方式与 Anthropic 形态的客户端完全一致 —— 只是会被视为「非 Claude Code」客户端（见下文）。

### 客户端分类

`claude-auth` 只依据一个判断来决定如何构建出站请求：入站的 `User-Agent` 是否以 `claude-cli/` 开头（`claude-auth.transformer.ts` 中的 `isClaudeCodeClient()`）。据此分为两条分支：

- **真实的 Claude Code 客户端** —— 请求本身已经具备真实 Claude Code 流量的形态。`claude-auth` 会原样转发客户端自己的身份请求头、system 块和 `anthropic-beta` 值（只合并 OAuth 所需的 beta），因为重新推导这些值反而有偏离 Claude Code 真实发送内容的风险。
- **其他任意客户端**（OpenAI SDK、Anthropic SDK、自定义工具、Cursor 等）—— 请求不具备 Claude Code 的形态，因此 `claude-auth` 会合成 Claude Code 对等请求会发送的账单块、系统身份、身份请求头和 `anthropic-beta` 值，使 Anthropic 无论 CCR 背后代理的是哪个客户端，都能看到同样形态的流量。

### 出站请求头

| 请求头 | 真实 Claude Code 客户端 | 其他客户端 |
|---|---|---|
| `Authorization` | `Bearer <access_token>`，来自 `claude_auth.json`；过期时自动刷新 | 相同 |
| `Content-Type` | `application/json`（由 `Anthropic` 设置） | 相同 |
| `anthropic-version` | `2023-06-01`（由 `Anthropic` 设置） | 相同 |
| `anthropic-beta` | 客户端自身的值，合并 `oauth-2025-04-20` | 依据模型能力目录合成 —— 见下文 |
| `User-Agent` | 原样转发 | `ANTHROPIC_USER_AGENT` 环境变量覆盖，否则为 `claude-cli/${CC_VERSION} (external, cli)` |
| `x-app` | 原样转发 | `cli` |
| `x-claude-code-session-id` | 原样转发 | 合成的 UUID，每个进程缓存一次 |
| `x-client-request-id` | 原样转发 | 合成的 UUID，每个请求都不同 |
| `anthropic-dangerous-direct-browser-access` | 原样转发 | `true` |
| `x-stainless-arch` / `-lang` / `-os` / `-package-version` / `-retry-count` / `-runtime` / `-runtime-version` / `-timeout` | 原样转发 | 依据当前进程（架构/操作系统/Node 版本）及固定的 Anthropic SDK 包版本合成 |

逐跳头部（`connection`、`host`、`accept-encoding`、`content-length`）不会被转发。账单标记（`x-anthropic-billing-header`）**不是** HTTP 请求头 —— 见[账单与身份 system 块](#账单与身份-system-块)。

#### `anthropic-beta` 请求头逻辑

`oauth-2025-04-20` 始终包含 —— Anthropic 要求订阅 OAuth Bearer 认证使用该 beta。

**真实 Claude Code 客户端**：客户端自身的 `anthropic-beta` token 原样保留，若其中不含 `oauth-2025-04-20` 则追加（大小写不敏感去重）。不增删任何其他 token。

**其他客户端**：该值依据[模型能力目录](#模型能力目录)构建，模拟 Claude Code 对该模型实际发送的内容：

- `claude-code-20250219` 与 `oauth-2025-04-20` —— 始终包含
- `context-1m-2025-08-07` —— 仅当请求的模型 id 携带 `[1m]` 后缀时（见 [1M 上下文](#1m-上下文)）
- `interleaved-thinking-2025-05-14`、`thinking-token-count-2026-05-13` —— 仅当目录标记该模型支持扩展思考
- `context-management-2025-06-27` —— 仅当模型具备 `context_management` 能力
- `prompt-caching-scope-2026-01-05` —— 始终包含
- `mid-conversation-system-2026-04-07` —— 仅当模型具备 `mid_conv_system` 能力
- `advanced-tool-use-2025-11-20` —— 始终包含
- `effort-2025-11-24` —— 仅当模型具备 `effort` 能力（这是一份显式的按模型排除名单，而非按名称前缀匹配 —— 例如 `claude-sonnet-4-5`、`claude-haiku-4-5` 被排除，但 `claude-sonnet-4-6` 包含在内）
- `fallback-credit-2026-06-01` —— 仅当模型具备 `fast_mode` 能力

设置 `ANTHROPIC_BETA_FLAGS` 会整体替换该合成列表（而非合并）。两个分支的 URL 都会附加 `?beta=true`。以上均无需手动配置 —— 全部自动生效。

### 账单与身份 system 块

对于**其他客户端**，`claude-auth` 还会在 Anthropic `system` 数组最前面（调用方自带的 system 文本之前）插入两个条目，与 Claude Code 自身发送的内容一致：

1. 账单标记文本块：`x-anthropic-billing-header: cc_version=${CC_VERSION}.${suffix}; cc_entrypoint=${CC_ENTRYPOINT}; cch=${sessionCch};` —— 尽管名字里带 "header"，它实际是以 `system[0]` 文本形式传输，**不是** HTTP 请求头。`suffix` 是依据第一条用户消息文本与 CLI 版本推导出的 3 位十六进制摘要；`cch` 是每个进程仅生成一次的随机 5 位十六进制值（并非依据请求内容推导）。二者均不带 `cache_control`。
2. 身份文本块：`You are Claude Code, Anthropic's official CLI for Claude.`（`system[1]`），若调用方自己的第一条 system 条目以该字符串开头（但不完全相同），会保留其原有的 `cache_control`。

对于**真实的 Claude Code 客户端**，其自身的 system 块 —— 包括自带的账单标记和身份字符串 —— 会被原样转发；`claude-auth` 不会触碰、移除或重新排列它们。

### 模型能力目录

`claude-model-catalog.ts` 维护一份按模型划分的能力表（上下文窗口、是否原生支持 1M、最大输出 token、默认 effort，以及 `capabilities` 列表，如 `effort`、`context_management`、`mid_conv_system`、`fast_mode`、`adaptive_thinking`），驱动上文的 beta 合成逻辑，同时驱动构建后的调整流程（`applyClaudeModelCapabilityAdjustments`）：从 `thinking`/`output_config` 中剥离该模型不支持的 `effort` 字段、调整 `thinking` 块形态（`adaptive` 还是 `enabled`），以及将 `max_tokens` 限制在该模型已知的上限内。这套目录用一次表查询取代了原先分散的按模型条件判断，查询前会先归一化模型 id（剥离 CCR 的 `provider,` 前缀、`[1m]` 标记，以及 Anthropic 的 `-YYYYMMDD` 日期后缀）。

### 1M 上下文

Claude Code 只有在请求的模型 id 携带 `[1m]` 标记时，才会从 wire `model` 字段中剥离该标记并添加 `context-1m-2025-08-07` beta；原生支持 1M 的模型无需该 beta 即可获得更大的窗口。CCR 对两条客户端分支都遵循相同的规则 —— 该标记永远不会被用来拒绝、降级或改路由请求。

### Prompt 缓存：精确透传 vs 归一化

当请求通过同协议、同目标的路径（Anthropic 精确透传）到达 `Anthropic` 时，调用方自身的 `cache_control` 位置会按原样保留 —— `context.protocolContext.anthropicCacheMode === "preserve"` 会跳过自动缓存重写。跨协议投影的请求（例如某个 OpenAI 形态的请求被路由到 Anthropic 目标）则会运行 `applyRawAnthropicPromptCaching` 来插入合理的缓存断点，因为源协议本身不带 `cache_control` 概念。这只影响缓存标记的*位置*，与上文的客户端分类/请求头逻辑无关。

### 认证恢复

`claude-auth` 返回一个 `__authRecovery` 钩子，在收到 401 时运行：它会重新加载 `claude_auth.json`，以应对另一个进程（例如并发运行的 `ccr claude-auth` 重新登录）已经在外部轮转了令牌的情况；只有在未检测到外部轮转时才会刷新并保存。它绝不会退化为使用未认证的请求重试。

### 使用量对齐，而非削减使用量

CCR 的目标是让请求**与真实 Claude Code 流量无法区分**，而不是最小化账号使用量。具体而言：

- 正常 Messages 路径不会添加 `/count_tokens` 预检 —— Claude Code 本身也不做预检。
- 不施加本地 200,000 token 上限。如果账号允许超额使用且客户端发送了更大的请求，CCR 会原样发送；如果 Anthropic 拒绝该请求，上游错误语义会原样保留。
- `[1m]` 及原生 1M 模型的行为与上文描述完全一致 —— 从不为节省用量而被抑制。
- `Router.longContext` 仍是独立的运维路由功能；本集成不会强制要求 API Key 长上下文通道，也不会改写该路由选择。
- `anthropic-ratelimit-unified-overage-in-use` 等响应头会在转换过程中保留，并以 debug 级别记录为用量可观测性信息 —— 不会被当作告警条件，因为同样的直接 Claude Code 请求本来也会使用超额额度。

### 环境变量覆盖

| 变量 | 作用 |
|---|---|
| `ANTHROPIC_CLI_VERSION` | 覆盖账单标记与合成 `User-Agent` 中使用的 `CC_VERSION`（默认 `2.1.220`） |
| `CLAUDE_CODE_ENTRYPOINT` | 覆盖账单标记与合成 `User-Agent` 中的 `cc_entrypoint` 值（默认 `cli`） |
| `ANTHROPIC_USER_AGENT` | 直接覆盖合成的 `User-Agent` 请求头（仅作用于非 Claude Code 分支；真实 Claude Code 客户端自身的 `User-Agent` 始终原样转发） |
| `ANTHROPIC_BETA_FLAGS` | 整体替换合成的 `anthropic-beta` 值（仅作用于非 Claude Code 分支） |

这些行为均无需任何 `claudeAuth.*` 配置项 —— 全部根据请求自动推导。

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
