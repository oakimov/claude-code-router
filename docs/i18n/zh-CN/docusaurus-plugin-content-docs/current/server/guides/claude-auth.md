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

请求可以来自 CCR 支持的任意**聊天**入站协议 —— Anthropic Messages（`/v1/messages`）、OpenAI Chat Completions（`/v1/chat/completions`）或 OpenAI Responses（`/v1/responses`）。聊天入站请求在路由前都会被归一化为内部 Unified 格式，因此非 Anthropic 形态的客户端（例如某个 OpenAI 形态的工具）同样可以被路由到 `claude-auth` 提供商，处理方式与 Anthropic 形态的客户端完全一致 —— 只是会被视为「非 Claude Code」客户端（见下文）。FIM（`/v1/fim/completions`）是独立流水线，不使用 `claude-auth`。

### 客户端分类

CCR 会在归一化之前根据完整指纹分类 Anthropic Messages 请求：

- **Claude Desktop** —— 识别两种原生传输：其一是 `anthropic-desktop-topbar: 1` 加 Anthropic JS SDK 请求头的 top-bar 形态；其二是当前 3P Agent SDK 形态，即完整的原生 CLI 请求头/请求体指纹，并且 UA 同时包含 `claude-desktop` 或 `claude-desktop-3p` 入口点以及 `agent-sdk/<version>`。Desktop 3P 自带固定版本的 Claude Code 与 Agent SDK，CCR 会保留这些版本，而不会替换为模拟配置版本。
- **Claude Code CLI** —— 要求 `claude-cli/<version>` UA、`x-app: cli`、session/Stainless 请求头和原生账单/身份 system 块，且不包含 Desktop Agent SDK 入口点。仅有 UA 不足以通过分类。
- **其他客户端** —— 包括不完整或未知指纹以及所有 OpenAI Chat/Responses 请求。只有该分支在路由到范围内的 Anthropic 提供商时才会生成固定的 Claude Code 模拟形态。

原生 Desktop 与 CLI 都使用 Anthropic 原始请求体/响应透传。CCR 只替换路由后的模型、上游 URL、提供商凭据和传输层管理的请求头。

### 出站请求头

| 请求头 | 原生 Desktop/CLI | 其他客户端 |
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

`oauth-2025-04-20` 仅在订阅 OAuth Bearer 认证分支包含 —— Anthropic 要求该认证使用此 beta。

**原生 Desktop/CLI**：客户端自身的 `anthropic-beta` token 原样保留；OAuth 配置会确保追加必需的 `oauth-2025-04-20`（大小写不敏感去重），API Key 配置不会生成 OAuth beta。不增删任何其他 token。

**其他客户端**：该值依据[模型能力目录](#模型能力目录)构建，模拟 Claude Code 对该模型实际发送的内容：

- `claude-code-20250219` —— 普通非 Haiku 配置；当前 CLI 的普通 Haiku 请求会省略它
- `oauth-2025-04-20` —— 仅 OAuth 配置
- `context-1m-2025-08-07` —— 仅当请求的模型 id 携带 `[1m]` 后缀时（见 [1M 上下文](#1m-上下文)）
- `interleaved-thinking-2025-05-14`、`thinking-token-count-2026-05-13` —— 仅当目录标记该模型支持扩展思考
- `context-management-2025-06-27` —— 仅当模型具备 `context_management` 能力
- `prompt-caching-scope-2026-01-05` —— 始终包含
- `mid-conversation-system-2026-04-07` —— 仅当模型具备 `mid_conv_system` 能力
- `advanced-tool-use-2025-11-20` —— 仅显式工具搜索请求；普通模拟配置不会无条件添加
- `effort-2025-11-24` 与 `fallback-credit-2026-06-01` —— 由请求/功能开关决定；模拟配置不会无条件添加

设置 `ANTHROPIC_BETAS` 会将逗号分隔的自定义 beta 追加到合成列表；OAuth 仍会确保必需的 `oauth-2025-04-20`。两个分支的 URL 都会附加 `?beta=true`。以上均无需手动配置 —— 全部自动生效。

Beta token 只会通过 `anthropic-beta` HTTP 请求头发送。CCR 不会在 Messages JSON 请求体中加入 `betas` 字段；该名称是 Anthropic SDK 的选项，SDK 会在发送请求前将其移除。

### 账单与身份 system 块

对于**路由到范围内 Anthropic 提供商的其他客户端**，CCR 会把 Anthropic `system` 数组精简为 Claude Code 2.1.280 配置的形态：

1. 账单标记文本块：对于一方 Anthropic 配置为 `x-anthropic-billing-header: cc_version=${CC_VERSION}.${suffix}; cc_entrypoint=unknown; cch=00000;` —— 尽管名字里带 "header"，它实际是以 `system[0]` 文本形式传输，**不是** HTTP 请求头。`suffix` 是依据实际发送的第一条用户消息文本与 CLI 版本推导出的 3 位十六进制摘要（经过下文的迁移后，该文本就是调用方被迁移的 system 提示词，因此客户端丢弃或摘要早期轮次时它保持不变）。当前 `2.1.280` 配置不再使用旧版随机 `cch` 行为。二者均不带 `cache_control`。
2. 身份文本块：`You are Claude Code, Anthropic's official CLI for Claude.`（`system[1]`）。模拟路径会将选定的缓存配置应用到这个可缓存块；调用方自带的 `cache_control` 不会覆盖固定的版本配置。

Anthropic 的 OAuth 计费校验会检查身份块之后的 `system[]` 内容，若其中带有外部 harness 的提示词，就以 "out of extra usage" 400 拒绝请求 —— 即使请求头正确，调用方自己的 system 提示词也会让流量明显不是 Claude Code。CCR 的第三方 Anthropic 模拟沿用 Claude Code OAuth 客户端（如 `opencode-claude-auth`）的做法：调用方在 `system[1]` 之后提供的全部内容会被迁移到第一条用户消息中（按原顺序，作为独立的文本块放在最前面），而不是留在 `system[]` 里。内容仍原样送达模型，只是成为第一条用户轮次的一部分。如果没有可附加的用户消息，则不做迁移 —— 调用方的 system 内容保留在 `system[]` 中，而不会被丢弃。

对于**其他客户端**，工具名也会在构建请求体之前改写为 Claude Code 的 OAuth 拼写：`bash` 变为 `mcp_Bash`，`read` 变为 `mcp_Read`，已是 `mcp_...` 的名称保持不变。工具定义、历史 assistant `tool_calls` 以及强制的 `tool_choice` 会保持一致。响应路径会通过请求级的名称映射，在 JSON 与流式响应中恢复调用方原始的工具名。

对于**原生 Desktop 与 CLI**，其自身的 system 块 —— 包括账单、身份及不透明字段 —— 会被原样转发，不做任何 system 提示词改写；唯一的缓存变化是下文“Prompt 缓存”一节所述的 OAuth TTL 延长。

### 模型能力目录

`claude-model-catalog.ts` 维护一份按模型划分的能力表（上下文窗口、是否原生支持 1M、最大输出 token、默认 effort，以及 `capabilities` 列表，如 `effort`、`context_management`、`mid_conv_system`、`fast_mode`、`adaptive_thinking`），驱动上文的 beta 合成逻辑，同时驱动构建后的调整流程（`applyClaudeModelCapabilityAdjustments`）：从 `thinking`/`output_config` 中剥离该模型不支持的 `effort` 字段、调整 `thinking` 块形态（`adaptive` 还是 `enabled`；`enabled` 总会带上 `budget_tokens`，取自客户端，或按请求的 effort 取 `max_tokens` 的一定比例，不低于 Anthropic 的最小值 1024 且小于 `max_tokens`，两者无法同时满足时省略 thinking），以及将 `max_tokens` 限制在该模型已知的上限内。这套目录用一次表查询取代了原先分散的按模型条件判断，查询前会先归一化模型 id（剥离 CCR 的 `provider,` 前缀、`[1m]` 标记，以及 Anthropic 的 `-YYYYMMDD` 日期后缀）。

### 1M 上下文

Claude Code 只有在请求的模型 id 携带 `[1m]` 标记时，才会从 wire `model` 字段中剥离该标记并添加 `context-1m-2025-08-07` beta；原生支持 1M 的模型无需该 beta 即可获得更大的窗口。CCR 对两条客户端分支都遵循相同的规则 —— 该标记永远不会被用来拒绝、降级或改路由请求。

### Prompt 缓存：原生透传 vs 模拟

原生 Claude Desktop 和 Claude Code CLI 请求的缓存标记位置会完整保留，包括按功能开关决定的 scope。CCR 不会为没有标记的原生请求添加标记，也不会移动已有标记。只有一处变化，且仅在 `claude-auth` OAuth 路由上：当请求中没有任何标记设置 TTL 时，每个 ephemeral 标记都会加上 `ttl: "1h"`。Claude Code 依据它向 CCR 认证的方式选择 TTL；使用 API Key（例如网关密钥）时，它对每个请求都使用默认的 5 分钟，因此超过五分钟的空闲就会重写整个对话缓存。任一标记已设置 TTL 的请求会原样转发。只检查真正可放置标记的位置（顶层、tools、system 块、消息内容块以及 `tool_result` 内容）；工具输入中的值属于用户数据，不会被改动。1 小时缓存写入的计费高于 5 分钟写入，见 [Claude Code prompt caching](https://code.claude.com/docs/en/prompt-caching#cache-lifetime)。如需保留客户端自己的 TTL，在 `config.json` 中设置 `"CLAUDE_AUTH_NATIVE_CACHE_TTL": "client"`（默认 `"1h"`）。直连的 `Anthropic` API Key 路由保持客户端标记不变。当前 Desktop 3P 对话通过 Desktop 内置的 Agent SDK 运行，因此可以像 Claude Code 一样在 system 和 message 内容上生成缓存断点。

只有“其他客户端”路径会生成缓存字段，使用 2.1.280 配置：账单块不加缓存标记，可缓存的 system 块使用该配置的缓存控制，被迁移的调用方 system 提示词获得 Claude Code 在其自身 system 提示词上设置的断点（这样客户端改写或截取历史时，工具和指令仍保持缓存），并在一个消息尾部设置最终断点。没有可迁移的用户消息时，只标记第一个和最后一个可缓存的 system 块，使请求不超过 Anthropic 的四个断点上限；请求体构建器不会发送的部分（空文本、没有 URL 的图片、没有数据的文件）不会承载尾部断点；若最后一个轮次完全没有可发送内容（构建器会省略它），尾部断点移到前一个符合条件的消息。尾部是工具结果时，断点位于 `tool_result` 块上，与 Claude Code 一致。CCR 没有通用的缓存归一化器；其他目标以及其他客户端协议都不会套用 Claude Code 的 system 或缓存改写。

早期版本把被迁移的提示词拼接到第一条用户消息的文本前面，而不是作为独立的块；账单后缀也取自迁移前的第一条用户消息。因此，升级 CCR 时正在进行的对话会有一次 prompt 缓存未命中，之后的轮次会重新命中缓存。

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
| `ANTHROPIC_CLI_VERSION` | 覆盖账单标记与合成 `User-Agent` 中使用的 `CC_VERSION`（默认 `2.1.280`） |
| `CLAUDE_CODE_ENTRYPOINT` | 覆盖账单标记与合成 `User-Agent` 中的 `cc_entrypoint` 值（账单默认 `unknown`，User-Agent 默认 `cli`） |
| `ANTHROPIC_USER_AGENT` | 直接覆盖合成的 `User-Agent` 请求头（仅作用于其他客户端分支；原生 Desktop/CLI 自身的 `User-Agent` 始终原样转发） |
| `ANTHROPIC_CUSTOM_HEADERS` | 向合成 CLI 配置添加按行分隔的自定义应用请求头；凭据和传输层请求头会被忽略 |
| `ANTHROPIC_BETAS` | 将逗号分隔的自定义 beta 追加到合成的 `anthropic-beta` 值（仅作用于非 Claude Code 分支） |

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
