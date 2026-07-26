---
sidebar_position: 7
---

# ccr claude-auth / ccr codex-auth / ccr qwen-auth / ccr antigravity-auth

需要 OAuth 或 JWT 令牌的提供商后端的认证命令。

## ccr claude-auth

通过 OAuth 与 PKCE，使用你的 Claude Pro 或 Max 订阅认证 Anthropic API。

```bash
ccr claude-auth
```

### 工作原理

1. CLI 生成 PKCE 挑战并输出来自 `claude.ai` 的授权 URL
2. 在浏览器中打开该 URL，登录你的 Claude 账户
3. Claude 重定向到 `http://localhost:1455/callback`，由 CCR 服务器用授权码交换令牌
4. 令牌保存到 `~/.claude-code-router/claude_auth.json`
5. 回到终端并按 Enter — CLI 会确认令牌已保存
6. `claude-auth` 转换器读取该令牌用于 API 请求
7. 令牌即将过期时，自动使用刷新令牌续期

### 前置要求

- [Claude Pro 或 Max](https://claude.ai) 订阅
- CCR 服务器必须正在运行（它在端口 1455 上托管 OAuth 回调）

另见：[Claude 订阅集成指南](/zh/docs/server/guides/claude-auth)。`claude-auth` 转换器会自动发送 Anthropic 的 `oauth-2025-04-20` beta，以便订阅 OAuth Bearer 令牌被接受。

---

## ccr codex-auth

通过 OpenAI OAuth 与 PKCE 认证 Codex（ChatGPT）后端 API。

```bash
ccr codex-auth
```

### 工作原理

1. CLI 生成 PKCE 挑战并输出来自 `auth.openai.com` 的授权 URL
2. 它在 `http://localhost:1455/auth/callback` 启动回调服务器
3. 您在浏览器中打开该 URL，登录您的 OpenAI / ChatGPT 账户
4. OpenAI 重定向到回调服务器，后者将授权码交换为令牌
5. 令牌保存到 `~/.claude-code-router/codex_auth.json`
6. `codex` 转换器读取此令牌用于 API 请求
7. 令牌即将过期时，自动使用刷新令牌续期

### 前置要求

- [ChatGPT Plus 或 Pro](https://chat.openai.com) 订阅

---

## ccr antigravity-auth

通过具有 PKCE 的 OAuth 认证 Google 的 Antigravity 网关。

```bash
ccr antigravity-auth
ccr antigravity-auth --manual
ccr antigravity-auth --project <gcp-project-id>
```

### 工作原理

1. CLI 生成 PKCE 挑战、写入验证器文件并打印 Google 授权 URL
2. 使用具有 Antigravity 访问权限的 Google 账户登录
3. Google 重定向到 `http://localhost:51121/oauth-callback`
4. Docker Compose 将 **51121 → 3456** 映射到 CCR 服务器（与 Codex `1455 → 3456` 的思路相同）
5. 公共 Fastify 路由 `GET /oauth-callback` 交换授权码并写入 `~/.claude-code-router/antigravity_auth.json`
6. 在终端中按 Enter 进行确认

### 选项

- `--manual` — 粘贴重定向 URL；CLI 会在没有服务器的情况下完成交换（无 compose / 无头模式）
- `--project <id>` — 将 `project_id` 注入验证器 / 认证文件

### 前置要求 / 说明

- CCR 服务器必须正在运行（托管 `/oauth-callback`）
- 使用 Docker 时：重新创建 compose 以发布 `51121:3456`
- 从非 IDE 客户端使用 Antigravity IDE OAuth 客户端凭据可能会违反 Google 服务条款

### 提供商配置

```json
{
  "name": "antigravity",
  "api_base_url": "https://daily-cloudcode-pa.sandbox.googleapis.com",
  "api_key": "oauth",
  "project_id": "$ANTIGRAVITY_PROJECT_ID",
  "models": [
    "gemini-3-pro-high",
    "gemini-3-flash",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking"
  ],
  "transformer": {
    "use": [
      ["gemini", { "cachedContent": false, "thoughtSignatureFallback": "skip" }],
      "antigravity-auth"
    ]
  }
}
```

该链中的 Gemini 选项（完整说明见 [转换器 → gemini](/zh/docs/server/config/transformers#选项-cachedcontent-与-thoughtsignaturefallback)）：

- **`cachedContent: false`** — Antigravity 没有 Google `cachedContents` 资源。Gemini 转换器默认是 `true`（公共 Gemini 可能创建/复用该服务端前缀缓存）。在此保留 `true` 会导致 404。
- **`thoughtSignatureFallback: "skip"`** — 默认值的显式写法。Gemini 3 / Antigravity 要求工具调用带有 `thoughtSignature`；Claude Code 无法在 Anthropic `tool_use` 上携带它，因此 CCR 缓存并还原签名。未命中时，`"skip"` 会在第一个 `functionCall` 上盖印 Google 的 `skip_thought_signature_validator` 哨兵，避免 400。取值名称指的是该哨兵 — **不是**“关闭回退”。仅在端点拒绝该哨兵时设为 `"none"`。

---

## ccr qwen-auth

打开基于浏览器的认证页面 `http://localhost:3456/qwen/auth` 进行令牌管理。

```bash
ccr qwen-auth
```

### 工作原理

1. 命令提示您在浏览器中打开 `http://localhost:3456/qwen/auth`（或直接导航到该页面）
2. 在认证页面上，使用**书签工具**（推荐）或手动粘贴令牌：
   - **书签工具**：将"Get Qwen Token"拖到书签栏，打开 `chat.qwen.ai` 并点击它 — 令牌自动发送回来
   - **手动**：在 `chat.qwen.ai` 的 DevTools Console 中运行 `copy(localStorage.getItem('token'))`，然后在认证页面上粘贴
3. 令牌经过 Qwen API 验证并保存到 `~/.claude-code-router/qwen-auth.json`
4. 自动令牌轮换 — 下次请求时检测过期令牌

### 前置要求

- Qwen Chat 账户以及访问 `qwen.aikit.club` 的权限