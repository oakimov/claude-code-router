---
sidebar_position: 7
---

# ccr claude-auth / ccr codex-auth / ccr qwen-auth

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