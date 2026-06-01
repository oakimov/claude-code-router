---
sidebar_position: 1
---

# Codex（ChatGPT）集成

Claude Code Router 可以使用 **ChatGPT/Codex 订阅** 将 Claude Code 请求通过 OpenAI 的模型路由。Codex 后端是 OpenAI ChatGPT 产品的基础，此集成让您能够将该订阅与 Claude Code 一起使用。

认证使用 OpenAI OAuth — 需要 ChatGPT Plus 或 Pro 订阅。

## 工作原理

1. `ccr codex-auth` 输出授权 URL 并在端口 1455 启动本地回调服务器
2. 您在浏览器中打开该 URL，登录您的 OpenAI / ChatGPT 账户
3. OpenAI 重定向到 `http://localhost:1455/auth/callback`，CCR 服务器在此交换授权码以获取令牌（PKCE 流程）
4. 令牌保存到 `~/.claude-code-router/codex_auth.json`
5. 返回终端并按下 Enter — CLI 确认令牌已保存
6. `codex` 转换器读取访问令牌并用于 API 请求认证
7. 令牌即将过期时，自动使用刷新令牌续期

## 前置要求

- [ChatGPT Plus 或 Pro](https://chat.openai.com) 订阅
- 正在运行的 Claude Code Router（Docker Compose 或本地安装）

## 设置步骤

### 1. 认证

运行 OAuth 流程：

```bash
ccr codex-auth
```

CLI 会输出一个授权 URL。在浏览器中打开该 URL，使用您的 OpenAI / ChatGPT 账户登录并授权应用程序。浏览器显示"Authentication Successful"后，返回终端并按下 Enter。令牌会自动保存。

### 2. 配置提供商

将 Codex 提供商添加到 `~/.claude-code-router/config.json`：

```json
{
  "Providers": [
    {
      "name": "codex",
      "baseUrl": "https://chatgpt.com/backend-api/codex",
      "apiKey": "oauth_dummy_key",
      "models": ["gpt-5", "gpt-5-high", "gpt-5-mini"],
      "transformer": {
        "use": ["codex"]
      }
    }
  ],
  "Router": {
    "default": "codex,gpt-5"
  }
}
```

### 3. 重启

```bash
docker compose restart ccr
```

## 功能特性

- **SSE 流式传输** — 完整流式支持，实时响应
- **推理/思考内容** — 支持具有推理能力的模型
- **工具调用** — 支持多个工具的 Function Calling
- **网络搜索** — 通过 `{ type: "web_search" }` 内置网络搜索
- **图片处理** — 支持图片输入

## 使用示例

将 Codex 设为默认模型或路由特定场景：

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
| `gpt-5-high` | 高性能版本（推理任务） |
| `gpt-5-mini` | 轻量版本（后台任务） |

## 故障排除

**令牌过期或无效**：重新运行 `ccr codex-auth` 刷新令牌。

**找不到提供商**：确保配置中的提供商名称与 `body.model` 匹配（例如 `codex,gpt-5`）。
