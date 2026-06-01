---
sidebar_position: 1
---

# Codex（ChatGPT）集成

Claude Code Router 支持将请求路由到 **Codex 后端 API**（即 GitHub Copilot 的 ChatGPT 模型，GPT-5 系列）。需要通过 GitHub 账户进行 OAuth 认证。

## 工作原理

1. `ccr codex-auth` 打开浏览器进行 GitHub Copilot 登录
2. OAuth 流程获取访问令牌并保存在 `~/.claude-code-router/codex-token.json`
3. `codex` 转换器使用该令牌进行 API 请求认证
4. 请求通过 Responses API 格式发送到 `https://api.githubcopilot.com`

## 前置要求

- 有效的 [GitHub Copilot](https://github.com/features/copilot) 订阅
- 正在运行的 Claude Code Router（Docker Compose 或本地安装）

## 设置步骤

### 1. 认证

运行 OAuth 流程：

```bash
ccr codex-auth
```

这将打开浏览器。使用您的 GitHub 账户登录并授权应用程序。令牌会自动保存。

验证令牌是否有效：

```bash
ccr codex-auth --check
```

### 2. 配置提供商

将 Codex 提供商添加到 `~/.claude-code-router/config.json`：

```json
{
  "Providers": [
    {
      "name": "codex",
      "baseUrl": "https://api.githubcopilot.com",
      "apiKey": "$CODEX_ACCESS_TOKEN",
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
