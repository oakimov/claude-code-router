---
sidebar_position: 2
---

# Qwen Chat 集成

Claude Code Router 支持通过 `qwen.aikit.club` API 路由到 **Qwen Chat（通义千问）**。它使用基于 JWT 的认证方式，您可以粘贴从 `chat.qwen.ai` Web 应用复制的令牌。

## 工作原理

1. CCR 服务器在 `http://localhost:3456/qwen/auth` 提供认证页面
2. 页面提供书签工具和手动粘贴表单两种方式获取 JWT 令牌
3. 提交后，令牌经过 Qwen API 验证并保存到 `~/.claude-code-router/qwen_auth.json`
4. 自动令牌轮换 — 下次请求时检测到过期令牌会提示您刷新
4. Qwen 在响应中附加的 `<details>...</details>` 元数据块会被自动移除

## 前置要求

- 可以访问 `qwen.aikit.club` 的 Qwen Chat API
- Qwen Chat 账户

## 设置步骤

### 1. 通过浏览器认证

CCR 服务器在 `http://localhost:3456/qwen/auth` 提供认证页面。在浏览器中打开：

```
http://localhost:3456/qwen/auth
```

页面提供两种方法：

**方法一 — 书签工具（推荐）：**
1. 在认证页面上，将 **"Get Qwen Token"** 按钮拖到您的书签栏
2. 在另一个标签页中打开 [chat.qwen.ai](https://chat.qwen.ai) 并登录
3. 在 Qwen 页面上点击书签工具 — 它会自动读取 localStorage 中的令牌并将其发送回认证页面

**方法二 — 手动粘贴：**
1. 打开 [chat.qwen.ai](https://chat.qwen.ai)，登录，然后打开开发者工具（`F12`）
2. 在 Console 标签页中运行：`copy(localStorage.getItem('token'))`
3. 返回认证页面并将令牌粘贴到表单中

令牌会经过 Qwen API 验证并保存到 `~/.claude-code-router/qwen-auth.json`。

### 3. 配置提供商

将 Qwen 提供商添加到 `~/.claude-code-router/config.json`：

```json
{
  "Providers": [
    {
      "name": "qwen",
      "baseUrl": "https://qwen.aikit.club/v1/chat/completions",
      "apiKey": "oauth_dummy_key",
      "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
      "transformer": {
        "use": ["qwen-auth", "reasoning", "OpenAI"]
      }
    }
  ],
  "Router": {
    "default": "qwen,qwen-max"
  }
}
```

`qwen-auth` 转换器负责：
- 添加 `Authorization: Bearer` 头
- 从响应中移除 `<details>...</details>` 元数据块

`reasoning` 转换器在多轮对话中捕获并重放 Qwen 的推理内容，保持连贯性。

`OpenAI` 转换器注册 `/v1/chat/completions` 端点 — Qwen 使用标准的 Chat Completions 格式。

### 4. 重启

```bash
docker compose restart ccr
```

## 模型参考

| 模型 | 说明 |
|------|------|
| `qwen-max` | 旗舰模型，最佳质量 |
| `qwen-plus` | 平衡性能和成本 |
| `qwen-turbo` | 快速，适合简单任务 |

## 故障排除

**无效的令牌**：JWT 令牌可能已过期。重新运行 `ccr qwen-auth` 并从 `chat.qwen.ai` 粘贴新令牌。

**响应中包含尾随元数据**：`<details>...</details>` 块应自动被移除。如果在原始响应中看到它，说明 `qwen-auth` 转换器可能未激活 — 检查它是否在提供商的 `transformer.use` 数组中。
