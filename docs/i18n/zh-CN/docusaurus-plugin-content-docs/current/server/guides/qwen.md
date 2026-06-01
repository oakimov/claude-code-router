---
sidebar_position: 2
---

# Qwen Chat 集成

Claude Code Router 支持通过 `qwen.aikit.club` API 路由到 **Qwen Chat（通义千问）**。它使用基于 JWT 的认证方式，您可以粘贴从 `chat.qwen.ai` Web 应用复制的令牌。

## 工作原理

1. `ccr qwen-auth` 提示您粘贴来自 `chat.qwen.ai` 的 JWT 令牌
2. 令牌保存在 `~/.claude-code-router/qwen-token.json`
3. 自动令牌轮换 — 检测到过期令牌时会提示您输入新令牌
4. Qwen 在响应中附加的 `<details>...</details>` 元数据块会被自动移除

## 前置要求

- 可以访问 `qwen.aikit.club` 的 Qwen Chat API
- Qwen Chat 账户

## 设置步骤

### 1. 获取 JWT 令牌

1. 在浏览器中打开 [chat.qwen.ai](https://chat.qwen.ai)
2. 打开开发者工具（`F12` 或 `Cmd+Option+I`）
3. 进入 **Application** → **Local Storage** → `https://chat.qwen.ai`
4. 找到包含 `access_token` 或 `token` 值的键
5. 复制完整的 JWT 令牌字符串

### 2. 认证

运行认证命令：

```bash
ccr qwen-auth
```

系统会提示您粘贴 JWT 令牌。CLI 会安全地存储它，并在需要时自动刷新。

### 3. 配置提供商

将 Qwen 提供商添加到 `~/.claude-code-router/config.json`：

```json
{
  "Providers": [
    {
      "name": "qwen",
      "baseUrl": "https://qwen.aikit.club/v1/chat/completions",
      "apiKey": "$QWEN_ACCESS_TOKEN",
      "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
      "transformer": {
        "use": ["qwen-auth", "OpenAI"]
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
