---
title: 其他命令
sidebar_position: 4
---

# 其他命令

管理 Claude Code Router 的其他 CLI 命令。

## ccr stop

停止运行中的服务器。

```bash
ccr stop
```

## ccr restart

重启服务器。

```bash
ccr restart
```

## ccr code

通过路由器执行 claude 命令。

```bash
ccr code [参数...]
```

## ccr ui

在浏览器中打开 Web UI。

```bash
ccr ui
```

Web UI 的 **Debug agent** 试验场（`/debug`）：

- **CCR 与直连** — CCR 走本地路由（`provider,model`）。直连使用供应商 `api_base_url` 和裸模型名。两种模式下 API 密钥、`$ENV` 占位符和 OAuth 令牌都从 CCR 配置在服务端读取，浏览器不需要提供。
- **入站协议**（CCR）— Chat Completions（`/v1/chat/completions`）、Messages（`/v1/messages`）或 Responses（`/v1/responses`）。
- **系统提示、推理力度与工具** — 指令区预填默认 Debug agent 系统提示。可从文件加载系统提示（文本）、工具（JSON 数组或对象）或用户提示（文本）。用户消息发送后清空；系统提示和工具在浏览器会话内保留。可切换模型继续同一段对话（消息、推理与工具调用）。可选 CCR 推理力度：`none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max` / `ultra`。工具调用只做 **桩实现**（仅供检查，不会执行用户 JSON）。
- **Token 用量** — 每条助手消息显示可折叠的 token 总量，展开后为读取、写入、缓存读取、缓存写入。
- **刷新 OAuth** — 对 `claude-auth`、Codex OAuth（不含 PAT）、`qwen-auth`、`antigravity-auth`、`xai-auth` 强制刷新。令牌不会返回给浏览器。

对话标签页通过 `POST /api/debug/chat` 流式传输。请求体标签会按所选入站端点（Chat Completions / Messages / Responses）预渲染本轮 JSON，可编辑后发送到请求 URL。响应区显示最新原始响应体、响应头和 HTTP 状态。可拖动请求与响应之间的分隔条调整高度。复制 cURL 会复制完整命令，授权密钥用 `PLACEHOLDER` 代替。

## ccr activate

输出用于与外部工具集成的 shell 环境变量。

```bash
ccr activate
```

## 全局选项

这些选项可用于任何命令：

| 选项 | 说明 |
|------|------|
| `-h, --help` | 显示帮助 |
| `-v, --version` | 显示版本号 |
| `--config <路径>` | 配置文件路径 |
| `--verbose` | 启用详细输出 |

## 示例

### 停止服务器

```bash
ccr stop
```

### 使用自定义配置重启

```bash
ccr restart --config /path/to/config.json
```

### 打开 Web UI

```bash
ccr ui
```

## 相关文档

- [入门](/docs/cli/intro) - Claude Code Router 简介
- [配置](/docs/server/config/basic) - 配置指南
