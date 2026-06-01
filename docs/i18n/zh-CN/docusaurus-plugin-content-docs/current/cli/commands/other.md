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

## ccr activate

输出用于与外部工具集成的 shell 环境变量。

```bash
ccr activate
```

## ccr codex-auth

通过 OAuth 认证 Codex（ChatGPT）API。打开浏览器进行 GitHub Copilot 登录并保存访问令牌。

```bash
ccr codex-auth
```

## ccr qwen-auth

认证 Qwen Chat API。提示您粘贴从 `chat.qwen.ai` localStorage 复制的 JWT 令牌，并自动管理令牌轮换。

```bash
ccr qwen-auth
```

## ccr chrome-bridge

启动 Chrome 内置模型（Gemini Nano）桥接进程。**必须在宿主机上运行**（而不是 Docker 容器内）。通过 CDP 连接 Chrome 的 Prompt API。

```bash
ccr chrome-bridge
```

## ccr model get

非交互式发现提供商的可用模型。获取远程模型列表，解析自定义 JSON 结构，并将缺失的模型追加到配置中。

```bash
ccr model get <provider-name>
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

- [入门](/zh/docs/intro) - Claude Code Router 简介
- [配置](/zh/docs/config/basic) - 配置指南
