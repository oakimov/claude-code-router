---
title: 快速开始
sidebar_position: 3
---

# 快速开始

5 分钟内启动并运行 Claude Code Router。

## 1. 启动路由器

运行已发布的 Docker 镜像：

```bash
mkdir -p ~/.claude-code-router
docker run -d --name ccr \
  -p 3456:3456 \
  -v ~/.claude-code-router:/root/.claude-code-router \
  ghcr.io/oakimov/claude-code-router:latest
```

路由器将在 `http://localhost:3456` 启动。请确保 `config.json` 中设置了 `"HOST": "0.0.0.0"`，以便端口映射能够访问服务器。

## 2. 配置路由器

编辑位于 `~/.claude-code-router/config.json` 的配置文件（已挂载到容器中）：

```json5
{
  "HOST": "0.0.0.0",
  "PORT": 3456,
  "Providers": [
    {
      "name": "my-provider",
      "baseUrl": "https://api.example.com/v1",
      "apiKey": "$YOUR_API_KEY",
      "models": ["model-name"]
    }
  ],
  "Router": {
    "default": "my-provider,model-name"
  }
}
```

编辑完成后重启服务：

```bash
docker restart ccr
```

您也可以访问 `http://localhost:3456/ui/` 通过 Web UI 可视化配置提供商。

## 3. 使用 Claude Code

配置环境变量后直接运行 Claude Code：

```bash
export ANTHROPIC_BASE_URL="http://localhost:3456/v1"
export ANTHROPIC_API_KEY="dummy"
claude
```

您的请求将通过 Claude Code Router 路由到您配置的提供商。

## 修改配置后重启

修改配置文件或通过 Web UI 更改后，重启服务：

```bash
docker restart ccr
```

## 下一步

- [基础配置](/docs/cli/config/basic) — 了解配置选项
- [路由配置](/docs/server/config/routing) — 配置智能路由规则
- [集成指南](/docs/category/integration-guides) — 提供商特定功能设置
