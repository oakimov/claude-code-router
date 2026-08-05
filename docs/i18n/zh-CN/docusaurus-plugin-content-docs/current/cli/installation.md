---
title: 安装
sidebar_position: 2
---

# 安装

使用已发布的 Docker 镜像运行 Claude Code Router。

## 前置要求

- **Docker**
- 来自您偏好的 LLM 提供商的 API 密钥

**使用 Docker 安装时无需 Node.js** —— 镜像自带运行时（当前为 Node 22 LTS）。

仅在以下情况需要本地安装 Node.js：从 npm 安装 CLI、从源码运行，或使用 [Chrome 端侧桥接](/docs/server/guides/chrome-on-device)。此时最低要求为 **Node.js ≥ 22.19.0**，并由每个已发布包的 `engines` 字段强制约束：

| 包 | 最低 Node 版本 |
| --- | --- |
| `@caeliq/claude-code-router`（CLI） | ≥ 22.19.0 |
| `@caeliq/llms`（core） | ≥ 22.19.0 |
| `@caeliq/ccr-shared` | ≥ 22.19.0 |

该下限来自 `undici`（用于向提供商发起请求的 HTTP 客户端）。`22.19.0` 是 Node 22 的 **LTS** 版本；任何更新的 Node 22 或 24 同样可用。在更旧的运行时上安装会出现 `EBADENGINE` 警告，并在运行时报错。

## 通过 Docker 安装

运行已发布的镜像，并挂载配置目录：

```bash
mkdir -p ~/.claude-code-router
docker run -d --name ccr \
  -p 3456:3456 \
  -v ~/.claude-code-router:/root/.claude-code-router \
  ghcr.io/oakimov/claude-code-router:latest
```

路由服务将在 `http://localhost:3456` 启动。请在 `config.json` 中设置 `"HOST": "0.0.0.0"`，以便端口映射能够访问服务器。

查看日志：

```bash
docker logs -f ccr
```

停止服务：

```bash
docker stop ccr && docker rm ccr
```

> **注意**：如需从仓库源码构建，请参见[服务器部署](/docs/server/deployment)指南 — 其中介绍了 `packages/server` 下的 `docker-compose.yml`，它会从源码本地构建镜像。

## 下一步

安装完成后，前往 [快速开始](/docs/cli/quick-start) 了解如何配置和使用路由器。
