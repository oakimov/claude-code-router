---
title: 安装
sidebar_position: 2
---

# 安装

使用已发布的 Docker 镜像运行 Claude Code Router。

## 前置要求

- **Docker**
- 来自您偏好的 LLM 提供商的 API 密钥

**使用 Docker 安装时无需本地 Bun** —— 镜像自带 Bun 运行时（钉在 Bun 1.4.x）。OpenCode 免费层模型依赖 Bun 的 TLS 指纹；Node undici 会被上游拒绝。

从 npm 安装 CLI、从源码运行，或使用 [Chrome 端侧桥接](/docs/server/guides/chrome-on-device) 时，需要本地 **Bun ≥ 1.4.0**，并由每个已发布包的 `engines.bun` 字段强制约束：

| 包 | 最低 Bun 版本 |
| --- | --- |
| `@caeliq/claude-code-router`（CLI） | ≥ 1.4.0 |
| `@caeliq/llms`（core） | ≥ 1.4.0 |
| `@caeliq/ccr-shared` | ≥ 1.4.0 |

CI 与 Docker 全链路使用 Bun（`oven-sh/setup-bun`、`oven/bun:1.4.4-debian`）。工具链不再依赖 Node。Docker 镜像使用 Debian/glibc，因为 `@cursor/sdk` 自带的 tree-sitter 原生插件无法在 Alpine/musl 上加载。

## 通过 Docker 安装

运行已发布的镜像，并挂载配置目录：

```bash
mkdir -p ~/.claude-code-router
docker run -d --name ccr \
  -p 3456:3456 \
  -v ~/.claude-code-router:/home/bun/.claude-code-router \
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
