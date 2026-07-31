---
title: 安装
sidebar_position: 2
---

# 安装

使用 Docker Compose 运行 Claude Code Router。

## 前置要求

- **Docker** 和 **Docker Compose**
- 来自您偏好的 LLM 提供商的 API 密钥

**使用 Docker 安装时无需 Node.js** —— 镜像自带运行时（当前为 Node 22 LTS）。

仅在以下情况需要本地安装 Node.js：从 npm 安装 CLI、从源码运行，或使用 [Chrome 端侧桥接](/docs/server/guides/chrome-on-device)。此时最低要求为 **Node.js ≥ 22.19.0**，并由每个已发布包的 `engines` 字段强制约束：

| 包 | 最低 Node 版本 |
| --- | --- |
| `@caeliq/claude-code-router`（CLI） | ≥ 22.19.0 |
| `@caeliq/llms`（core） | ≥ 22.19.0 |
| `@caeliq/ccr-shared` | ≥ 22.19.0 |

该下限来自 `undici`（用于向提供商发起请求的 HTTP 客户端）。`22.19.0` 是 Node 22 的 **LTS** 版本；任何更新的 Node 22 或 24 同样可用。在更旧的运行时上安装会出现 `EBADENGINE` 警告，并在运行时报错。

## 通过 Docker Compose 安装

克隆仓库并使用提供的 Compose 文件启动服务：

```bash
git clone https://github.com/oakimov/claude-code-router.git
cd claude-code-router/packages/server
docker compose up --build -d
```

路由服务将在 `http://localhost:3456` 启动。

查看日志：

```bash
docker compose logs -f ccr
```

停止服务：

```bash
docker compose down
```

## 下一步

安装完成后，前往 [快速开始](/docs/cli/quick-start) 了解如何配置和使用路由器。
