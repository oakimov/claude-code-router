---
title: 安装
sidebar_position: 2
---

# 安装

使用 Docker Compose 运行 Claude Code Router。

## 前置要求

- **Docker** 和 **Docker Compose**
- 来自您偏好的 LLM 提供商的 API 密钥

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

安装完成后，前往 [快速开始](/zh/docs/quick-start) 了解如何配置和使用路由器。
