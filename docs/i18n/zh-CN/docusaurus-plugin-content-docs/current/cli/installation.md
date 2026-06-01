---
title: 安装
sidebar_position: 2
---

# 安装

您可以使用 Docker Compose（推荐）或包管理器来运行 Claude Code Router。

## 前置要求

- **Docker** 和 **Docker Compose**（使用 Docker 方式）
- **Node.js**: >= 18.0.0（使用 npm/pnpm 方式）
- 来自您偏好的 LLM 提供商的 API 密钥

## 通过 Docker Compose 安装（推荐）

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

## 通过包管理器安装（备选）

如果不使用 Docker，也可以通过包管理器全局安装。

### 通过 npm 安装

```bash
npm install -g @musistudio/claude-code-router
```

### 通过 pnpm 安装

```bash
pnpm add -g @musistudio/claude-code-router
```

### 通过 Yarn 安装

```bash
yarn global add @musistudio/claude-code-router
```

## 验证安装（包管理器方式）

安装完成后，验证 `ccr` 命令是否可用：

```bash
ccr --version
```

您应该看到版本号显示。

## 下一步

安装完成后，前往 [快速开始](/zh/docs/quick-start) 了解如何配置和使用路由器。
