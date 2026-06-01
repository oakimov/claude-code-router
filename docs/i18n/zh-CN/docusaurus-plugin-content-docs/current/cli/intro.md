---
title: 开始使用
sidebar_position: 1
slug: /
---

# 欢迎使用 Claude Code Router

[![npm version](https://badge.fury.io/js/%40musistudio%2Fclaude-code-router.svg)](https://www.npmjs.com/package/@musistudio/claude-code-router)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Node Version](https://img.shields.io/node/v/@musistudio/claude-code-router.svg)

**Claude Code Router** 是一个强大的工具，允许你在没有 Anthropic 账户的情况下使用 [Claude Code](https://claude.ai/code)，并将请求路由到其他 LLM 提供商。

## 特性

- **多提供商支持**: 路由到 DeepSeek、Gemini、Groq、OpenRouter 等
- **智能路由**: 内置不同任务类型的场景（后台、思考、网络搜索、图像）
- **项目级配置**: 每个项目自定义路由
- **自定义路由函数**: 编写 JavaScript 定义自己的路由逻辑
- **转换器系统**: 无缝适配不同提供商之间的 API 差异
- **代理系统**: 可扩展的插件架构，实现自定义功能
- **Web UI**: 内置管理界面，方便配置
- **CLI 集成**: 与现有的 Claude Code 工作流无缝集成

## 快速开始

### 安装

```bash
git clone https://github.com/oakimov/claude-code-router.git
cd claude-code-router/packages/server
docker compose up --build -d
```

### 基本使用

配置 Claude Code 使用路由器：

```bash
export ANTHROPIC_BASE_URL="http://localhost:3456/v1"
export ANTHROPIC_API_KEY="dummy"
claude
```

## 服务管理

```bash
docker compose up --build -d    # 启动路由器
docker compose down             # 停止路由器
docker compose restart ccr      # 重启路由器
docker compose logs -f ccr      # 查看日志
```

## Web UI

在浏览器中访问 `http://localhost:3456/ui/` 管理配置和监控服务。

## 架构

Claude Code Router 由四个主要组件组成：

- **Server** (`@CCR/server`): 处理 API 路由和转换的核心服务器
- **Shared** (`@CCR/shared`): 共享常量和工具
- **UI** (`@CCR/ui`): Web 管理界面（React + Vite）

## 下一步

- [安装指南](/zh/docs/cli/installation) — 详细安装说明
- [快速开始](/zh/docs/quick-start) — 5 分钟入门
- [配置](/zh/docs/cli/config/basic) — 了解如何配置路由器
- [集成指南](/zh/docs/category/integration-guides) — 提供商特定功能设置

## 许可证

MIT © [oakimov](https://github.com/oakimov)
