---
title: 分支更新日志 — 新增功能一览
date: 2026-06-01
tags: [changelog, fork, features]
---

# 分支更新日志

本分支在 claude-code-router 基础上新增了多项提供商集成、UI 改进和基础设施变更。以下是各功能的时间线。

{/* truncate */}

## 2026 年 1 月 — Mistral 集成

添加了 **Mistral AI 转换器**，支持直接 API 调用、推理参数处理和专用的 Docker Compose 配置。将 Mistral 转换逻辑解耦为共享工具函数。

- `feat: add Mistral transformer for direct API support`
- `fix: correctly handle reasoning parameter for Mistral API`

## 2026 年 4 月 — Gemini 稳定性、代码质量与构建流水线

大幅改进了 Gemini/Gemma 流式传输的可靠性。修复了因请求体中 `thoughtSignature` 位置错误导致的 Gemini 500 错误以及多轮对话中的工具调用失败。项目已完全本地化（所有注释翻译为英文）。**UI 包已集成到 Docker 构建流程**中，管理界面可直接从容器提供服务。

- `build: include UI package in Docker build process`
- `refactor: localize codebase by translating all comments to English`
- `fix: resolve Gemini 500 errors and tool use failures`
- `feat: add ThinkingSequencer for ordered Gemini streaming`

## 5 月 7 日 — DeepSeek 推理重放

实现了 DeepSeek 模型的**强制推理重放**。DeepSeek 要求将之前的助手推理内容包含在后续请求中 — 否则模型在多轮对话中会丢失上下文。`reasoning` 转换器自动捕获响应中的推理输出并在下一请求中重放。

- `feat: implement mandatory reasoning replay for DeepSeek provider`

## 5 月 7 日 — Codex（ChatGPT）集成

添加了 **Codex 转换器**，支持 OpenAI Responses API，采用 PKCE 流程的 OAuth 认证（`ccr codex-auth`）。支持 SSE 流式传输、推理/思考内容、工具调用、网络搜索和图像处理。需要 ChatGPT Plus 或 Pro 订阅。

- `feat: add Codex transformer and support for Codex authentication`

## 5 月 8 日 — 模型发现

添加了**非交互式模型发现**（`ccr model get`）。从任何提供商的 API 获取远程模型列表，解析自定义 JSON 结构，并将缺失的模型追加到本地配置中，不修改现有条目。

- `feat: add support for custom provider model discovery`

## 5 月 10 日 — Chrome 内置模型（Gemini Nano）

添加了 **Chrome 内置模型桥接和转换器**，用于 Chrome 的 Gemini Nano 模型（约 4GB 本地 LLM）。通过 Chrome DevTools 协议（CDP）通信，提供零成本、零延迟的本地推理。支持卡死恢复（动态温度调节）、通过 `responseConstraint` 的结构化 JSON 输出以及完整 SSE 流式传输。

- `feat: add Chrome on-device bridge and transformer for Gemini Nano`

## 5 月 12 日 — 桥接稳定性改进

增强了 Chrome 桥接：多会话仪表板、空闲会话回收、通过反射检测改进了工具循环预防，以及优化了系统提示以保持工具交互的一致性。

- `feat(bridge): add multi-session dashboard, idle eviction`
- `refactor: migrate bridge to explicit <tool_result> XML tags`

## 5 月 18 日 — 透传模式与 DeepSeek 思考修复

添加了基于提供商的透传模式，支持直接的 Anthropic 风格认证处理，并修复了 DeepSeek 思考功能，无需客户端发送 reasoning 参数。

- `feat: add per-provider passthrough mode`
- `fix: enable DeepSeek thinking without client reasoning param`

## 5 月 25 日 — Codex 修复与 Anthropic Effort 透传

修复了 Codex 转换器的工具调用参数，恢复了流式传输用量计算，并直接传递 Claude Code 的 effort 参数而非通过 budget_tokens 映射。

- `fix(anthropic): pass through Claude Code effort directly`

## 6 月 1 日 — Qwen Chat 集成

添加了 **Qwen Chat 认证转换器**，支持基于 JWT 的认证（`ccr qwen-auth`），并在 `/qwen/auth` 提供基于浏览器的认证页面，支持书签工具以便从 `chat.qwen.ai` 轻松提取令牌。自动令牌轮换和移除 Qwen 响应中的 `<details>...</details>` 元数据块。

- `feat(qwen): add Qwen Chat auth (ccr qwen-auth) and provider transformer`

## 文档

所有功能都有专门的集成指南文档：

- [Codex 集成](/docs/server/guides/codex)
- [Qwen Chat 集成](/docs/server/guides/qwen)
- [Chrome 内置模型](/docs/server/guides/chrome-on-device)
- [DeepSeek 推理重放](/docs/server/guides/deepseek-reasoning)
- [模型发现](/docs/server/guides/model-discovery)
