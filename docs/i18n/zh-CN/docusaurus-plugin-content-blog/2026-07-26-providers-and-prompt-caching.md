---
title: 新提供商与提示缓存修复
date: 2026-07-26
tags: [changelog, fork, providers, caching]
---

# 新提供商与提示缓存修复

自 [6 月分支更新日志](/zh/blog/2026/06/01/fork-changelog) 以来，本分支新增了多个订阅类提供商，并修复了转换器栈中的提示缓存可靠性。以下日期来自 git 历史：

## 2026 年 6 月 18 日 — Claude Auth（Pro/Max 订阅）

添加了 **claude-auth OAuth PKCE 流程**（`ccr claude-auth`），可将 Claude Pro/Max 订阅令牌像其他订阅类提供商一样通过网关路由。

- `feat: add claude-auth OAuth PKCE flow for Claude Pro/Max subscription routing`

## 2026 年 7 月 19 日 — Cursor SDK 提供商

添加了 **Cursor SDK 提供商桥接**及文档，支持通过路由器用 Cursor 后端模型，并提供流式传输与思考内容传输。

- `feat(cursor): add Cursor SDK provider bridge with docs`

## 2026 年 7 月 20 日 — Claude Auth 缓存控制往返保留

加固了 claude-auth：多客户端 `anthropic-beta` 请求头解析，以及 **cache-control 往返保留**，使提示缓存断点在转换器链路中不被错误剥离或改写。

- `feat(claude-auth): multi-client anthropic-beta resolution and cache-control round-trip preservation`

## 2026 年 7 月 24 日 — 提供商原生提示缓存

上线了跨 Anthropic、OpenAI/Codex、OpenRouter、Vercel、Gemini 等相关路径的**提供商原生提示缓存**，以及 OpenCode Zen 会话自愈。这是长会话与多提供商路由下缓存可靠性的主要修复。

- `feat(caching+opencode): provider-native prompt caching and OpenCode Zen session self-heal`
- `fix(codex): make PAT and OAuth auth reliable`

## 2026 年 7 月 26 日 — Antigravity OAuth 与 Cursor/Gemini 加固

添加了 **Antigravity OAuth**（`ccr antigravity-auth`），用于 Google Antigravity 后端的 Claude/Gemini 访问，并加固了 Gemini/Cursor 流式传输、思考内容传输以及中断后的 SDK 流恢复。

- `feat(antigravity+cursor): add Antigravity OAuth and harden Gemini/Cursor paths`
- `fix(cursor): harden streaming and thinking transport`
- `fix(cursor): recover cleanly after interrupted SDK streams`

## 文档

- [Claude Auth](/zh/docs/server/guides/claude-auth)
- [Cursor SDK](/zh/docs/server/guides/cursor)
- [Codex 集成](/zh/docs/server/guides/codex)
- [更早的分支时间线](/zh/blog/2026/06/01/fork-changelog)
