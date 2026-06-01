---
sidebar_position: 8
---

# ccr chrome-bridge

启动 Chrome 内置模型（Gemini Nano）桥接进程。

```bash
ccr chrome-bridge
```

## 说明

启动一个桥接进程，通过 Chrome DevTools 协议（CDP）连接到 Chrome 的 Prompt API，使路由器能够使用 Chrome 内置的 Gemini Nano 模型（约 4GB 本地 LLM）。

**必须在宿主机上运行**（不是在 Docker 内），因为它需要直接访问 Chrome 的调试端口。

## 工作原理

1. 桥接连接到 Chrome 的远程调试端口（`--remote-debugging-port=9229`）
2. 与 Prompt API 通信以发送提示和接收响应
3. 路由器的 `chrome-on-device` 转换器在 OpenAI Chat Completions 格式和 Prompt API 之间做转换
4. 响应通过标准 SSE 流式返回

## 功能特性

- **零 API 费用** — 完全在您的设备上本地推理
- **流式支持** — 完整 SSE 流式传输
- **结构化输出** — 使用 `responseConstraint` 实现可靠的 JSON 工具调用
- **自动卡死恢复** — 如果模型卡死，自动以更高温度重试

## 前置要求

- Google Chrome（Canary 或 Dev）并启用 Gemini Nano
- 宿主机上安装 Node.js

详细设置说明请参见 [Chrome 内置模型集成指南](/zh/docs/server/guides/chrome-on-device)。
