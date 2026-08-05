---
sidebar_position: 3
---

# Chrome 内置模型（Gemini Nano）

Claude Code Router 可以使用 **Chrome 内置的 Gemini Nano** 模型 — 一个约 4GB 的本地 LLM，完全在您的设备上运行，无需 API 费用，也无需外部网络延迟。通过 Chrome DevTools 协议（CDP）访问 Chrome 的 Prompt API。

## 工作原理

1. `ccr chrome-bridge` 启动一个桥接进程，连接到 Chrome 的 Prompt API
2. `chrome-on-device` 转换器在 OpenAI Chat Completions 格式和 Chrome 的 Prompt API 格式之间做转换
3. 响应通过标准 SSE 流式返回
4. 模型完全在本地 Chrome 中运行 — 数据不会离开您的机器

## 前置要求

- **Google Chrome**（Canary 或 Dev 频道）并启用 Gemini Nano
- **宿主机上安装 Node.js**（桥接进程在宿主机上运行，不在 Docker 内）
- 正在运行的 Claude Code Router

## 设置步骤

### 1. 在 Chrome 中启用 Gemini Nano

1. 安装 [Chrome Canary](https://www.google.com/chrome/canary/) 或 Chrome Dev
2. 打开 `chrome://flags/#prompt-api-for-gemini-nano`，启用 **Prompt API for Gemini Nano**
3. 打开 `chrome://flags/#optimization-guide-on-device-model`，启用 **Enables optimization guide on device**
4. 重启 Chrome
5. 等待模型下载完成（检查 `chrome://components/` 中的 **Optimization Guide On Device Model**）
6. 验证模型已就绪：打开 DevTools 控制台并运行：
   ```js
   (await ai.languageModel.capabilities()).available
   ```
   应返回 `"readily"`

### 2. 启动桥接

在宿主机上运行桥接进程（不是在 Docker 中）：

```bash
ccr chrome-bridge
```

桥接默认监听 `http://127.0.0.1:9229`。

### 3. 配置提供商

将 Chrome 提供商添加到 `~/.claude-code-router/config.json`：

```json
{
  "Providers": [
    {
      "name": "chrome",
      "baseUrl": "http://127.0.0.1:9229",
      "apiKey": "dummy",
      "models": ["gemini-nano"],
      "transformer": {
        "use": ["chrome-on-device"]
      }
    }
  ],
  "Router": {
    "background": "chrome,gemini-nano"
  }
}
```

### 4. 重启

```bash
docker compose restart ccr
```

## 功能特性

- **零 API 费用** — 完全本地推理
- **零外部延迟** — 无需网络请求
- **流式和非流式** — 完整 SSE 流式支持
- **结构化输出** — 使用 `responseConstraint` 实现可靠的 JSON 工具调用
- **自动卡死恢复** — 如果模型卡死（仅输出空白字符），桥接会自动以更高温度重试

## 会话管理

桥接维护持久的 `LanguageModel` 会话 — 会话历史会在会话内自然延续，而不是每轮重建。

- **多会话支持**：请求通过 `User-Agent + IP` 哈希指纹进入独立会话，允许多个并发的 Claude Code 实例而不会造成上下文污染。内置的 Web 仪表盘（在桥接端口上提供服务）显示所有会话的实时统计，包括轮数、空闲时间和上下文使用量。
- **空闲会话驱逐**：空闲超过 5 分钟的会话会自动销毁以释放资源。`cli` 会话（仪表盘默认）永远不会被驱逐。也可以通过仪表盘的 Evict 按钮手动驱逐会话。
- **自动压缩**：在上下文使用量达到 85% 时触发，重置会话但保留系统提示词。
- 桥接通过 CDP 连接 Chrome，使用 5 分钟协议超时来处理缓慢的模型推理。

## 限制

- **模型质量** — Gemini Nano 是小型本地模型，最适合简单任务和后台工作
- **Chrome 依赖** — 需要启用 Gemini Nano 的 Chrome；非所有浏览器可用
- **宿主机 Node.js** — 桥接进程必须在宿主机上运行，而不是 Docker 中
- **无外部工具** — 限于模型的内置功能

## 故障排除

**桥接无法连接**：确保 Chrome 以 `--remote-debugging-port=9229` 运行，否则桥接无法连接到 Prompt API。

**模型不可用**：检查 `chrome://components/` 中的 Optimization Guide 模型状态。如果未下载，请等待并重启 Chrome。

**响应缓慢**：首次请求可能较慢，因为模型需要加载。后续请求会更快。
