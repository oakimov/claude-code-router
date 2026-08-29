---
sidebar_position: 2
---

# Cursor SDK 集成

Claude Code Router 可以通过官方 `@cursor/sdk` 将 Claude Code 请求路由到 **Cursor** 模型。与 HTTP 提供商不同，`cursor-sdk` 转换器在进程内完成上游调用，并返回 OpenAI 兼容的 SSE/JSON，再由 AnthropicTransformer 转换回 Claude Code 格式。

默认模式为 **bridge**：由 Cursor 决定下一步，但 **工具仍由 Claude Code 托管**。隔离工作区中会拒绝 Cursor 内置工具；主机工具通过自定义 MCP（`custom-user-tools`）暴露给 SDK。

## 前置要求

- Cursor 账户，以及以 `crsr_` 开头的 API 密钥（来自 Cursor 控制台）
- 正在运行的 Claude Code Router（Docker Compose 或本地）
- 从源码运行或发布包时需要 **Node.js ≥ 22.19.0**（`undici` 的 engines 要求；`@cursor/sdk` 需要 ≥ 22.13）

## 认证

Cursor 认证**不**使用浏览器 OAuth CLI。解析顺序：

1. 以 `crsr_` 开头的提供商 `api_key`（具体密钥，而非未解析的 `$…` / `${…}` 占位符）
2. 否则使用环境变量 `CURSOR_API_KEY`

推荐写法：

```json
"api_key": "crsr_your_key_here"
```

或把密钥留在环境中：

```json
"api_key": "$CURSOR_API_KEY"
```

并导出 / 注入该环境变量（Docker Compose 在设置时会把 `CURSOR_API_KEY` 传入容器）。

## 设置步骤

### 1. 配置提供商

将 Cursor 提供商添加到 `~/.claude-code-router/config.json`：

```json
{
  "Providers": [
    {
      "name": "cursor",
      "api_base_url": "https://cursor.com",
      "api_key": "$CURSOR_API_KEY",
      "models": ["composer-2", "claude-opus-4-8", "gpt-5.4"],
      "transformer": {
        "use": [
          [
            "cursor-sdk",
            {
              "cursorMode": "bridge"
            }
          ]
        ]
      }
    }
  ],
  "Router": {
    "default": "cursor,composer-2"
  }
}
```

说明：

- 在 `config.json` 中使用 `api_base_url` / `api_key`（不是 `baseUrl` / `apiKey`）
- `api_base_url` 主要用于提供商标识；SDK 调用不会对该 URL 发起 HTTP fetch
- 使用 `ccr model get cursor` 发现实时模型列表（见下文）

### 2. 重启

```bash
docker compose restart ccr
# 或
ccr restart
```

## 模式

通过转换器条目传参：`["cursor-sdk", { … }]`。

| 选项 | 默认 | 说明 |
|------|------|------|
| `cursorMode` | `"bridge"` | `bridge` — Claude Code 托管工具，拒绝 Cursor 内置工具。`plan` — 仅文本/推理。`agent` — Cursor agent 模式；可用 `cursorCwd`。 |
| `cursorCwd` | （会话工作区） | `cursorMode` 为 `agent` 时设置 SDK 本地 cwd。 |
| `sandboxEnabled` | `false` | 可选开启 Cursor 本地沙箱。Docker / 不支持的主机上强制关闭。也可在支持的桌面主机上设置 `CCR_CURSOR_SANDBOX=1`。 |

### Bridge 模式（推荐用于 Claude Code）

1. CCR 创建 / 恢复进程内 Cursor agent 会话
2. 将 Claude Code 请求中的主机工具注册为 SDK custom tools
3. Cursor 需要工具时，CCR 挂起调用并向 Claude Code 流式返回 OpenAI 风格的 `tool_calls`
4. Claude Code 执行工具并回传结果；CCR 解析挂起的 promise 并继续流
5. 隔离工作区中的 deny-hooks 阻止 Cursor 内置工具，文件系统/shell 仍由 Claude Code 负责

隔离工作区位于：

```text
~/.claude-code-router/cursor-sdk-workspaces/
```

#### 主机环境锚定

Cursor 的 harness 提示词由服务端根据 SDK 工作区根目录生成，因此模型会在**系统层**被告知：隔离工作区就是它的项目。当 CCR 运行在 Docker 中时，这一说法是自洽的（Linux、`/root/...`、空目录），而用户的真实项目位于 Claude Code 主机上。把系统上下文视为权威的模型可能据此认为自己被限制在工作区内，并开始给工具路径加上工作区前缀。

为避免这一点，bridge 模式会从每个请求中提取主机的 `<env>` 块（项目根目录、平台、系统版本，以及主机上报的其他字段），并在三处声明真实拓扑——*工具在另一台机器上执行*：

- 工作区内的 `AGENTS.md`，Cursor 会将其作为项目规则注入
- 发送给 agent 的提示词的开头与结尾
- Cursor 内置工具被拒绝时返回的消息

主机信息每轮都会重新读取，且只有当上报的环境确实发生变化时才重写工作区文件。Cursor 只在 agent 会话创建时加载一次工作区规则，因此重写会作用于该目录的下一个会话——当前这一轮始终通过提示词本身获得最新的主机信息。当请求不包含环境块时，CCR 不会猜测根目录，而是退回到要求模型只使用对话中出现过的绝对主机路径。

:::warning
当 CCR 运行在容器中时，不要把 `cursorCwd` 设置为主机项目路径。该路径不存在时会被创建，从而在容器内的真实项目路径上产生一个空的幽灵目录。
:::

#### 工作区路径检测

提示词只能起预防作用，因此 bridge 模式还会检查每次主机工具调用的参数。如果任何字符串参数引用了隔离工作区——包括出现在 shell 命令中，例如 `cd <workspace> && ls`——该调用**不会**转发给 Claude Code。模型会收到一条纠正性的工具结果，其中指出违规参数与真实的主机根目录，然后重试。

每个会话最多纠正三次；超过后调用将原样转发，因为持续坚持的模型可能是在执行用户关于该路径的明确要求。

当主机项目根目录本身位于隔离工作区根目录之下时，检测会被完全跳过。相关次数记入会话指标（`scratchPathViolations`、`scratchPathCorrections`），每次调用以 `warn` 级别记录，并在每轮结束时汇总：

```text
cursor-sdk turn produced scratch-workspace tool paths
```

检索该日志消息即可对比不同模型的行为——这一失效模式会影响严格遵循系统提示词的模型，而不影响 Cursor 原生模型。

#### 工作区生命周期

隔离工作区会在其会话释放时删除（空闲 TTL、LRU 淘汰或显式释放）。因崩溃或强制结束而残留的目录，会在超过 24 小时后由每小时一次的清理任务回收。两条路径都只会删除同时满足以下条件的目录：位于工作区根目录的直接子级，**且**名称为 32 位会话键，因此为 `agent` 模式提供的 `cursorCwd` 永远不会被删除。

### Plan / agent 模式

- **plan** — 规划/对话助手；不执行工具
- **agent** — Cursor agent，使用其自身的本地工具语义；若希望由 Claude Code 拥有工具，请优先使用 bridge

## 模型发现

Cursor 模型通过 `@cursor/sdk` 列出（不是 REST `/models`）：

```bash
ccr model get cursor
```

当提供商名为 `cursor`，或 `transformer.use` 包含 `cursor-sdk` 时，CCR 会识别为 Cursor 提供商。发现阶段的认证规则与服务器相同（`crsr_` / `CURSOR_API_KEY`）。

将模型写入 `config.json` 后请重启服务。

## 会话

Cursor 对话在进程内保持状态：

- 会话键来自 `x-ccr-cursor-session` 头、入站捕获的客户端会话 id（`protocolContext.sessionId` / `req.sessionId` / Claude `metadata.user_id`），或匿名 `hash(model + 首条 user 文本)` —— 不使用 system / harness 版本
- LRU 上限 **32**；空闲 TTL **15 分钟**
- 进行中的会话（活动流、running run、或已挂起工具）不会被空闲淘汰
- 流在中途失败时，若 agent 会话已有历史，下一次请求使用精简 follow-up，而不是重发全文

## Docker 运行

`@cursor/sdk` 含平台原生包，会在运行镜像中单独安装（版本取自 `packages/server/package.json`）。

确保容器能拿到密钥：

```yaml
environment:
  - CURSOR_API_KEY=${CURSOR_API_KEY}
```

即使配置了沙箱，Docker 内也会禁用。

## 转换器行为

`cursor-sdk` 转换器：

- 在进程内运行 `@cursor/sdk` Agent 的 create/send/stream
- 通过 `__providerResponse` 返回已就绪的 `Response`（跳过对提供商 URL 的 HTTP `fetch`）
- 向 AnthropicTransformer 发出 OpenAI chat.completion / chat.completion.chunk SSE
- 支持 Claude Code 的流式与非流式请求
- 在 SDK 可用时，将 effort / reasoning 字段映射到 SDK 模型选择
- 保持 Cursor 缓存在 SDK agent 会话内原生处理，同时从 SDK usage delta 向 Claude Code 暴露有界的 cache-read usage
- 从 `run.stream()` 的 `thinking` 消息以及 `Agent.send({ onDelta })` 的 `thinking-delta` 更新转发 Cursor thinking，并发出 Claude Code 所期望的 Anthropic 兼容 `signature_delta`
- 在 Claude Code 2.1.89+ 上，交互式显示需要客户端设置 `"showThinkingSummaries": true`；若未设置，CCR 仍会传输 thinking 块且 Claude Code 会持久化它，但交互 UI 会隐藏摘要
- 在 Anthropic 边界对 Claude Code 的尾部 turn 做一次分类，使用确切的协议标记而非提示文本正则，并将该意图保留在请求本地上下文中，而不会序列化到上游
- 通过一个有界、可重放的响应生产者合并相同的重叠重试，从而保证一个逻辑 turn 只存在一个 `agent.send` 与一个 Cursor 迭代消费者
- 将最后一个订阅者的 stop/interrupt 视为真实的 SDK 取消，并在可以创建替换会话之前等待有界的退役
- 仅当挂起的 run 仍然活跃、工具结果集合完全匹配且没有有意义的 steering 时才继续；被拒结果+替换文本、无匹配/死掉的 run、清理失败以及会话记录偏离，都会退役 agent 并重放完整会话记录
- 仅当下一条主机会话记录恰好是已提交的 assistant 文本/工具调用再加一条支持的 user 消息时，才会用精简提示复用空闲 agent；更大的后缀会完全重放，且从不会用 `local.force` 代替生命周期或会话记录对齐

## 用法

```json
{
  "Router": {
    "default": "cursor,composer-2",
    "think": "cursor,claude-opus-4-8",
    "background": "cursor,claude-haiku-4-5"
  }
}
```

## 故障排除

**找不到 Cursor API key**：将 `Providers[].api_key` 设为以 `crsr_` 开头的密钥，或导出 `CURSOR_API_KEY`。`$CURSOR_API_KEY` 这类占位符只有在环境变量真正被设置时才会生效。

**密钥前缀错误**：Cursor 控制台密钥以 `crsr_` 开头，不是 `sk-`。

**Node engines 错误**：本地安装 / 发布需要 Node **≥ 22.19.0**。

**`ccr model get cursor` 没有返回模型**：确认认证以及提供商使用了 `cursor-sdk`。写入模型后请重启。

**工具在 Cursor 内运行，而不是 Claude Code**：使用 `cursorMode: "bridge"`（默认），且不要启用会改变托管假设的不支持沙箱选项。

**负载下会话被释放 / 流中断**：会话有上限（32），且在非进行中时会在 15 分钟后被空闲淘汰。长对话请优先使用稳定的会话头。

## 相关文档

- [提供商配置](/docs/server/config/providers)
- [转换器配置](/docs/server/config/transformers)
- [模型发现](/docs/server/guides/model-discovery)
- [`ccr model get`](/docs/cli/commands/model-get)
