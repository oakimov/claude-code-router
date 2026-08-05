![](blog/images/claude-code-router-img.png)

[![](https://img.shields.io/badge/%F0%9F%87%AC%F0%9F%87%A7-English-000aff?style=flat)](README.md)
[![](https://img.shields.io/github/license/musistudio/claude-code-router)](https://github.com/musistudio/claude-code-router/blob/main/LICENSE)

Claude Code Router 是面向 Claude Code 的自适应 LLM 网关。它会将每个请求路由到最合适的模型和提供商，同时在不同 API 之间保留工具调用、流式输出、扩展思考和多轮上下文。

## ✨ 功能

- **自适应模型路由**：按场景路由请求，包括后台任务、思考、长上下文、网络搜索和图像工作流。
- **工具调用与思考**：在不同 API 格式的提供商之间保留工具调用、工具结果和推理内容。
- **多提供商支持**：支持多种模型提供商，如 OpenRouter、DeepSeek、Ollama、Gemini、Antigravity、Volcengine、SiliconFlow、Codex、Claude 订阅、Qwen、Chrome 端侧和 Cursor（SDK）。
- **请求/响应转换**：使用 transformers 为不同提供商定制请求和响应。
- **原生客户端协议**：通过同一路由器和回退管道接受 Anthropic Messages、OpenAI Chat Completions 和 OpenAI Responses 请求。
- **动态模型切换**：在 Claude Code 中使用 `/model` 命令即时切换模型。
- **CLI 模型管理**：使用 `ccr model` 直接在终端管理模型和提供商。
- **GitHub Actions 集成**：在 GitHub 工作流中触发 Claude Code 任务。
- **插件系统**：使用自定义 transformers 扩展功能。

## 🛠 本 fork 的改进

本 fork 基于 [claude-code-router](https://github.com/musistudio/claude-code-router)，包含以下增强功能：

- **改进的 LLM 支持**：修复了 Gemini/Gemma 的流式传输，增强了 OpenAI API 处理能力。
- **推理与流式重构**：将流式传输和推理逻辑模块化为可复用的工具，提升可维护性。
- **Mistral 集成**：添加了对 Mistral 推理参数的特殊处理，并解耦了转换逻辑。
- **构建与部署**：将 UI 包集成到 Docker 构建流程，并添加了 Docker Compose 配置。
- **代码质量**：代码库本地化（英文注释），改进错误处理，处理 Copilot 审查反馈。
- **Gemini 稳定性与工具调用修复**：修正了 Gemini 请求体中 `thoughtSignature` 的位置（Gemini 3 要求将其作为 `functionCall` 部分的同级字段，且仅验证每个步骤的第一个此类部分）；过滤了出站 Gemini 请求中的合成 `ccr_` 占位符签名，防止 Gemini 500 错误；修复了 Anthropic 转换器中 `tool_result` 内容数组序列化问题，使模型接收纯文本而非 JSON 包裹的数组（解决了 Claude Code 中的"Error editing file"问题）；修复了 Fastify `onSend` 钩子，防止错误响应时出现未处理的 `invalid type 'object'` 拒绝。
- **Codex (ChatGPT) 集成**：添加了用于 ChatGPT 后端 API (Responses API) 的 Codex 转换器，支持基于 OAuth 的身份验证（`ccr codex-auth`）和通过 `api_key: "at-..."` 的 PAT 认证，以及 SSE 流式传输、推理/思考内容、带网络搜索的工具调用和图片处理。
- **Cursor SDK 集成**：添加了 `cursor-sdk` 转换器，通过 `@cursor/sdk` 在进程中运行 Cursor 模型。默认的 **bridge** 模式保持 Claude Code 作为工具宿主（拒绝 Cursor 内置工具）；支持 `plan` / `agent` 模式、`crsr_` / `CURSOR_API_KEY` 身份验证、`ccr model get cursor` 模型发现，以及 Docker 运行时安装 SDK 原生包。
- **Claude 订阅集成**：添加了 `claude-auth` 支持，通过 OAuth (`ccr claude-auth`) 使用 Claude Pro 或 Max 订阅进行路由，使用 `claude-auth` + `Anthropic` 转换器链。
- **Antigravity 集成**：添加了 Google Antigravity OAuth 支持（`ccr antigravity-auth`），使用 `antigravity-auth` + `gemini` 转换器链，目标为 Antigravity / `cloudcode-pa` API。支持该额度下的 Gemini 和 Claude 模型、思路签名往返/回退，以及针对 Gemini 后端 Claude 模型的 Claude 工具模式清理。需要 `gemini` 选项 `{"cachedContent": false}`，因为 Antigravity 没有 Google `cachedContents` 资源（保留默认值 `true` 会导致 404）。
- **Qwen Chat 集成**：添加了 `qwen-auth` 转换器，用于 Qwen Chat 后端（`qwen.aikit.club/v1/chat/completions`），支持基于 JWT 的认证（`ccr qwen-auth`），用户粘贴从 `chat.qwen.ai` localStorage 复制的令牌，自动令牌轮换，以及去除 Qwen 注入到响应中的尾部 `<details>...</details>` 元数据块。
- **DeepSeek 推理重放**：为 DeepSeek 模型（例如通过 OpenCode/ZenGo）实现了强制推理重放。DeepSeek 要求在后续请求中包含前次助手的推理内容——`reasoning` 转换器会自动重放先前轮次的推理输出。
- **模型发现**：为任意 API 提供商启用了非交互式模型发现。使用 `ccr model get <provider>`，该工具自动获取远程模型，使用可配置路径解析自定义 JSON 结构，并将缺失的模型追加到本地配置，同时保留现有设置。
- **Chrome 端侧模型**：添加了用于 Chrome 内置 Gemini Nano（约 4GB 本地模型）的 `chrome-on-device` 转换器。通过桥接进程（`ccr chrome-bridge`）与 Chrome 的 Prompt API 通信（通过 CDP）。使用 `responseConstraint` 实现结构化 JSON 输出（工具调用 + 文本），支持流式和非流式，暴露与 OpenAI 兼容的 `/v1/chat/completions` 端点，并用最小化工具系统提示替换 Claude Code 的系统提示。零 API 成本，零外部提供商延迟。
  - **稳定性与提示**：在系统提示中实现了"OPERATIONAL OVERRIDE"以防止幻觉并强制遵守用户提供的路径。
  - **停滞恢复**：为大量空白内容添加了分层重试机制：如果模型停滞（发出 1000+ 空白字符），桥接器会中止并重新尝试（无约束、提高温度——动态温度缩放）。
  - **上下文感知**：为工具输出添加了标签（"Tool Result:"），并指示模型在再次调用工具前检查现有结果。

## 🚀 快速开始

### 1. 安装

#### 前置要求

开始之前，请确保系统已安装以下内容：
- **Docker**（推荐）：通过已发布的镜像运行路由器的首选方式。参见 [Docker 安装指南](https://docs.docker.com/get-docker/)。
- **Node.js**（可选）：从源码运行、发布包或使用 **Chrome 端侧**桥接时需要。本 fork 要求 **Node.js ≥ 22.19.0**（`undici` 需要）。参见 [Node.js 下载](https://nodejs.org/)。
- **Claude Code**：参见 [官方快速入门指南](https://code.claude.com/docs/en/quickstart) 获取安装说明。

#### 使用 Docker 快速启动

使用已发布的 Docker 镜像启动 Claude Code Router 是最快的方式：

```shell
mkdir -p ~/.claude-code-router
docker run -d --name ccr \
  -p 3456:3456 \
  -v ~/.claude-code-router:/root/.claude-code-router \
  ghcr.io/oakimov/claude-code-router:latest
```

镜像自带服务器和 UI，在 `http://localhost:3456` 暴露代理，并将 `~/.claude-code-router` 挂载为配置目录（容器内的 `/root/.claude-code-router`）。请在 `config.json` 中设置 `"HOST": "0.0.0.0"`，以便端口映射能够访问服务器。修改配置后使用 `docker restart ccr` 重启；使用 `docker logs -f ccr` 查看日志。

### 2. 配置

创建并配置您的 `~/.claude-code-router/config.json` 文件。更多细节可参考 `config.example.json`。

`config.json` 包含几个关键部分：

- **`PROXY_URL`**（可选）：可以为 API 请求设置代理，例如：`"PROXY_URL": "http://127.0.0.1:7890"`。回环目标（`localhost`、`127.0.0.1`、`::1`）始终绕过代理。`NO_PROXY` / `no_proxy` 环境变量中列出的主机也会绕过代理（逗号分隔的主机、`.domain` / `*.domain`、CIDR、可选的 `:port`）。
- **`LOG`**（可选）：设为 `true` 可启用日志。设为 `false` 时不创建日志文件。默认值为 `true`。
- **`LOG_LEVEL`**（可选）：设置日志级别。可用选项：`"fatal"`、`"error"`、`"warn"`、`"info"`、`"debug"`、`"trace"`。默认值为 `"debug"`。
- **日志系统**：Claude Code Router 使用两套独立的日志系统：
  - **服务器级日志**：使用 pino 将 HTTP 请求、API 调用和服务器事件记录在 `~/.claude-code-router/logs/` 目录下，文件名形如 `ccr-*.log`
  - **应用级日志**：路由决策和业务逻辑事件记录在 `~/.claude-code-router/claude-code-router.log`
- **`APIKEY`**（可选）：可以设置密钥来验证请求。API 客户端可在 `Authorization` 头（例如 `Bearer your-secret-key`）或 `x-api-key` 头中提供。Web UI 会将其换成一个不透明的 `HttpOnly` 同站会话 cookie，绝不会把密钥存储到浏览器存储中。UI 会话保存在内存中，CCR 重启后需要重新登录。示例：`"APIKEY": "your-secret-key"`。
- **`HOST`**（可选）：可以设置服务器的主机地址。如果未设置 `APIKEY`，出于安全原因主机将被强制为 `127.0.0.1`，以防止未经授权的访问。示例：`"HOST": "0.0.0.0"`。
- **速率限制**：每条路由默认限制为每分钟 1000 个请求。共享默认值由 `packages/shared/src/constants.ts` 中的 `RATE_LIMIT_CONFIG` 定义；在那里修改可更新所有路由的限制。
- **`NON_INTERACTIVE_MODE`**（可选）：设为 `true` 时，启用与 GitHub Actions、Docker 容器或其他 CI/CD 系统等非交互环境的兼容性。它会设置适当的环境变量（`CI=true`、`FORCE_COLOR=0` 等）并配置 stdin 处理，防止进程在自动化环境中挂起。示例：`"NON_INTERACTIVE_MODE": true`。

- **`Providers`**：用于配置不同的模型提供商。
- **`Router`**：用于设置路由规则。`default` 指定默认模型，如果没有配置其他路由，所有请求都会使用它。
- **`API_TIMEOUT_MS`**：指定 API 调用的超时时间（毫秒）。

#### 环境变量插值

Claude Code Router 支持环境变量插值，用于安全地管理 API 密钥。您可以在 `config.json` 中使用 `$VAR_NAME` 或 `${VAR_NAME}` 语法引用环境变量：

```json
{
  "OPENAI_API_KEY": "$OPENAI_API_KEY",
  "GEMINI_API_KEY": "${GEMINI_API_KEY}",
  "Providers": [
    {
      "name": "openai",
      "api_base_url": "https://api.openai.com/v1/chat/completions",
      "api_key": "$OPENAI_API_KEY",
      "models": ["gpt-5", "gpt-5-mini"]
    }
  ]
}
```

这样可以将敏感 API 密钥保存在环境变量中，而不是硬编码在配置文件中。插值会递归处理嵌套对象和数组。

以下是一个最小配置示例：

```json
{
  "APIKEY": "your-secret-key",
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "$OPENROUTER_API_KEY",
      "models": ["anthropic/claude-sonnet-4"],
      "transformer": {
        "use": ["openrouter"]
      }
    }
  ],
  "Router": {
    "default": "openrouter,anthropic/claude-sonnet-4"
  }
}
```

> **另请参阅**：完整的配置参考、各提供商示例和路由选项位于 `docs/docs/server/config/basic.md`、`docs/docs/server/config/providers.md`、`docs/docs/server/config/transformers.md` 和 `docs/docs/server/config/routing.md`。

#### 添加新提供商

如果要添加新提供商并自动发现其模型，请按以下步骤操作：

1. **添加最小配置**：在 `config.json` 的 `Providers` 数组中添加一个只包含基本信息的新条目：
   ```json
   {
     "name": "my-new-provider",
     "api_base_url": "https://api.example.com/v1/chat/completions",
     "api_key": "$MY_API_KEY",
     "models": []
   }
   ```
2. **执行模型发现**：运行发现命令以获取可用模型：
   ```shell
   ccr model get my-new-provider
   ```
3. **同步模型**：该命令会列出远程模型并提示您将缺失的模型追加到配置中。
4. **重启**：重启服务以加载更新后的配置：
   ```shell
   ccr restart
   ```

> **提示**：关于模型发现选项、自定义 JSON 响应格式和交互式模型管理的更完整说明，请参阅[CLI 模型管理](#5-cli-模型管理)部分。

### 3. 使用 Router 运行 Claude Code

#### 通过 `ccr code`

使用路由器启动 Claude Code：

```shell
ccr code
```

#### 通过 Claude Code 设置（替代方案）

您也可以编辑 `settings.json` 文件（通常在 `~/.claude/settings.json`）来让 Claude Code 始终使用路由器。模型使用 `<provider>,<model>` 语法指定：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:3456",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gemini,gemma-4-31b-it",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "opencode,minimax-m2.7",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "opencode,glm-5.1"
  }
}
```

| 变量 | 用途 |
|---|---|
| `ANTHROPIC_BASE_URL` | 将 Claude Code 指向路由器的代理地址 |
| `ANTHROPIC_AUTH_TOKEN` | 必须与路由器 `config.json` 中的 `APIKEY` 值匹配 |
| `ANTHROPIC_MODEL` | 默认模型（覆盖以下各层级的默认值） |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | 快速 / 高性价比模型（相当于 Haiku） |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | 均衡性能模型（相当于 Sonnet） |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | 最强能力模型（相当于 Opus） |

这样您可以直接运行 `claude`，而无需使用 `ccr code`。

#### 通过 OpenAI 兼容客户端

OpenAI SDK 客户端无需 Anthropic 兼容层即可使用 CCR：

```shell
export OPENAI_BASE_URL=http://127.0.0.1:3456/v1
export OPENAI_API_KEY=your-router-api-key
```

CCR 在 `/v1/chat/completions`（别名 `/chat/completions`）接受 Chat Completions，在 `/v1/responses`（别名 `/responses`）接受 Responses。发送 `provider,model` 格式的模型可显式选择目标，或发送裸模型并配置 `Router.default`。JSON 和 SSE 响应都会转换回客户端所用的协议。

兼容层支持普通文本、图像、函数工具/结果、推理强度和用量上报。有状态 Responses 功能（如 `store: true`、`previous_response_id`、会话、后台模式、提供商文件 ID 和不支持的托管工具）会返回显式 400 错误，而不是被静默丢弃。

> **注意**：修改配置文件后，需要重启服务才能生效：
>
> ```shell
> ccr restart
> ```

### 4. UI 模式

如需更直观的体验，可以使用 UI 模式管理配置：

```shell
ccr ui
```

这会打开一个基于 Web 的界面，您可以在其中轻松查看和编辑 `config.json` 文件。

![UI](blog/images/ui.png)

### 5. CLI 模型管理

对于偏好终端工作流的用户，可以使用交互式 CLI 模型选择器：

```shell
ccr model
```
![](blog/images/models.gif)

`ccr model` 让您查看已配置的模型、按场景切换模型、添加模型以及创建带 transformer 配置的提供商 — 全部带有校验和提示。

如需对任何拥有模型列表端点的提供商进行非交互式模型发现：

```shell
ccr model get claude
ccr model get gemini
ccr model get openai
```

`ccr model get <provider>` 会获取远程模型，然后提示追加缺失的模型并移除 API 不再返回的已配置模型。内置端点支持 `anthropic`/`claude`、`gemini`、`openai`、`codex` 和 `cursor`。其他提供商可以使用 `models_api_url` 加上 `models_response_format`（`listPath`、`idPath`、`stripPrefix`）来解析自定义 JSON 响应。

> **另请参阅**：`docs/docs/server/guides/model-discovery.md` 和 `docs/docs/cli/commands/model-get.md`。
>
> **注意**：将模型同步到 `config.json` 后，请使用 `ccr restart` 重启服务。

#### Antigravity 身份验证

使用账户 OAuth 而非 API 密钥，将 Claude Code 路由到 Google Antigravity 网关（`cloudcode-pa`）。

```shell
ccr antigravity-auth
# 或无头/远程模式：
ccr antigravity-auth --manual
ccr antigravity-auth --project <gcp-project-id>
```

此命令：
1. 打印一个 Google OAuth URL（PKCE；回调 `http://localhost:51121/oauth-callback`）
2. CCR 服务器处理公开的 `/oauth-callback` 路由（Docker 映射 `51121:3456`，与 Codex 的 `1455:3456` 类似）
3. Token 保存到 `~/.claude-code-router/antigravity_auth.json`（Docker 中为挂载的配置目录）

示例提供商：

```json
{
  "name": "antigravity",
  "api_base_url": "https://daily-cloudcode-pa.sandbox.googleapis.com",
  "api_key": "oauth",
  "project_id": "$ANTIGRAVITY_PROJECT_ID",
  "models": ["gemini-3-flash", "claude-sonnet-4-6", "claude-opus-4-6-thinking"],
  "transformer": {
    "use": [
      ["gemini", { "cachedContent": false, "thoughtSignatureFallback": "skip" }],
      "antigravity-auth"
    ]
  }
}
```

必须设置 `cachedContent: false` — Antigravity 没有 Google 的 `cachedContents` 资源，保留 Gemini 默认值（`true`）会导致 404。`thoughtSignatureFallback` 选项在 `docs/docs/server/config/transformers.md` 中有说明。

> **注意**：身份验证期间请保持 CCR 服务器运行。在非 IDE 客户端中使用 Antigravity IDE OAuth 客户端凭据可能违反 Google 的服务条款。

#### Codex 提供商身份验证

Codex 提供商支持两种身份验证模式：

- **OAuth**：通过 `ccr codex-auth` — 浏览器流程由 CCR 服务器在端口 `1455` 上处理回调（Docker 映射 `1455:3456`）；token 存储在 `~/.claude-code-router/codex_auth.json` 中并自动刷新。
- **PAT**：通过在 `api_key` 中直接填入 `"at-..."`（或包含 `at-` token 的环境变量）— 跳过 `ccr codex-auth`。

`at-` 值始终被视为 PAT，绝不会静默回退到 OAuth；任何其他占位值都会选择 OAuth token。

> **另请参阅**：完整的 Codex 设置、两种身份验证模式、提供商配置和故障排查见 `docs/docs/server/guides/codex.md`。

#### Cursor 提供商身份验证

Cursor 提供商通过官方 `@cursor/sdk` 在进程内运行模型（无浏览器 OAuth CLI）。认证从提供商 `api_key`（以 `crsr_` 开头）解析，然后是 `CURSOR_API_KEY`。默认 **bridge** 模式让 Claude Code 作为工具宿主；Cursor 内置工具在隔离工作区中被拒绝。使用 `ccr model get cursor` 发现模型。

> **另请参阅**：完整的 Cursor 设置、bridge/plan/agent 模式和配置见 `docs/docs/server/guides/cursor.md`。

#### Claude 订阅身份验证

通过 OAuth 将 Claude Code 路由到您的 Claude Pro 或 Max 订阅：

```shell
ccr claude-auth
```

OAuth 流程由 CCR 服务器在端口 `1455` 上处理回调（Docker 映射 `1455:3456`）；token 存储在 `~/.claude-code-router/claude_auth.json` 中并自动刷新。Claude 订阅提供商需要 `claude-auth` + `Anthropic` transformer 链。

> **另请参阅**：完整的设置、transformer 链、客户端分类和计费/身份细节见 `docs/docs/server/guides/claude-auth.md`。

#### Qwen 提供商身份验证

通过保存从 `chat.qwen.ai` localStorage 获取的 JWT 来与 Qwen Chat 认证：

```shell
ccr qwen-auth
```

CCR 服务器在 `/qwen/auth` 提供一个认证页面，支持小书签或手动粘贴。token 会被校验、保存到 `~/.claude-code-router/qwen_auth.json` 并自动刷新。Qwen 提供商需要 `qwen-auth` + `reasoning` + `OpenAI` transformer 链。

> **另请参阅**：完整的 Qwen 设置和提供商配置见 `docs/docs/server/guides/qwen.md`。

#### Chrome 端侧桥接

通过宿主机侧的桥接进程使用 Chrome 内置的 Gemini Nano（约 4GB 本地模型），零 API 成本：

```bash
ccr chrome-bridge            # 默认：端口 3457，CDP 9222
ccr chrome-bridge --port 3457 --cdp 9222
```

桥接通过 CDP 连接 Chrome 的 Prompt API，维护持久的模型会话，并在 `127.0.0.1:3457` 暴露 OpenAI 兼容 API（`/v1/chat/completions`）。它用极简的面向工具的提示词替换 Claude Code 的系统提示词，并使用 `responseConstraint`（JSON Schema）强制输出结构化的 `{text, tool_calls[]}`。

**前置要求**：启用 `chrome://flags/#optimization-guide-on-device-model` 和 `chrome://flags/#prompt-api-for-gemini-nano-multimodal-input`，重启 Chrome，并等待约 4GB 模型下载完成。

> **Docker 注意**：桥接必须运行在 Docker **宿主机**上，而不是容器内 — 请将提供商主机设置为 `http://host.docker.internal:3457`。完整的设置、提供商配置、特性和限制见 `docs/docs/server/guides/chrome-on-device.md`。

### 6. 预设管理

保存、共享和复用配置：

```shell
ccr preset export my-preset                    # 将当前配置导出为预设
ccr preset export my-preset --description "My OpenAI config" --author "Your Name" --tags "openai,production"
ccr preset install /path/to/preset             # 从目录安装
ccr preset list
ccr preset info my-preset
ccr preset delete my-preset
```

预设将您的配置（以及元数据）作为一个目录存储，其中包含 `manifest.json`。导出时敏感字段会被清理为 `{{field}}` 占位符，预设还可以包含输入 schema，以便在安装时收集所需的值（例如 API 密钥）。

> **另请参阅**：`docs/docs/cli/commands/preset.md`。

### 7. Activate 命令（环境变量设置）

`activate` 命令允许您在 shell 中全局设置环境变量，从而可以直接使用 `claude` 命令，或将 Claude Code Router 与基于 Agent SDK 构建的应用程序集成。

要激活环境变量，请运行：

```shell
eval "$(ccr activate)"
```

此命令会以 shell 友好的格式输出所需的环境变量，并在当前 shell 会话中设置。激活后，您可以：

- **直接使用 `claude` 命令**：无需使用 `ccr code` 即可运行 `claude` 命令。`claude` 命令会自动将请求路由到 Claude Code Router。
- **与 Agent SDK 应用程序集成**：基于 Anthropic Agent SDK 构建的应用程序会自动使用已配置的路由器和模型。

`activate` 命令设置以下环境变量：

- `ANTHROPIC_AUTH_TOKEN`：配置中的 API 密钥
- `ANTHROPIC_BASE_URL`：本地路由器端点（默认：`http://127.0.0.1:3456`）
- `NO_PROXY`：设为 `127.0.0.1` 以防止代理干扰
- `DISABLE_TELEMETRY`：禁用遥测
- `DISABLE_COST_WARNINGS`：禁用成本警告
- `API_TIMEOUT_MS`：配置中的 API 超时

> **注意**：使用激活的环境变量前，请确保 Claude Code Router 服务正在运行（`ccr start`）。环境变量仅在当前 shell 会话中有效。要使其持久化，可以将 `eval "$(ccr activate)"` 添加到 shell 配置文件中（例如 `~/.zshrc` 或 `~/.bashrc`）。

#### Providers 与 Transformers

`Providers` 数组定义每个提供商：`name`、`api_base_url`、`api_key`、`models`，以及可选的 `transformer` 对象。`transformer.use` 列表全局（所有模型）或按模型键应用 transformers，某些 transformers 通过嵌套的 `[name, options]` 数组接受选项。

> **另请参阅**：提供商 schema 和各提供商示例见 `docs/docs/server/config/providers.md`；transformer 参考和选项传递见 `docs/docs/server/config/transformers.md`。

**内置 Transformers：**

- `Anthropic` — 原样透传到 Anthropic 端点。`OpenAI` — 注册 `/v1/chat/completions` 路由（请求体已是 OpenAI 格式）。
- 提供商适配器：`deepseek`、`groq`、`mistral`、`openrouter`、`gemini` / `vertex-gemini`、`codex`、`claude-auth`、`antigravity-auth`、`qwen-auth`、`cursor-sdk`、`chrome-on-device`。
- `maxtoken` — 设置特定的 `max_tokens`。`tooluse` — 通过 `tool_choice` 优化工具使用。`reasoning` — 跨轮次回放提供商的 `reasoning_content`。`sampling` — 映射 `temperature` / `top_p` / `top_k` / `repetition_penalty`。`enhancetool` — 为工具调用参数添加容错层（会使工具调用信息不再流式传输）。`cleancache` — 清除 `cache_control`。`customparams` — 注入自定义请求参数。
- 实验性 gist/CLI 集成：`gemini-cli`、`chutes-glm`、`qwen-cli`、`rovo-cli`。

> **另请参阅**：完整的 transformer 参考 — 包括 `gemini` 的 `cachedContent` / `thoughtSignatureFallback` 选项和 `openrouter` 的提供商路由参数 — 见 `docs/docs/server/config/transformers.md`。

**自定义 Transformers：**

通过 `config.json` 中的 `transformers` 字段加载您自己的 transformers，例如 `{ "transformers": [{ "path": "/User/xxx/.claude-code-router/plugins/gemini-cli.js", "options": { "project": "xxx" } }] }`。完整的自定义 transformer 指南见 `docs/docs/server/config/transformers.md`。

#### Router

`Router` 对象定义不同场景使用的模型：

- `default`：常规任务的默认模型。
- `background`：后台任务的模型。可以使用更小、本地的模型来节省成本。
- `think`：推理密集型任务的模型，如 Plan 模式。
- `longContext`：处理长上下文的模型（例如 > 60K tokens）。
- `longContextThreshold`（可选）：触发长上下文模型的 token 数阈值。未指定时默认为 60000。
- `webSearch`：用于处理网络搜索任务，要求模型本身支持该功能。如果使用 openrouter，需要在模型名后添加 `:online` 后缀。
- `image`（beta）：用于处理与图像相关的任务（由 CCR 的内置 agent 支持）。如果模型不支持工具调用，需要将 `config.forceUseImageAgent` 属性设为 `true`。

- 您还可以在 Claude Code 中使用 `/model` 命令动态切换模型：
`/model provider_name,model_name`
示例：`/model openrouter,anthropic/claude-3.5-sonnet`

#### 自定义 Router

如需更高级的路由逻辑，可在 `config.json` 中设置 `CUSTOM_ROUTER_PATH`，指向一个导出 `async function(req, config)` 的 JS 模块，该函数返回 `"provider,model"` 字符串，或返回 `null` 以回退到默认路由器。参见 `custom-router.example.js` 和 `docs/docs/server/advanced/custom-router.md`。

##### 子代理路由

Claude Code 子代理请求可通过以下方式路由：

1. 在系统或消息文本中使用显式的 `<CCR-SUBAGENT-MODEL>provider,model</CCR-SUBAGENT-MODEL>` 标签（标签在上游请求前会被剥离）。也接受 `provider/model` 格式。
2. 或在 Claude Code 将请求标记为子代理时，使用环境变量 `CLAUDE_CODE_SUBAGENT_MODEL=provider,model`。

标签优先于环境变量。如果两者都未设置，则应用常规 Router 规则。

**示例：**

```
<CCR-SUBAGENT-MODEL>openrouter,anthropic/claude-3.5-sonnet</CCR-SUBAGENT-MODEL>
请帮我分析这段代码片段的潜在优化点...
```

```bash
export CLAUDE_CODE_SUBAGENT_MODEL="openrouter,anthropic/claude-3.5-sonnet"
```

## 提示缓存

CCR 会自动将 Claude Code 的 Anthropic 缓存意图转换为每个上游的原生缓存机制：

- Anthropic、Claude Auth 和 Vertex Claude 使用 Anthropic 自动提示缓存，同时保留有界的显式块标记。
- OpenAI Chat 和 Responses 使用稳定的 `prompt_cache_key` 值；GPT-5.6+ 模型还会收到显式的内容断点。Codex 对每个模型使用其独立的原生契约：稳定的提示键加会话路由头，无显式内容断点。
- OpenRouter 使用粘性 `session_id` 路由加模型原生缓存。Vercel AI Gateway 接收 `providerOptions.gateway.caching: "auto"`。
- Mistral 和 Cerebras 接收原生 `prompt_cache_key` 值。Qwen/DashScope 接收最终内容缓存标记。DeepSeek、Groq 和 Vertex OpenAI 保留其原生隐式缓存。
- Gemini 和 Vertex Gemini 使用隐式缓存，并为足够大的稳定系统/工具前缀创建/复用原生 CachedContent 资源。
- Cursor SDK 和 Chrome 端侧复用稳定的原生会话。

不兼容提供商的缓存字段只在转换后移除。上游上报的缓存读/写用量会转换回 Anthropic 的 `cache_read_input_tokens` 和 `cache_creation_input_tokens`，供 Claude Code 使用。

参见[实现计划与评审](tasks/caching-plan.md)了解提供商矩阵和验证范围。

## Status Line（Beta）

为了在运行时更好地监控 claude-code-router 的状态，v1.0.40 版本内置了 statusline 工具，您可以在 UI 中启用。
![statusline-config.png](blog/images/statusline-config.png)

效果如下：
![statusline](blog/images/statusline.png)

## 🤖 GitHub Actions

将 Claude Code Router 集成到您的 CI/CD 管道中。设置好 [Claude Code Actions](https://docs.anthropic.com/en/docs/claude-code/github-actions) 后，修改 `.github/workflows/claude.yaml` 以使用路由器：

```yaml
name: Claude Code

on:
  issue_comment:
    types: [created]
  # ... 其他触发器

jobs:
  claude:
    if: |
      (github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude')) ||
      # ... 其他条件
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: read
      issues: read
      id-token: write
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Prepare Environment
        run: |
          curl -fsSL https://bun.sh/install | bash
          mkdir -p $HOME/.claude-code-router
          cat << 'EOF' > $HOME/.claude-code-router/config.json
          {
            "log": true,
            "NON_INTERACTIVE_MODE": true,
            "OPENAI_API_KEY": "${{ secrets.OPENAI_API_KEY }}",
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "OPENAI_MODEL": "deepseek-chat"
          }
          EOF
        shell: bash

      - name: Start Claude Code Router
        run: |
          nohup ~/.bun/bin/bunx @caeliq/claude-code-router@1.0.8 start &
        shell: bash

      - name: Run Claude Code
        id: claude
        uses: anthropics/claude-code-action@beta
        env:
          ANTHROPIC_BASE_URL: http://localhost:3456
        with:
          anthropic_api_key: "any-string-is-ok"
```

> **注意**：在 GitHub Actions 或其他自动化环境中运行时，请确保在配置中设置 `"NON_INTERACTIVE_MODE": true`，以防止进程因 stdin 处理问题而挂起。

此设置支持有趣的自动化场景，例如在非高峰时段运行任务以降低 API 成本。

## 📝 延伸阅读

- [Codex API](https://developers.openai.com/codex/sdk) — `codex` transformer 使用的 ChatGPT 后端 API 的开发者文档（OAuth PKCE、Responses API、流式输出、工具调用）
- [Chrome Prompt API](https://developer.chrome.com/docs/ai/prompt-api) — `chrome-on-device` transformer 和桥接使用的端侧 Gemini Nano API
- [Provider 集成经验](tasks/lessons.md) — LLM 提供商集成的宝贵经验（DeepSeek、Mistral、Gemini、Codex、Gemini Nano）
