![](blog/images/claude-code-router-img.png)

[![](https://img.shields.io/badge/%F0%9F%87%AC%F0%9F%87%A7-English-000aff?style=flat)](README.md)
[![](https://img.shields.io/github/license/musistudio/claude-code-router)](https://github.com/musistudio/claude-code-router/blob/main/LICENSE)

## ✨ 功能

- **模型路由**：根据需求将请求路由到不同模型（例如，后台任务、思考、长上下文）。
- **多提供商支持**：支持多种模型提供商，如 OpenRouter、DeepSeek、Ollama、Gemini、Antigravity、Volcengine、SiliconFlow、Codex、Claude 订阅、Qwen、Chrome 端侧模型以及 Cursor (SDK)。
- **请求/响应转换**：使用转换器（transformer）为不同提供商定制请求和响应。
- **动态模型切换**：在 Claude Code 中使用 `/model` 命令动态切换模型。
- **CLI 模型管理**：通过 `ccr model` 直接从终端管理模型和提供商。
- **GitHub Actions 集成**：在 GitHub 工作流中触发 Claude Code 任务。
- **插件系统**：通过自定义转换器扩展功能。

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

开始前，请确保系统已安装以下组件：
- **Docker 和 Docker Compose**（推荐）：运行路由器的主要方式。参见 [Docker 安装指南](https://docs.docker.com/get-docker/)。
- **Node.js**（可选）：从源码运行、发布包或使用 **Chrome 端侧桥接** 时需要。本 fork 需要 **Node.js ≥ 22.13.0**（`@cursor/sdk` 所需）。参见 [Node.js 下载](https://nodejs.org/)。
- **Claude Code**：参见[官方快速开始指南](https://code.claude.com/docs/en/quickstart)获取安装说明。

#### 使用 Docker 快速启动

启动 Claude Code Router 最快的方式是使用 Docker Compose：

```shell
cd packages/server
docker compose up --build -d
```

Compose 设置将服务器和 UI 构建到 `ccr` 容器中，将代理暴露在 `http://localhost:3456`，并将 `packages/server/ccr-config` 挂载到容器内的 `/root/.claude-code-router`。

### 2. 配置

创建并配置您的 `~/.claude-code-router/config.json` 文件。更多详情可参考 `config.example.json`。

`config.json` 文件包含几个关键部分：

- **`PROXY_URL`**（可选）：您可以为 API 请求设置代理，例如：`"PROXY_URL": "http://127.0.0.1:7890"`。回环地址（`localhost`、`127.0.0.1`、`::1`）始终绕过代理；环境变量 `NO_PROXY` / `no_proxy` 中列出的主机也会绕过代理（逗号分隔，支持 `.domain` / `*.domain`、CIDR、可选 `:port`）。
- **`LOG`**（可选）：您可以设置 `true` 启用日志记录。设为 `false` 时不创建日志文件。默认值为 `true`。
- **`LOG_LEVEL`**（可选）：设置日志级别。可用选项：`"fatal"`、`"error"`、`"warn"`、`"info"`、`"debug"`、`"trace"`。默认值为 `"debug"`。
- **日志系统**：Claude Code Router 使用两个独立的日志系统：
  - **服务器级日志**：HTTP 请求、API 调用和服务器事件使用 pino 记录在 `~/.claude-code-router/logs/` 目录，文件名如 `ccr-*.log`
  - **应用级日志**：路由决策和业务逻辑事件记录在 `~/.claude-code-router/claude-code-router.log`
- **`APIKEY`**（可选）：您可以设置密钥用于请求认证。设置后，客户端必须在 `Authorization` 请求头（例如 `Bearer your-secret-key`）或 `x-api-key` 请求头中提供此密钥。示例：`"APIKEY": "your-secret-key"`。
- **`HOST`**（可选）：您可以设置服务器的主机地址。如果未设置 `APIKEY`，出于安全考虑，主机将强制为 `127.0.0.1`，以防止未经授权的访问。示例：`"HOST": "0.0.0.0"`。
- **`NON_INTERACTIVE_MODE`**（可选）：设为 `true` 时，启用与非交互式环境（如 GitHub Actions、Docker 容器或其他 CI/CD 系统）的兼容性。这会设置适当的环境变量（`CI=true`、`FORCE_COLOR=0` 等）并配置 stdin 处理，防止进程在自动化环境中挂起。示例：`"NON_INTERACTIVE_MODE": true`。

- **`Providers`**：用于配置不同的模型提供商。
- **`Router`**：用于设置路由规则。`default` 指定默认模型，若未配置其他路由，则用于所有请求。
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

以下是一个综合示例：

```json
{
  "APIKEY": "your-secret-key",
  "PROXY_URL": "http://127.0.0.1:7890",
  "LOG": true,
  "API_TIMEOUT_MS": 600000,
  "NON_INTERACTIVE_MODE": false,
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "sk-xxx",
      "models": [
        "google/gemini-2.5-pro-preview",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.7-sonnet:thinking"
      ],
      "transformer": {
        "use": ["openrouter"]
      }
    },
    {
      "name": "deepseek",
      "api_base_url": "https://api.deepseek.com/chat/completions",
      "api_key": "sk-xxx",
      "models": ["deepseek-chat", "deepseek-reasoner"],
      "transformer": {
        "use": ["deepseek"],
        "deepseek-chat": {
          "use": ["tooluse"]
        }
      }
    },
    {
      "name": "ollama",
      "api_base_url": "http://localhost:11434/v1/chat/completions",
      "api_key": "ollama",
      "models": ["qwen2.5-coder:latest"]
    },
    {
      "name": "gemini",
      "api_base_url": "https://generativelanguage.googleapis.com/v1beta/models/",
      "api_key": "sk-xxx",
      "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemma-4-31b-it"],
      "transformer": {
        "use": ["gemini"]
      }
    },
    {
      "name": "volcengine",
      "api_base_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
      "api_key": "sk-xxx",
      "models": ["deepseek-v3-250324", "deepseek-r1-250528"],
      "transformer": {
        "use": ["deepseek"]
      }
    },
    {
      "name": "modelscope",
      "api_base_url": "https://api-inference.modelscope.cn/v1/chat/completions",
      "api_key": "",
      "models": ["Qwen/Qwen3-Coder-480B-A35B-Instruct", "Qwen/Qwen3-235B-A22B-Thinking-2507"],
      "transformer": {
        "use": [
          [
            "maxtoken",
            {
              "max_tokens": 65536
            }
          ],
          "enhancetool"
        ],
        "Qwen/Qwen3-235B-A22B-Thinking-2507": {
          "use": ["reasoning"]
        }
      }
    },
    {
      "name": "dashscope",
      "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
      "api_key": "",
      "models": ["qwen3-coder-plus"],
      "transformer": {
        "use": [
          [
            "maxtoken",
            {
              "max_tokens": 65536
            }
          ],
          "enhancetool"
        ]
      }
    },
    {
      "name": "aihubmix",
      "api_base_url": "https://aihubmix.com/v1/chat/completions",
      "api_key": "sk-",
      "models": [
        "glm-4.5",
        "claude-opus-4-20250514",
        "gemini-2.5-pro"
      ]
    }
  ],
  "Router": {
    "default": "deepseek,deepseek-chat",
    "background": "ollama,qwen2.5-coder:latest",
    "think": "deepseek,deepseek-reasoner",
    "longContext": "openrouter,google/gemini-2.5-pro-preview",
    "longContextThreshold": 60000,
    "webSearch": "gemini,gemini-2.5-flash"
  }
}
```

#### 添加新提供商

如果要添加新提供商并自动发现其模型，请按以下步骤操作：

1. **添加最小配置**：在 `config.json` 的 `Providers` 数组中添加一个新条目，仅包含基本信息：
   ```json
   {
     "name": "my-new-provider",
     "api_base_url": "https://api.example.com/v1/chat/completions",
     "api_key": "$MY_API_KEY",
     "models": []
   }
   ```
2. **执行模型发现**：运行发现命令获取可用模型：
   ```shell
   ccr model get my-new-provider
   ```
3. **同步模型**：该命令会列出远程模型，并提示您将缺失的模型追加到配置中。
4. **重启**：重启服务以加载更新后的配置：
   ```shell
   ccr restart
   ```

> **提示**：关于模型发现选项、自定义 JSON 响应格式和交互式模型管理的更详细说明，请参阅 [CLI 模型管理](#5-cli-模型管理) 部分。

### 3. 使用 Router 运行 Claude Code

#### 通过 `ccr code`

使用路由器启动 Claude Code：

```shell
ccr code
```

#### 通过 Claude Code 设置（替代方案）

您也可以通过编辑 Claude Code 的 `settings.json` 文件（通常位于 `~/.claude/settings.json`），配置 Claude Code 始终使用路由器。模型使用 `<provider>,<model>` 语法指定：

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
| `ANTHROPIC_MODEL` | 默认模型（覆盖以下各层级默认值） |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | 快速/经济模型（相当于 Haiku） |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | 平衡性能模型（相当于 Sonnet） |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | 最大能力模型（相当于 Opus） |

此方式允许直接运行 `claude` 而无需使用 `ccr code`。

> **注意**：修改配置文件后，需要重启服务使更改生效：
>
> ```shell
> ccr restart
> ```

### 4. UI 模式

为获得更直观的体验，您可以使用 UI 模式管理配置：

```shell
ccr ui
```

这将打开一个基于 Web 的界面，可轻松查看和编辑您的 `config.json` 文件。

![UI](/blog/images/ui.png)

### 5. CLI 模型管理

对于偏好终端工作流的用户，可以使用交互式 CLI 模型选择器：

```shell
ccr model
```
![](blog/images/models.gif)

该命令提供交互式界面，用于：

- 查看当前配置
- 查看所有配置的模型（default、background、think、longContext、webSearch、image）
- 切换模型：快速更改每个路由器类型使用的模型
- 添加新模型：向现有提供商添加模型
- 创建新提供商：设置完整的提供商配置，包括：
   - 提供商名称和 API 端点
   - API 密钥
   - 可用模型
   - 转换器配置，支持：
     - 多个转换器（openrouter、deepseek、gemini 等）
     - 转换器选项（例如，带自定义限制的 maxtoken）
     - 特定提供商的路由（例如，OpenRouter 提供商偏好）

CLI 工具会验证所有输入，并提供有用的提示以引导您完成配置过程，轻松管理复杂设置，无需手动编辑 JSON 文件。

对于非交互式模型发现，您还可以直接测试提供商访问并列出远程模型：

```shell
ccr model get claude
ccr model get gemini
ccr model get openai
```

该命令：
- 使用配置的 API 密钥调用提供商的模型列表端点
- 打印提供商返回的远程模型
- 提示追加缺失的模型并移除 API 不再返回的已配置模型

内置端点支持包括 `anthropic`/`claude`、`gemini`、`openai`、`codex` 和 `cursor`。对于使用 `claude-auth` 的 Claude 订阅提供商，发现过程会读取 `~/.claude-code-router/claude_auth.json` 并发送所需的 Anthropic OAuth beta 请求头；此时提供商 `api_key` 仅为占位符。对于其他提供商，您可以配置 `models_api_url` 和自定义 `models_response_format` 来处理不同的 JSON 响应结构。

对于 `codex` 提供商，模型发现会发送当前 Codex CLI `client_version`，因为 ChatGPT 后端可能根据客户端版本对新发布的 Codex 模型 slug 进行门控。CCR 默认使用发布时已知的最新稳定版本；可通过提供商上的 `codex_client_version` 进行覆盖，或在测试较新的 Codex CLI 版本时通过 `CCR_CODEX_CLIENT_VERSION` 环境变量覆盖。运行时的 Codex 请求由核心 Codex 转换器单独处理，该转换器会伪装 Codex CLI 的请求版本和身份请求头，不依赖 CCR 的 CLI 包。

`models_response_format` 对象支持：
- `listPath`：模型数组的 JSON 路径（例如 `"data"`、`"models"`，或根数组使用 `""`）
- `idPath`：每个模型对象中用作 ID 的字段名（例如 `"id"`、`"name"`、`"slug"`）
- `stripPrefix`：可选前缀，从模型 ID 中移除（例如 `"models/"`）

示例：

```json
{
  "name": "together.ai",
  "api_base_url": "https://api.together.ai/v1/chat/completions",
  "models_api_url": "https://api.together.ai/v1/models",
  "api_key": "$TOGETHERAI_API_KEY",
  "models": [],
  "models_response_format": {
    "listPath": "",
    "idPath": "id"
  }
}
```

您也可以通过 CLI 标志覆盖这些设置以便测试：
```shell
ccr model get my-provider --list-path data --id-path id --strip-prefix "v1/"
```

如果提供商返回的模型有变化，`ccr model get <provider>` 可以分别通过独立的确认提示追加缺失条目和移除不再可用的已配置条目。

> **注意**：将模型同步到 `config.json` 后，使用 `ccr restart` 重启服务，使运行中的服务器加载更新后的提供商列表。

#### Antigravity 身份验证

通过 Google 的 Antigravity 网关（`cloudcode-pa`）路由 Claude Code 请求，使用账户 OAuth 而非 API 密钥。

```shell
ccr antigravity-auth
# 或无头/远程模式：
ccr antigravity-auth --manual
ccr antigravity-auth --project <gcp-project-id>
```

该命令：
1. 打印 Google OAuth URL（PKCE；回调 `http://localhost:51121/oauth-callback`）
2. CCR 服务器处理公共 `/oauth-callback` 路由（Docker 映射 `51121:3456`，类似 Codex 的 `1455:3456`）
3. 令牌保存到 `~/.claude-code-router/antigravity_auth.json`（Docker 中为挂载的配置目录）

提供商示例：

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

为何需要这些 Gemini 选项：
- **`cachedContent: false`** — Antigravity 没有 Google `cachedContents` 资源。Gemini 默认值为 `true`；保留会导致 404。
- **`thoughtSignatureFallback: "skip"`** — 默认值的显式形式。当重放的工具调用缺少缓存的 `thoughtSignature` 时，CCR 会在该步骤的**第一个** `functionCall` 上盖印 Google 的 `skip_thought_signature_validator` 哨兵，防止网关返回 400。仅在端点拒绝该哨兵时才改为 `"none"`。

> **注意**：认证过程中请保持 CCR 服务器运行。在非 IDE 客户端中使用 Antigravity IDE OAuth 客户端凭据可能违反 Google 的条款。

#### Codex 提供商身份验证

Codex 提供商支持两种认证模式：

- **OAuth**：通过 `ccr codex-auth`
- **PAT**：通过 `api_key: "at-..."`

##### OAuth 模式

在使用 OAuth 的 Codex 模型之前，请先使用您的 OpenAI 账户进行认证：

```shell
ccr codex-auth
```

该命令：
1. 在浏览器中打开 OpenAI OAuth 授权页面
2. 登录后，OAuth 回调由运行中的 CCR 服务器处理
3. 令牌存储在 `~/.claude-code-router/codex_auth.json`
4. CLI 和服务器会独立地在令牌过期前五分钟刷新

CCR 从 OAuth ID 令牌中推导出所选 ChatGPT 工作空间和 FedRAMP 路由状态。运行时请求和 `ccr model get codex` 都会发送相同的 Codex bearer、账户和路由请求头。刷新操作使用原子凭据文件和跨进程锁，因此单独运行的 CLI 和服务器不会重用相同的轮换刷新令牌。运行时的 OAuth 401 会执行一次受保护的凭据重新加载/刷新重试。

> **注意**：服务器必须运行才能使用 `ccr codex-auth`，因为它托管了 OAuth 回调端点。

**在 Docker 中运行**：

OAuth 回调使用端口 `1455`，该端口在 `docker-compose.yml` 中映射到 CCR 服务器端口（`"1455:3456"`）。在 Docker 中运行时：

```shell
docker exec -it claude-code-router ccr codex-auth
```

CLI 会打印一个 URL，在主机浏览器中打开。登录后，浏览器重定向到 `http://localhost:1455/auth/callback`，Docker 将其转发到容器。令牌通过卷挂载的 `./ccr-config` 目录在容器重启间持久保存。

##### PAT 模式

如果提供商 `api_key` 以 `at-` 开头，CCR 将其视为 Codex 个人访问令牌（PAT）并直接使用。在 PAT 模式下，您**不需要**运行 `ccr codex-auth`。

```json
{
  "name": "codex",
  "api_base_url": "https://chatgpt.com/backend-api/codex",
  "api_key": "at-your-personal-access-token",
  "models": ["gpt-5.4"],
  "transformer": {
    "use": ["codex"]
  }
}
```

CCR 在调用 Codex 后端之前，通过 OpenAI 的 `/whoami` 端点解析 PAT 的账户、用户、计划和 FedRAMP 元数据。运行时请求和 `ccr model get codex` 随后会在需要时发送 `Authorization`、`ChatGPT-Account-ID` 和 `X-OpenAI-Fedramp`。元数据请求由服务器去重并做短暂缓存。

认证模式是明确的：`at-` 值始终视为 PAT。无效或已撤销的 PAT 会以 PAT 认证失败告终，不会静默替换为 OAuth。任何非 PAT 的占位符都会使用 `~/.claude-code-router/codex_auth.json` 中的 OAuth 令牌。

> **另请参阅**：完整的 Codex 设置和故障排除记录在 `docs/docs/server/guides/codex.md` 中。

#### Cursor 提供商身份验证

Cursor 提供商使用官方 `@cursor/sdk`（无需浏览器 OAuth CLI）。认证解析顺序：

1. 提供商 `api_key` 以 `crsr_` 开头（Cursor 仪表盘 API 密钥）
2. 否则使用环境变量 `CURSOR_API_KEY`

提供商示例：

```json
{
  "name": "cursor",
  "api_base_url": "https://cursor.com",
  "api_key": "$CURSOR_API_KEY",
  "models": ["composer-2"],
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
```

- **bridge**（默认）：Claude Code 宿主工具；Cursor 内置工具在 `~/.claude-code-router/cursor-sdk-workspaces/` 下的隔离工作空间中被拒绝
- 使用 `ccr model get cursor` 发现模型（通过 `Cursor.models.list` 列出，而非 REST `/models`）
- Docker Compose 在设置时将 `CURSOR_API_KEY` 传入容器；Docker 中强制关闭本地 Cursor 沙箱
- Cursor 提示缓存是持有中 SDK 代理会话的原生特性。CCR 将每次请求的令牌估算报告给 Claude Code，并将 SDK 缓存读取差值映射为有界的 Anthropic 缓存读取用量。
- Cursor 思考内容通过 SDK 流 `thinking` 消息和令牌级 `Agent.send({ onDelta })` `thinking-delta` 更新转发，然后以合成签名关闭，使 Claude Code 可将其渲染为 Anthropic 扩展思考。Claude Code 2.1.89+ 默认隐藏交互式思考摘要；在传递给 Claude Code 的设置文件中启用 `"showThinkingSummaries": true` 以显示。这是客户端渲染设置：CCR 即使在禁用时仍会传输思考块。
- 停止 Cursor 响应会取消拥有的 SDK 运行（带有限清理），并使不安全的 SDK 会话失效；运行中发送失败时，Cursor 会使用原生 `local.force` 重试，然后 CCR 才回退到全新完整转录会话。

> **另请参阅**：完整的 Cursor 设置记录在 `docs/docs/server/guides/cursor.md` 中。

#### Claude 订阅身份验证

Claude 订阅提供商使用 OAuth，通过您的 Claude Pro 或 Max 订阅访问 Anthropic 的 API。在使用此方式的 Claude 模型之前，您必须先进行认证：

```shell
ccr claude-auth
```

该命令：
1. 在浏览器中打开 Claude OAuth 授权页面
2. 登录后，OAuth 回调由运行中的 CCR 服务器处理（端口 1455）
3. 令牌存储在 `~/.claude-code-router/claude_auth.json`
4. `claude-auth` 转换器会在令牌过期时自动刷新

> **注意**：服务器必须运行才能使用 `ccr claude-auth`，因为它托管了端口 1455 上的 OAuth 回调端点。

**在 Docker 中运行**：

OAuth 回调使用端口 `1455`，该端口在 `docker-compose.yml` 中映射到 CCR 服务器端口（`"1455:3456"`）。在 Docker 中运行时：

```shell
docker exec -it claude-code-router ccr claude-auth
```

CLI 会打印一个 URL，在主机浏览器中打开。登录后，浏览器重定向到 `http://localhost:1455/callback`，Docker 将其转发到容器。令牌通过卷挂载的 `./ccr-config` 目录在容器重启间持久保存。

Claude 订阅提供商需要 `claude-auth` + `Anthropic` 转换器链：

```json
{
  "name": "claude-subscription",
  "api_base_url": "https://api.anthropic.com",
  "api_key": "no-key",
  "models": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
  "transformer": {
    "use": ["claude-auth", "Anthropic"]
  }
}
```

> **另请参阅**：完整的 Claude 订阅设置记录在 `docs/docs/server/guides/claude-auth.md` 中。

**自定义转换器：**

#### Qwen 提供商身份验证

Qwen 提供商使用单个 JWT 向 Qwen Chat 后端进行认证。在使用 Qwen 模型之前，您必须将令牌保存到本地 CCR 配置中：

```shell
ccr qwen-auth
```

该命令：
1. 打印浏览器内认证页面的 URL（`http://127.0.0.1:<port>/qwen/auth`）
2. 该页面提供两个选项：
   - **书签工具（推荐）**：将"Get Qwen Token"链接拖到书签栏，然后在已登录的 Qwen 页面点击。令牌会自动发送回 CCR。
   - **手动粘贴**：在 `chat.qwen.ai` 登录，打开开发者工具（F12）→ 控制台，运行 `copy(localStorage.getItem('token'))`，将 JWT 粘贴到表单中并提交。
3. 令牌通过 `qwen.aikit.club/v1/validate` 验证后保存到 `~/.claude-code-router/qwen_auth.json`（权限 0600）
4. `qwen-auth` 转换器在令牌接近过期时（6 小时内）自动刷新

> **注意**：服务器必须运行才能使用 `ccr qwen-auth`，因为它托管了 `/qwen/auth` 上的认证表单。与 Codex 流程不同，这里不需要 OAuth 回调——令牌直接粘贴到表单中。

**在 Docker 中运行**：

Qwen 认证页面在常规 CCR 端口上提供服务（无单独的回调端口）。在 Docker 中运行时：

```shell
docker exec -it claude-code-router ccr qwen-auth
```

CLI 会打印一个 URL，在主机浏览器中打开（`http://localhost:3456/qwen/auth`，Docker 将其转发到容器）。令牌通过卷挂载的 `./ccr-config` 目录在容器重启间持久保存。

**书签工具的自定义主机/端口**：书签工具的重定向目标硬编码在 JS 中，因为它在 Qwen 页面的上下文中运行（无法获知 CCR 的地址）。默认指向 `http://127.0.0.1:3456`。如果您的 CCR 服务器在不同主机或端口上，请在启动服务器前设置 `QWEN_AUTH_REDIRECT` 环境变量，例如 `QWEN_AUTH_REDIRECT=http://192.168.1.10:8080`——书签工具将重定向到该地址。

#### Chrome 端侧桥接

`chrome-on-device` 转换器需要一个在主机上运行的桥接进程，用于与 Chrome 的 Gemini Nano 模型通信：

```bash
# 启动桥接器（默认：端口 3457，CDP 端口 9222）
ccr chrome-bridge

# 自定义端口
ccr chrome-bridge --port 3457 --cdp 9222
```

桥接器：
1. 检查 Chrome 是否已启用远程调试运行（端口 9222）
2. 如果未运行，则使用必需的标志启动 Chrome（`--remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug-profile`）
3. 通过 Puppeteer/CDP 连接到 Chrome，协议超时 5 分钟以处理慢速模型推理
4. 加载访问 Prompt API（`window.LanguageModel`）的页面，并在所有请求间保持持久的 `LanguageModel` 会话——对话历史在会话内自然延续，而非每次请求重建
5. 将 Claude Code 的系统提示替换为最小化工具提示（5 个核心工具），使用 `responseConstraint`（JSON Schema）强制模型发出结构化 JSON，包含 `{text, tool_calls[]}` 字段
6. 在 `0.0.0.0:3457` 上暴露与 OpenAI 兼容的 HTTP API：
   - `GET /v1/models` — 返回可用模型及实时上下文使用量
   - `GET /v1/models/{model_name}` — 返回单个模型信息（display_name、max_input_tokens、capabilities）
   - `POST /v1/chat/completions` — 支持流式和非流式的聊天补全
   - `GET /health` — 健康检查

**前置要求**：需启用 Chrome 标志（参见 Chrome 端侧提供商配置部分）。模型（约 4GB）必须已下载。

> **Docker 用户注意**：桥接器运行在 Docker **主机**上，而非容器内。请在 `config.json` 中设置提供商主机为 `http://host.docker.internal:3457`。

### 6. 预设管理

预设允许您轻松保存、共享和重用配置。您可以将当前配置导出为预设，并从文件或 URL 安装预设。

```shell
# 将当前配置导出为预设
ccr preset export my-preset

# 使用元数据导出
ccr preset export my-preset --description "我的 OpenAI 配置" --author "您的名字" --tags "openai,production"

# 从本地目录安装预设
ccr preset install /path/to/preset

# 列出所有已安装的预设
ccr preset list

# 显示预设信息
ccr preset info my-preset

# 删除预设
ccr preset delete my-preset
```

**预设功能：**
- **导出**：将当前配置保存为预设目录（包含 manifest.json）
- **安装**：从本地目录安装预设
- **敏感数据处理**：导出期间自动清理 API 密钥和其他敏感数据（标记为 `{{field}}` 占位符）
- **动态配置**：预设可包含输入模式，用于在安装期间收集所需信息
- **版本控制**：每个预设包含版本元数据，用于跟踪更新

**预设文件结构：**
```
~/.claude-code-router/presets/
├── my-preset/
│   └── manifest.json    # 包含配置和元数据
```

### 7. Activate 命令（环境变量设置）

`activate` 命令允许您在 shell 中全局设置环境变量，使您能直接使用 `claude` 命令或将 Claude Code Router 与使用 Agent SDK 构建的应用程序集成。

要激活环境变量，请运行：

```shell
eval "$(ccr activate)"
```

此命令以 shell 友好的格式输出必要的环境变量，这些变量将在当前 shell 会话中设置。激活后，您可以：

- **直接使用 `claude` 命令**：无需使用 `ccr code` 即可运行 `claude` 命令。`claude` 命令将自动通过 Claude Code Router 路由请求。
- **与 Agent SDK 应用程序集成**：使用 Anthropic Agent SDK 构建的应用程序将自动使用配置的路由器和模型。

`activate` 命令设置以下环境变量：

- `ANTHROPIC_AUTH_TOKEN`：来自配置的 API 密钥
- `ANTHROPIC_BASE_URL`：本地路由器端点（默认：`http://127.0.0.1:3456`）
- `NO_PROXY`：设置为 `127.0.0.1` 以防止代理干扰
- `DISABLE_TELEMETRY`：禁用遥测
- `DISABLE_COST_WARNINGS`：禁用成本警告
- `API_TIMEOUT_MS`：来自配置的 API 超时时间

> **注意**：使用激活的环境变量前，请确保 Claude Code Router 服务正在运行（`ccr start`）。环境变量仅在当前 shell 会话中有效。要使其持久化，可将 `eval "$(ccr activate)"` 添加到您的 shell 配置文件（例如 `~/.zshrc` 或 `~/.bashrc`）中。

#### Providers

`Providers` 数组是定义要使用的不同模型提供商的地方。每个提供商对象需要：

- `name`：提供商的唯一名称。
- `api_base_url`：聊天补全的完整 API 端点。
- `api_key`：您的提供商 API 密钥。
- `models`：此提供商可用的模型名称列表。
- `transformer`（可选）：指定用于处理请求和响应的转换器。

#### Transformers

转换器允许您修改请求和响应负载，以确保与不同提供商 API 的兼容性。

- **全局转换器**：将转换器应用于提供商的所有模型。在此示例中，`openrouter` 转换器应用于 `openrouter` 提供商下的所有模型。
  ```json
  {
    "name": "openrouter",
    "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
    "api_key": "sk-xxx",
    "models": [
      "google/gemini-2.5-pro-preview",
      "anthropic/claude-sonnet-4",
      "anthropic/claude-3.5-sonnet"
    ],
    "transformer": { "use": ["openrouter"] }
  }
  ```
- **特定模型转换器**：将转换器应用于特定模型。在此示例中，`deepseek` 转换器应用于所有模型，额外的 `tooluse` 转换器仅应用于 `deepseek-chat` 模型。

  ```json
  {
    "name": "deepseek",
    "api_base_url": "https://api.deepseek.com/chat/completions",
    "api_key": "sk-xxx",
    "models": ["deepseek-chat", "deepseek-reasoner"],
    "transformer": {
      "use": ["deepseek"],
      "deepseek-chat": { "use": ["tooluse"] }
    }
  }
  ```

- **向转换器传递选项**：某些转换器（如 `maxtoken`）接受选项。要传递选项，使用嵌套数组，其中第一个元素是转换器名称，第二个元素是选项对象。
  ```json
  {
    "name": "siliconflow",
    "api_base_url": "https://api.siliconflow.cn/v1/chat/completions",
    "api_key": "sk-xxx",
    "models": ["moonshotai/Kimi-K2-Instruct"],
    "transformer": {
      "use": [
        [
          "maxtoken",
          {
            "max_tokens": 16384
          }
        ]
      ]
    }
  }
  ```

**可用的内置转换器：**

- `Anthropic`：如果仅使用 `Anthropic` 转换器，它将透传原始请求和响应参数（可用来直接连接 Anthropic 端点）。
- `deepseek`：适配 DeepSeek API 的请求/响应。
- `gemini`：适配 Gemini API 的请求/响应（也是与 Antigravity 联用的方言阶段；相同选项适用于 `vertex-gemini`）。Claude Code 的思考深度设置（作为 `output_config.effort` 发送）驱动思考深度：Gemini 3+ 使用 `thinkingLevel`（Gemini 3 Pro 为 `low`/`high`，较新 Pro 增加 `medium`，Flash/Lite 增加 `minimal`），或 Gemini 2.5 及经 Antigravity 服务的 Claude 使用 `thinkingBudget`——切勿同时设置两者（API 会拒绝）。超出系列范围时向上取整（Gemini 3 Pro 的 `medium`，任何模型上的 `xhigh`/`max` → `high`），且不会重写配置的模型 ID（例如 `gemini-3-pro-low` 仍访问 `gemini-3-pro-low`）。通过 `["gemini", { ... }]` 传递选项：
  - **`cachedContent`**（布尔值，默认 `true`）：CCR 是否可使用 Google 独立的 **`cachedContents` HTTP 资源**来存储/复用公共 Gemini API 上的提示前缀。这是 Gemini 服务端上下文缓存——**不是** Anthropic `cache_control`，也**不是** Claude Code 的本地提示缓存。普通 Gemini 保持 `true`；**Antigravity 必须设为 `false`**（以及任何没有 `cachedContents` 的网关），否则会 404。
  - **`thoughtSignatureFallback`**（`"skip"` | `"none"`，默认 `"skip"`）：当重放的工具调用缺少缓存的 Gemini `thoughtSignature` 时如何处理（Claude Code 的 Anthropic `tool_use` 无法携带该字段，因此 CCR 按 tool-call id 缓存并还原签名；未命中否则会 400）。`"skip"` 表示在该步骤的**第一个** `functionCall` 上盖印 Google 文档中的哨兵 `skip_thought_signature_validator`——选项名称指该哨兵，**不是**"禁用回退"。Gemini/Antigravity 保持 `"skip"`；仅在端点拒绝该哨兵时（部分 Vertex）设为 `"none"`。真实缓存签名始终优先；哨兵是最后手段。
- `mistral`：适配 Mistral API 的请求/响应。
- `openrouter`：适配 OpenRouter API 的请求/响应。它还可以接受 `provider` 路由参数，指定 OpenRouter 应使用哪些底层提供商。更多详情请参阅 [OpenRouter 文档](https://openrouter.ai/docs/features/provider-routing)。示例如下：
  ```json
    "transformer": {
      "use": ["openrouter"],
      "moonshotai/kimi-k2": {
        "use": [
          [
            "openrouter",
            {
              "provider": {
                "only": ["moonshotai/fp8"]
              }
            }
          ]
        ]
      }
    }
  ```
- `groq`：适配 groq API 的请求/响应。
- `maxtoken`：设置特定的 `max_tokens` 值。
- `tooluse`：通过 `tool_choice` 优化某些模型的工具使用。
- `gemini-cli`（实验性）：通过 Gemini CLI [gemini-cli.js](https://gist.github.com/musistudio/1c13a65f35916a7ab690649d3df8d1cd) 对 Gemini 的非官方支持。
- `reasoning`：用于处理 `reasoning_content` 字段。
- `sampling`：用于处理采样信息字段，如 `temperature`、`top_p`、`top_k` 和 `repetition_penalty`。
- `enhancetool`：对 LLM 返回的工具调用参数增加一层容错处理（这会导致工具调用信息不再流式返回）。
- `cleancache`：清除请求中的 `cache_control` 字段。
- `vertex-gemini`：处理使用 Vertex 认证的 Gemini API。
- `chutes-glm`：通过 Chutes [chutes-glm-transformer.js](https://gist.github.com/vitobotta/2be3f33722e05e8d4f9d2b0138b8c863) 对 GLM 4.5 模型的非官方支持。
- `qwen-cli`（实验性）：通过 Qwen CLI [qwen-cli.js](https://gist.github.com/musistudio/f5a67841ced39912fd99e42200d5ca8b) 对 qwen3-coder-plus 模型的非官方支持。
- `rovo-cli`（实验性）：通过 Atlassian Rovo Dev CLI [rovo-cli.js](https://gist.github.com/SaseQ/c2a20a38b11276537ec5332d1f7a5e53) 对 GPT-5 的非官方支持。
- `codex`：适配 Codex (ChatGPT) 后端 API 的请求/响应。支持通过 `ccr codex-auth` 的 OAuth 或 `api_key` 以 `at-` 开头的 PAT 认证。
- `claude-auth`：使用您的 Claude Pro 或 Max 订阅 OAuth 令牌向 Anthropic API 认证请求。将 Unified 格式转换为 Anthropic 格式并处理 SSE 响应转换。需与 `Anthropic` 组成提供商链，并通过 `ccr claude-auth` 认证。
- `antigravity-auth`：用于 Google Antigravity 网关（`cloudcode-pa`）的 OAuth + 封包中间件。链接在 `gemini` **之后**。Antigravity 上 Gemini 阶段必须设置 `cachedContent: false`（没有 `cachedContents` 资源）；除非端点拒绝 Google 的 thought-signature 哨兵，否则保持 `thoughtSignatureFallback: "skip"`。使用 `ccr antigravity-auth` 认证。
- `chrome-on-device`：将请求路由到 Chrome 的端侧 Gemini Nano 模型（通过 Prompt API）。使用 `responseConstraint` 实现结构化 JSON 输出。需要在主机上运行桥接进程（`ccr chrome-bridge`）。

**Chrome 端侧提供商配置：**

`chrome-on-device` 转换器将请求路由到 Chrome 内置的 Gemini Nano 模型。这是一个约 4GB 的端侧模型，本地运行，无 API 成本。通过桥接进程访问 Chrome 的 Prompt API（`window.LanguageModel`）。

**前置要求：**

1. 系统已安装 Google Chrome（macOS、Windows 或 Linux）
2. 启用 Chrome 标志（一次性）：
   - `chrome://flags/#optimization-guide-on-device-model` → **Enabled**
   - `chrome://flags/#prompt-api-for-gemini-nano-multimodal-input` → **Enabled**
3. 启用标志后重启 Chrome，等待模型下载（约 4GB）
4. 在主机上启动桥接进程：`ccr chrome-bridge`

**提供商配置：**

```json
{
  "name": "chrome-nano",
  "api_base_url": "http://127.0.0.1:3457",
  "api_key": "placeholder",
  "models": ["gemini-nano"],
  "transformer": {
    "use": ["chrome-on-device", "tooluse"]
  }
}
```

> **注意**：`tooluse` 转换器必须与 `chrome-on-device` 一起使用，以启用按需工具调用系统（包括用于纯文本响应的 `ExitTool`）并注入必要的系统提示，帮助模型在思考和行动之间转换。

**启动桥接器：**

桥接器是一个独立的 HTTP 服务器，运行在主机上，通过 CDP（Chrome DevTools Protocol）桥接 HTTP 请求到 Chrome 的 Prompt API：

```bash
# 启动桥接器（默认：端口 3457，CDP 端口 9222）
ccr chrome-bridge

# 自定义端口
ccr chrome-bridge --port 3457 --cdp 9222
```

桥接器会使用必需的标志自动启动 Chrome（如果尚未运行）：`--remote-debugging-port=9222 --user-data-dir=<temp_dir>`。

> **Docker 用户注意**：桥接器必须在 Docker **主机**上运行（不在容器内），因为它需要通过 CDP 直接访问 Chrome。当 CCR 在 Docker 中运行时，请在 `config.json` 中设置提供商主机为 `http://host.docker.internal:3457`。

**工作原理：**

1. 转换器将 Claude Code 的系统提示替换为最小化工具提示，列出 5 个核心工具（Bash、Read、Write、Edit、ExitTool）
2. 桥接器维护持久的 `LanguageModel` 会话——每个客户端指纹（`User-Agent + IP` 哈希）一个。对话历史在会话内自然延续，而非每轮重建。它调用 `session.promptStreaming()` 并附带 `responseConstraint`（JSON Schema），强制结构化输出：`{"tool_calls": [{"name": "...", "arguments": {...}}]}`。文本响应由模型调用 `ExitTool` 处理。
3. 桥接器转换用户消息中的 Claude Code 内部上下文块以节省有限的上下文预算：包含工具调用或结果的 `<system-reminder>` 块转换为结构化 `<tool_result>` 标签，而其他 `<system-reminder>` 块和不受支持工具的 `<command-*>` / `<local-command-*>` 块被剥离
4. 桥接器将结构化 JSON 响应解析为 OpenAI 格式的 SSE 块（`chat.completion.chunk`）或单个非流式响应（`chat.completion`）
5. 从解析的 JSON 中检测工具调用，转换为响应中的 `tool_calls`；`finish_reason` 相应设置为 `"tool_calls"` 或 `"stop"`
6. 支持多轮工具使用——连续请求在同一个持久会话中处理
7. **多会话支持**：请求按 `User-Agent + IP` 哈希指纹分隔为不同会话，允许多个并发的 Claude Code 实例而不会上下文污染。内置 Web 仪表盘（在桥接器端口提供服务）显示所有会话的实时统计信息，包括轮次计数、空闲时间和上下文使用率
8. **空闲会话逐出**：空闲超过 5 分钟的会话自动销毁以释放资源。`cli` 会话（仪表盘默认）永不逐出。也可通过仪表盘的 Evict 按钮手动逐出会话
9. 上下文使用率达到 85% 时自动触发压缩，重置会话同时保留系统提示

**限制：**

- **工具调用**：使用 `responseConstraint`（JSON Schema）实现结构化输出，而非原生函数调用——这种方式可靠，但依赖模型遵循模式
- **多轮一致性**：小型端侧模型可能会偶尔在同一个工具调用上循环，或用文本响应替代调用所需工具。带修正提示的重试机制可缓解此问题
- **无思考/推理块**：Prompt API 不区分思考与可见输出
- **上下文窗口**：限制为 9216 个令牌；上下文使用率达 85% 时自动触发压缩。上下文溢出时会逐出旧交互
- **输出限制**：模型可能在大量空白内容上停滞（例如 Python 缩进）。桥接器使用"先写后改"的增量文件创建（每个 Write 调用 3 行）和空白停滞检测与中止
- **跨平台支持**：兼容 macOS、Windows 和 Linux（需安装 Chrome 并手动启用标志）

**Codex 提供商配置：**

Codex 转换器连接到 ChatGPT 后端 API，提供对 GPT-5.x 模型的访问。支持 OAuth 认证或 `api_key` 中的 PAT。

```json
{
  "name": "codex",
  "api_base_url": "https://chatgpt.com/backend-api/codex",
  "api_key": "oauth_dummy_key",
  "models": ["gpt-5.4"],
  "transformer": {
    "use": ["codex"]
  }
}
```

> **OAuth 模式**：将 `api_key` 保留为占位符，运行 `ccr codex-auth`。OAuth 令牌存储在 `~/.claude-code-router/codex_auth.json`。

```json
{
  "name": "codex",
  "api_base_url": "https://chatgpt.com/backend-api/codex",
  "api_key": "at-your-personal-access-token",
  "models": ["gpt-5.4"],
  "transformer": {
    "use": ["codex"]
  }
}
```

> **PAT 模式**：如果 `api_key` 以 `at-` 开头，CCR 直接使用并跳过 `ccr codex-auth`。

> **注意**：如果 `api_key` 不是 PAT，CCR 回退到使用 `~/.claude-code-router/codex_auth.json` 中的 OAuth 令牌。

**Qwen 提供商配置：**

Qwen 提供商使用 `qwen-auth` 转换器（用于 `Authorization: Bearer <jwt>` 请求头和尾部 `<details>` 去除）与现有的 `OpenAI` 转换器配对（后者注册 `POST /v1/chat/completions` 端点）。

```json
{
  "name": "qwen",
  "api_base_url": "https://qwen.aikit.club/v1/chat/completions",
  "api_key": "qwen-placeholder",
  "models": ["qwen3-max", "qwen3-coder-plus"],
  "transformer": {
    "use": ["qwen-auth", "reasoning", "OpenAI"]
  }
}
```

链中需要三个转换器：

- `qwen-auth` — 在每个出站请求上设置 `Authorization: Bearer <jwt>` 请求头（从 `~/.claude-code-router/qwen_auth.json` 加载/刷新 JWT），并去除 Qwen 注入到响应中的尾部 `<details>...</details>` 块。
- `reasoning` — 将 Claude Code 统一的 `reasoning` 字段映射到请求，使 Qwen 端点的 `enable_thinking` 和 `thinking_budget` 参数被填充。
- `OpenAI` — 注册 `POST /v1/chat/completions` 路由。它是仅注册端点的薄桩，不进行 body 转换，因此必须保持在链的最后。

> **注意**：`api_key` 字段是占位符——实际认证通过存储在 `~/.claude-code-router/qwen_auth.json` 中的 JWT 处理。使用 Qwen 提供商前请运行 `ccr qwen-auth` 进行认证。

**Claude 订阅提供商配置：**

`claude-auth` 转换器使用您的 Claude Pro 或 Max 订阅 OAuth 令牌（而非静态 API 密钥）将请求路由到 Anthropic 的 API。

```json
{
  "name": "claude-subscription",
  "api_base_url": "https://api.anthropic.com",
  "api_key": "no-key",
  "models": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
  "transformer": {
    "use": ["claude-auth", "Anthropic"]
  }
}
```

链中需要两个转换器：

- `claude-auth` — 将请求从 Unified（OpenAI）格式转换为 Anthropic 格式，注入 `Authorization: Bearer <token>`（从 `~/.claude-code-router/claude_auth.json` 加载/刷新令牌），并将 Anthropic SSE 响应转换回 Unified 格式。
- `Anthropic` — 注册 `POST /v1/messages` 路由。在提供商链中不做 body 转换，作为无操作端点桩。

> **注意**：`api_key` 字段是占位符——实际认证通过存储在 `~/.claude-code-router/claude_auth.json` 中的 OAuth 令牌处理。使用此提供商前请运行 `ccr claude-auth`。该转换器会自动发送 Anthropic 的 `oauth-2025-04-20` beta，使订阅 OAuth Bearer 令牌被接受。

**通过 OpenCode 使用 DeepSeek（强制推理重放）：**

DeepSeek 模型需要在后续请求中重放前次助手的推理内容。当通过 OpenCode 等提供商使用 DeepSeek 模型时，在模型级别应用 `reasoning` 转换器以自动处理此需求：

```json
{
  "name": "opencode",
  "api_base_url": "https://opencode.ai/zen/go/v1/chat/completions",
  "api_key": "$OPENCODE_API_KEY",
  "models": ["deepseek-v4-pro", "deepseek-v4-flash"],
  "transformer": {
    "use": ["OpenAI"],
    "deepseek-v4-pro": {
      "use": ["reasoning"]
    },
    "deepseek-v4-flash": {
      "use": ["reasoning"]
    }
  }
}
```

> **注意**：`reasoning` 转换器必须特定应用于 DeepSeek 模型（而非提供商级别）。它会按 DeepSeek API 的要求，重放先前轮次的助手推理输出。

**自定义转换器：**

您还可以创建自己的转换器，并通过 `config.json` 中的 `transformers` 字段加载它们。

```json
{
  "transformers": [
    {
      "path": "/User/xxx/.claude-code-router/plugins/gemini-cli.js",
      "options": {
        "project": "xxx"
      }
    }
  ]
}
```

#### Router

`Router` 对象定义了在不同场景下使用哪个模型：

- `default`：用于常规任务的默认模型。
- `background`：用于后台任务的模型。这可以是一个较小的本地模型以节省成本。
- `think`：用于推理密集型任务（如计划模式）的模型。
- `longContext`：用于处理长上下文（例如 > 60K 令牌）的模型。
- `longContextThreshold`（可选）：触发长上下文模型的令牌数阈值。如未指定，默认值为 60000。
- `webSearch`：用于处理网络搜索任务，需要模型本身支持该功能。如果使用 openrouter，需要在模型名称后添加 `:online` 后缀。
- `image`（测试版）：用于处理图片相关任务（由 CCR 内置代理支持）。如果模型不支持工具调用，需将 `config.forceUseImageAgent` 属性设为 `true`。

- 您也可以在 Claude Code 中使用 `/model` 命令动态切换模型：
`/model provider_name,model_name`
示例：`/model openrouter,anthropic/claude-3.5-sonnet`

#### 自定义 Router

对于更高级的路由逻辑，您可以在 `config.json` 中通过 `CUSTOM_ROUTER_PATH` 指定自定义路由脚本。这允许您实现超出默认场景的复杂路由规则。

在 `config.json` 中：

```json
{
  "CUSTOM_ROUTER_PATH": "/User/xxx/.claude-code-router/custom-router.js"
}
```

自定义路由文件必须是一个导出 `async` 函数的 JavaScript 模块。该函数接收请求对象和配置对象作为参数，应返回提供商和模型名称的字符串（例如 `"provider_name,model_name"`），或返回 `null` 以回退到默认路由。

以下是基于 `custom-router.example.js` 的 `custom-router.js` 示例：

```javascript
// /User/xxx/.claude-code-router/custom-router.js

/**
 * 自定义路由函数，根据请求确定使用哪个模型。
 *
 * @param {object} req - 来自 Claude Code 的请求对象，包含请求体。
 * @param {object} config - 应用程序的配置对象。
 * @returns {Promise<string|null>} - 解析为 "provider,model_name" 字符串的 Promise，或返回 null 以使用默认路由。
 */
module.exports = async function router(req, config) {
  const userMessage = req.body.messages.find((m) => m.role === "user")?.content;

  if (userMessage && userMessage.includes("explain this code")) {
    // 为代码解释使用更强大的模型
    return "openrouter,anthropic/claude-3.5-sonnet";
  }

  // 回退到默认路由配置
  return null;
};
```

##### 子代理路由

Claude Code 子代理请求可通过以下方式路由：

1. 在系统或消息文本中使用显式 `<CCR-SUBAGENT-MODEL>provider,model</CCR-SUBAGENT-MODEL>` 标签（该标签在发送上游前会被剥离）。也支持 `provider/model`。
2. 或在 Claude Code 将请求标记为子代理时，使用环境变量 `CLAUDE_CODE_SUBAGENT_MODEL=provider,model`。

标签优先于环境变量。若两者都未设置，则使用常规 Router 规则。

**示例：**

```
<CCR-SUBAGENT-MODEL>openrouter,anthropic/claude-3.5-sonnet</CCR-SUBAGENT-MODEL>
请帮我分析这段代码片段，寻找潜在的优化点...
```

```bash
export CLAUDE_CODE_SUBAGENT_MODEL="openrouter,anthropic/claude-3.5-sonnet"
```

## 提示缓存

CCR 自动将 Claude Code 的 Anthropic 缓存意图转换为各上游的原生缓存机制：

- Anthropic、Claude Auth 和 Vertex Claude 使用 Anthropic 自动提示缓存，同时保留有界的显式块标记。
- OpenAI Chat 和 Responses 使用稳定的 `prompt_cache_key` 值；GPT-5.6+ 模型还接收显式内容断点。Codex 对每个模型使用其独立的原生契约：稳定的提示键与会话路由请求头，无显式内容断点。
- OpenRouter 使用粘性 `session_id` 路由加模型原生缓存。Vercel AI Gateway 接收 `providerOptions.gateway.caching: "auto"`。
- Mistral 和 Cerebras 接收原生 `prompt_cache_key` 值。Qwen/DashScope 接收最终内容缓存标记。DeepSeek、Groq 和 Vertex OpenAI 保留其原生隐式缓存。
- Gemini 和 Vertex Gemini 使用隐式缓存，并为足够大的稳定系统/工具前缀创建/复用原生 CachedContent 资源。
- Cursor SDK 和 Chrome On-Device 复用稳定的原生会话。

仅在与转换完成后移除提供商不兼容的缓存字段。上游报告的缓存读/写用量会转换回 Anthropic 的 `cache_read_input_tokens` 和 `cache_creation_input_tokens`，以供 Claude Code 使用。

参见[实现计划和审查](tasks/caching-plan.md)了解提供商矩阵和验证范围。

## Status Line（Beta）
为了更好地监控 claude-code-router 运行时的状态，v1.0.40 版本内置了 statusline 工具，您可以在 UI 中启用它。
![statusline-config.png](/blog/images/statusline-config.png)

效果如下：
![statusline](/blog/images/statusline.png)

## 🤖 GitHub Actions

将 Claude Code Router 集成到您的 CI/CD 管道中。设置 [Claude Code Actions](https://docs.anthropic.com/en/docs/claude-code/github-actions) 后，修改您的 `.github/workflows/claude.yaml` 以使用路由器：

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

这种设置可实现有趣的自动化，例如在非高峰时段运行任务以降低 API 成本。

## 📝 延伸阅读

- [Codex API](https://developers.openai.com/codex/sdk) — 用于 `codex` 转换器的 ChatGPT 后端 API 开发者文档（OAuth PKCE、Responses API、流式传输、工具调用）
- [Chrome Prompt API](https://developer.chrome.com/docs/ai/prompt-api) — 用于 `chrome-on-device` 转换器和桥接器的端侧 Gemini Nano API
- [提供商集成经验教训](tasks/lessons.md) — LLM 提供商集成的硬核知识（DeepSeek、Mistral、Gemini、Codex、Gemini Nano）
