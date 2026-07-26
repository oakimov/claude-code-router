---
title: 转换器
sidebar_position: 4
---

# 转换器

转换器是适配不同 LLM 提供商 API 差异的核心机制。它们在不同格式之间转换请求和响应，处理认证，并管理提供商特定的功能。

## 理解转换器

### 什么是转换器？

转换器是一个插件，它可以：
- **转换请求**：从统一格式转换为提供商特定格式
- **转换响应**：从提供商格式转换回统一格式
- **处理认证**：为提供商 API 处理认证
- **修改请求**：添加或调整参数

### 数据流

```
┌─────────────────┐
│ 传入请求        │ (来自 Claude Code 的 Anthropic 格式)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  transformRequestOut            │ ← 将传入请求解析为统一格式
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  UnifiedChatRequest             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  transformRequestIn (可选)      │ ← 在发送前修改统一请求
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  提供商 API 调用                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  transformResponseIn (可选)     │ ← 将提供商响应转换为统一格式
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  transformResponseOut (可选)    │ ← 将统一响应转换为 Anthropic 格式
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ 传出响应        │ (返回给 Claude Code 的 Anthropic 格式)
└─────────────────┘
```

### 转换器接口

所有转换器都实现以下接口：

```typescript
interface Transformer {
  // 将统一请求转换为提供商特定格式
  transformRequestIn?: (
    request: UnifiedChatRequest,
    provider: LLMProvider,
    context: TransformerContext
  ) => Promise<Record<string, any>>;

  // 将提供商请求转换为统一格式
  transformRequestOut?: (
    request: any,
    context: TransformerContext
  ) => Promise<UnifiedChatRequest>;

  // 将提供商响应转换为统一格式
  transformResponseIn?: (
    response: Response,
    context?: TransformerContext
  ) => Promise<Response>;

  // 将统一响应转换为提供商格式
  transformResponseOut?: (
    response: Response,
    context: TransformerContext
  ) => Promise<Response>;

  // 自定义端点路径（可选）
  endPoint?: string;

  // 转换器名称（用于自定义转换器）
  name?: string;

  // 自定义认证处理器（可选）
  auth?: (
    request: any,
    provider: LLMProvider,
    context: TransformerContext
  ) => Promise<any>;

  // Logger 实例（自动注入）
  logger?: any;
}
```

### 关键类型

#### UnifiedChatRequest

```typescript
interface UnifiedChatRequest {
  messages: UnifiedMessage[];
  model: string;
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  tools?: UnifiedTool[];
  tool_choice?: any;
  reasoning?: {
    effort?: ThinkLevel;  // "none" | "low" | "medium" | "high" | "xhigh" | "max"
    max_tokens?: number;
    enabled?: boolean;
  };
}
```

#### UnifiedMessage

```typescript
interface UnifiedMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string | null | MessageContent[];
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: {
      name: string;
      arguments: string;
    };
  }>;
  tool_call_id?: string;
  thinking?: {
    content: string;
    signature?: string;
  };
}
```

## 内置转换器

### anthropic

将请求转换为兼容 Anthropic 风格的 API：

```json
{
  "transformers": [
    {
      "name": "anthropic",
      "providers": ["deepseek", "groq"]
    }
  ]
}
```

**功能：**
- 在 Anthropic 消息格式和 OpenAI 格式之间转换
- 处理工具调用和工具结果
- 支持思考/推理内容块
- 管理流式响应

### cursor-sdk

通过 `@cursor/sdk` 在进程内运行 Cursor 模型（不会对 `api_base_url` 发起 HTTP fetch）。

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

**功能：**
- 拥有上游 Agent create/send/stream 调用，并通过 `__providerResponse` 返回 OpenAI 兼容 SSE/JSON
- 默认 `cursorMode: "bridge"` — Claude Code 托管工具，拒绝 Cursor 内置工具
- 可选模式：`plan`、`agent`（可配 `cursorCwd`）
- 认证：以 `crsr_` 开头的 `api_key`，或 `CURSOR_API_KEY`
- 通过 `ccr model get` 使用 `Cursor.models.list` 发现模型

另见：
- [Codex 集成指南](/zh/docs/server/guides/codex)
- [Claude 订阅集成指南](/zh/docs/server/guides/claude-auth)
- [Cursor SDK 集成指南](/zh/docs/server/guides/cursor)
- [CLI 认证命令](/zh/docs/cli/commands/auth)

### antigravity-auth

用于 Google Antigravity 网关（`cloudcode-pa`）的 OAuth + 请求封包中间件。

使用 `ccr antigravity-auth` 认证（可选 `--manual`、`--project <id>`）。令牌保存在 `~/.claude-code-router/antigravity_auth.json`。Docker Compose 发布 `51121:3456`，以便 Google 重定向到 `http://localhost:51121/oauth-callback` 时到达 CCR 服务器。

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

**功能：**
- 注入 Antigravity OAuth bearer，并将 Gemini `generateContent` 请求体包装为 Antigravity 请求封包
- 必须链接在 `gemini`（或兼容的 Gemini 方言转换器）**之后**
- 在 daily → autopush → prod 主机间对传输 / 权限失败做端点回退

**示例中必需的 Gemini 选项：**
- `cachedContent: false` — Antigravity 没有 Google `cachedContents` 资源；保留 Gemini 默认值（`true`）会导致 404。详见 [gemini 选项](#选项-cachedcontent-与-thoughtsignaturefallback)。
- `thoughtSignatureFallback: "skip"` — 默认值的显式写法；当缺少工具调用的 thought signature 时，盖印 Google 的 `skip_thought_signature_validator` 哨兵，避免 Gemini/Antigravity 返回 400。仅在端点拒绝该哨兵时改为 `"none"`。

另见：[CLI 认证命令](/zh/docs/cli/commands/auth)

### deepseek

专门用于 DeepSeek API 的转换器：

```json
{
  "transformers": [
    {
      "name": "deepseek",
      "providers": ["deepseek"]
    }
  ]
}
```

**功能：**
- DeepSeek 特定的推理格式
- 处理响应中的 `reasoning_content`
- 支持思考预算令牌

### gemini

用于 Google Gemini API 的转换器（也是与 Antigravity 联用的方言阶段）。

```json
{
  "name": "gemini",
  "api_base_url": "https://generativelanguage.googleapis.com/v1beta/models/",
  "api_key": "$GEMINI_API_KEY",
  "models": ["gemini-3-flash"],
  "transformer": {
    "use": [
      ["gemini", { "cachedContent": true, "thoughtSignatureFallback": "skip" }]
    ]
  }
}
```

**功能：**
- 将 Claude Code 的 `output_config.effort` 翻译为 Gemini 思考深度：Gemini 3+ 使用 `thinkingLevel`（Gemini 3 Pro 为 `low`/`high`，较新 Pro 增加 `medium`，Flash/Lite 增加 `minimal`），Gemini 2.5 使用 `thinkingBudget` — 切勿同时设置两者
- 从不重写已配置的模型 id（带 tier 后缀的 id 继续访问同一上游 id）
- 以下选项同样适用于 `vertex-gemini`

#### 选项：`cachedContent` 与 `thoughtSignatureFallback`

作为 `gemini` / `vertex-gemini` 在 `transformer.use` 中的第二项传入：

```json
["gemini", { "cachedContent": false, "thoughtSignatureFallback": "skip" }]
```

**`cachedContent`**（布尔值，**默认 `true`**）

控制 CCR 是否使用 Google 独立的 **`cachedContents` HTTP 资源**，在公共 Gemini API 上存储并复用提示词前缀（system / tools / history）。这是 Gemini 的*服务端*上下文缓存 — **不是** Anthropic 的 `cache_control` 标记，也**不是** Claude Code 本地提示缓存。

| 取值 | 行为 | 何时使用 |
| --- | --- | --- |
| `true`（默认） | CCR 可能创建/更新 `cachedContents` 对象，并在后续轮次发送 `cachedContent` 引用 | 公共 Gemini（`generativelanguage.googleapis.com`），需要 Google 侧前缀缓存时 |
| `false` | 从不调用 `cachedContents` | **Antigravity 必需**（该网关没有 `cachedContents` 端点 — 保留 `true` 会产生 404 与无用重试）。对任何未实现该资源的 Gemini 兼容代理也应设为 `false` |

**`thoughtSignatureFallback`**（`"skip"` \| `"none"`，**默认 `"skip"`**）

Gemini 3（以及 Antigravity）会在每个 `functionCall` part 上附带不透明的 **`thoughtSignature`**。Claude Code 的 Anthropic 协议无法在 `tool_use` 上携带该字段，因此 CCR 按 tool-call id 缓存签名，并在同一工具被重放时还原。当签名确实缺失（缓存未命中，或会话被重路由到另一上游）时，除非请求在该步骤的**第一个** `functionCall` 上包含 Google 文档中的哨兵字符串 `skip_thought_signature_validator`，否则 Gemini 会返回 **400**。

| 取值 | 行为 | 何时使用 |
| --- | --- | --- |
| `"skip"`（默认） | 未命中时在首个 function call 上盖印该哨兵，使本轮得以继续。该名称表示“使用 Google 的 *skip_thought_signature_validator* 哨兵”，**不是**“跳过 / 关闭回退”。 | 公共 Gemini 与 Antigravity 保持此值。CCR 在可用时仍优先使用真实缓存签名；哨兵只是最后手段（Google 警告反复使用会降低工具调用质量） |
| `"none"` | 从不盖印哨兵 | 仅当端点**拒绝**该哨兵时（部分 Vertex 部署有此情况）。在真实签名再次可用之前，缺少缓存签名的工具重放会返回 400 |

### maxtoken

限制请求中的 max_tokens：

```json
{
  "transformers": [
    {
      "name": "maxtoken",
      "options": {
        "max_tokens": 8192
      },
      "models": ["deepseek,deepseek-chat"]
    }
  ]
}
```

### customparams

向请求中注入自定义参数：

```json
{
  "transformers": [
    {
      "name": "customparams",
      "options": {
        "include_reasoning": true,
        "custom_header": "value"
      }
    }
  ]
}
```

## 创建自定义转换器

### 简单转换器：修改请求

最简单的转换器只修改发送到提供商之前的请求。

**示例：为所有请求添加自定义头**

```javascript
// custom-header-transformer.js
module.exports = class CustomHeaderTransformer {
  name = 'custom-header';

  constructor(options) {
    this.headerName = options?.headerName || 'X-Custom-Header';
    this.headerValue = options?.headerValue || 'default-value';
  }

  async transformRequestIn(request, provider, context) {
    // 添加自定义头（将被 auth 方法使用）
    request._customHeaders = {
      [this.headerName]: this.headerValue
    };
    return request;
  }

  async auth(request, provider) {
    const headers = {
      'authorization': `Bearer ${provider.apiKey}`,
      ...request._customHeaders
    };
    return {
      body: request,
      config: { headers }
    };
  }
};
```

**在配置中使用：**

```json
{
  "transformers": [
    {
      "name": "custom-header",
      "path": "/path/to/custom-header-transformer.js",
      "options": {
        "headerName": "X-My-Header",
        "headerValue": "my-value"
      }
    }
  ]
}
```

### 中级转换器：请求/响应转换

此示例展示如何在不同 API 格式之间转换。

**示例：Mock API 格式转换器**

```javascript
// mockapi-transformer.js
module.exports = class MockAPITransformer {
  name = 'mockapi';
  endPoint = '/v1/chat';  // 自定义端点

  // 从 MockAPI 格式转换为统一格式
  async transformRequestOut(request, context) {
    const messages = request.conversation.map(msg => ({
      role: msg.sender,
      content: msg.text
    }));

    return {
      messages,
      model: request.model_id,
      max_tokens: request.max_tokens,
      temperature: request.temp
    };
  }

  // 从统一格式转换为 MockAPI 格式
  async transformRequestIn(request, provider, context) {
    return {
      model_id: request.model,
      conversation: request.messages.map(msg => ({
        sender: msg.role,
        text: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
      })),
      max_tokens: request.max_tokens || 4096,
      temp: request.temperature || 0.7
    };
  }

  // 将 MockAPI 响应转换为统一格式
  async transformResponseIn(response, context) {
    const data = await response.json();

    const unifiedResponse = {
      id: data.request_id,
      object: 'chat.completion',
      created: data.timestamp,
      model: data.model,
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: data.reply.text
        },
        finish_reason: data.stop_reason
      }],
      usage: {
        prompt_tokens: data.tokens.input,
        completion_tokens: data.tokens.output,
        total_tokens: data.tokens.input + data.tokens.output
      }
    };

    return new Response(JSON.stringify(unifiedResponse), {
      status: response.status,
      statusText: response.statusText,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
```

### 高级转换器：流式响应处理

此示例展示如何处理流式响应。

**示例：向流式响应添加自定义元数据**

```javascript
// streaming-metadata-transformer.js
module.exports = class StreamingMetadataTransformer {
  name = 'streaming-metadata';

  constructor(options) {
    this.metadata = options?.metadata || {};
    this.logger = null;  // 将由系统注入
  }

  async transformResponseOut(response, context) {
    const contentType = response.headers.get('Content-Type');

    // 处理流式响应
    if (contentType?.includes('text/event-stream')) {
      return this.transformStream(response, context);
    }

    // 处理非流式响应
    return response;
  }

  async transformStream(response, context) {
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();

    const transformedStream = new ReadableStream({
      start: async (controller) => {
        const reader = response.body.getReader();
        let buffer = '';

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (!line.trim() || !line.startsWith('data: ')) {
                controller.enqueue(encoder.encode(line + '\n'));
                continue;
              }

              const data = line.slice(6).trim();
              if (data === '[DONE]') {
                controller.enqueue(encoder.encode(line + '\n'));
                continue;
              }

              try {
                const chunk = JSON.parse(data);

                // 添加自定义元数据
                if (chunk.choices && chunk.choices[0]) {
                  chunk.choices[0].metadata = this.metadata;
                }

                // 记录日志以便调试
                this.logger?.debug({
                  chunk,
                  context: context.req.id
                }, '转换流式数据块');

                const modifiedLine = `data: ${JSON.stringify(chunk)}\n\n`;
                controller.enqueue(encoder.encode(modifiedLine));
              } catch (parseError) {
                // 如果解析失败，透传原始行
                controller.enqueue(encoder.encode(line + '\n'));
              }
            }
          }
        } catch (error) {
          this.logger?.error({ error }, '流式转换错误');
          controller.error(error);
        } finally {
          controller.close();
          reader.releaseLock();
        }
      }
    });

    return new Response(transformedStream, {
      status: response.status,
      statusText: response.statusText,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      }
    });
  }
};
```

### 真实示例：推理内容转换器

这是基于代码库中实际的 `reasoning.transformer.ts`。

```typescript
// reasoning-transformer.ts
import { Transformer, TransformerOptions } from "@caeliq/llms";

export class ReasoningTransformer implements Transformer {
  static TransformerName = "reasoning";
  enable: boolean;

  constructor(private readonly options?: TransformerOptions) {
    this.enable = this.options?.enable ?? true;
  }

  // 转换请求以添加推理参数
  async transformRequestIn(request: UnifiedChatRequest): Promise<UnifiedChatRequest> {
    if (!this.enable) {
      request.thinking = {
        type: "disabled",
        budget_tokens: -1,
      };
      request.enable_thinking = false;
      return request;
    }

    if (request.reasoning) {
      request.thinking = {
        type: "enabled",
        budget_tokens: request.reasoning.max_tokens,
      };
      request.enable_thinking = true;
    }
    return request;
  }

  // 转换响应以将 reasoning_content 转换为 thinking 格式
  async transformResponseOut(response: Response): Promise<Response> {
    if (!this.enable) return response;

    const contentType = response.headers.get("Content-Type");

    // 处理非流式响应
    if (contentType?.includes("application/json")) {
      const jsonResponse = await response.json();
      if (jsonResponse.choices[0]?.message.reasoning_content) {
        jsonResponse.thinking = {
          content: jsonResponse.choices[0].message.reasoning_content
        };
      }
      return new Response(JSON.stringify(jsonResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    // 处理流式响应
    if (contentType?.includes("stream")) {
      // [流式转换代码在这里]
      // 参见代码库中的完整实现
    }

    return response;
  }
}
```

## 转换器注册

### 方法 1：静态名称（基于类）

在 TypeScript/ES6 中创建转换器时使用：

```typescript
export class MyTransformer implements Transformer {
  static TransformerName = "my-transformer";

  async transformRequestIn(request: UnifiedChatRequest): Promise<any> {
    // 转换逻辑
    return request;
  }
}
```

### 方法 2：实例名称（基于实例）

用于 JavaScript 转换器：

```javascript
module.exports = class MyTransformer {
  constructor(options) {
    this.name = 'my-transformer';
    this.options = options;
  }

  async transformRequestIn(request, provider, context) {
    // 转换逻辑
    return request;
  }
};
```

## 应用转换器

### 全局应用（提供商级别）

为提供商的所有请求应用：

```json
{
  "Providers": [
    {
      "NAME": "deepseek",
      "HOST": "https://api.deepseek.com",
      "APIKEY": "your-api-key",
      "transformers": ["anthropic"]
    }
  ]
}
```

### 模型特定应用

仅应用于特定模型：

```json
{
  "transformers": [
    {
      "name": "maxtoken",
      "options": {
        "max_tokens": 8192
      },
      "models": ["deepseek,deepseek-chat"]
    }
  ]
}
```

注意：模型格式为 `provider,model`（例如 `deepseek,deepseek-chat`）。

### 全局转换器（所有提供商）

将转换器应用于所有提供商：

```json
{
  "transformers": [
    {
      "name": "custom-logger",
      "path": "/path/to/custom-logger.js"
    }
  ]
}
```

### 传递选项

某些转换器接受配置选项：

```json
{
  "transformers": [
    {
      "name": "maxtoken",
      "options": {
        "max_tokens": 8192
      }
    },
    {
      "name": "customparams",
      "options": {
        "custom_param_1": "value1",
        "custom_param_2": 42
      }
    }
  ]
}
```

## 最佳实践

### 1. 不可变性

始终创建新对象而不是修改现有对象：

```javascript
// 不好的做法
async transformRequestIn(request) {
  request.max_tokens = 4096;
  return request;
}

// 好的做法
async transformRequestIn(request) {
  return {
    ...request,
    max_tokens: request.max_tokens || 4096
  };
}
```

### 2. 错误处理

始终优雅地处理错误：

```javascript
async transformResponseIn(response) {
  try {
    const data = await response.json();
    // 处理数据
    return new Response(JSON.stringify(processedData), {
      status: response.status,
      headers: response.headers
    });
  } catch (error) {
    this.logger?.error({ error }, '转换失败');
    // 如果转换失败，返回原始响应
    return response;
  }
}
```

### 3. 日志记录

使用注入的 logger 进行调试：

```javascript
async transformRequestIn(request, provider, context) {
  this.logger?.debug({
    model: request.model,
    provider: provider.name
  }, '转换请求');

  // 转换逻辑

  return modifiedRequest;
}
```

### 4. 流处理

处理流式响应时，始终：
- 使用缓冲区处理不完整的数据块
- 正确释放 reader 锁
- 处理流中的错误
- 完成时关闭 controller

```javascript
const transformedStream = new ReadableStream({
  start: async (controller) => {
    const reader = response.body.getReader();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // 处理流...
      }
    } catch (error) {
      controller.error(error);
    } finally {
      controller.close();
      reader.releaseLock();
    }
  }
});
```

### 5. 上下文使用

`context` 参数包含有用信息：

```javascript
async transformRequestIn(request, provider, context) {
  // 访问请求 ID
  const requestId = context.req.id;

  // 访问原始请求
  const originalRequest = context.req.original;

  // 转换逻辑
}
```

## 测试转换器

### 手动测试

1. 将转换器添加到配置
2. 启动服务器：`ccr restart`
3. 检查日志：`tail -f ~/.claude-code-router/logs/ccr-*.log`
4. 发出测试请求
5. 验证输出

### 调试技巧

- 添加日志记录以跟踪转换步骤
- 使用流式和非流式请求进行测试
- 使用无效输入验证错误处理
- 检查错误时是否返回原始响应

## 下一步

- [高级主题](/docs/server/advanced/custom-router) - 高级路由自定义
- [Agents](/docs/server/advanced/agents) - 使用 agents 扩展
- [核心包](/docs/server/intro) - 了解 @caeliq/llms