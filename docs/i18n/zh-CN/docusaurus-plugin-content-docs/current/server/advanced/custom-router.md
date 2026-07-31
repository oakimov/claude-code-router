---
title: 自定义路由器
sidebar_position: 1
---

# 自定义路由器

使用 JavaScript 编写自己的路由逻辑。

## 创建自定义路由器

创建一个导出路由函数的 JavaScript 文件：

```javascript
// custom-router.js
module.exports = async function(req, config) {
  // 获取用户消息
  const userMessage = req.body.messages.find(m => m.role === 'user')?.content;

  // 自定义逻辑
  if (userMessage && userMessage.includes('解释代码')) {
    return 'openrouter,anthropic/claude-3.5-sonnet';
  }

  // 返回 null 以使用默认路由
  return null;
};
```

## 参数说明

路由函数接收以下参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `req` | object | 来自 Claude Code 的请求对象，包含请求体 |
| `config` | object | 应用程序的配置对象 |

## 配置

在 `config.json` 中设置 `CUSTOM_ROUTER_PATH` 以使用您的自定义路由器：

```json
{
  "CUSTOM_ROUTER_PATH": "/path/to/custom-router.js"
}
```

## 返回格式

路由函数应返回以下格式的字符串：

```
{provider-name},{model-name}
```

示例：

```
deepseek,deepseek-chat
```

如果返回 `null`，则回退到默认路由配置。

## 错误处理

如果路由函数抛出错误或返回无效格式，路由器将回退到默认路由配置。

## 示例：基于时间的路由

```javascript
module.exports = async function(req, config) {
  const hour = new Date().getHours();

  // 工作时间使用更快的模型
  if (hour >= 9 && hour <= 18) {
    return 'groq,llama-3.3-70b-versatile';
  }

  // 非工作时间使用更强大的模型
  return 'deepseek,deepseek-chat';
};
```

## 示例：成本优化

```javascript
module.exports = async function(req, config) {
  const userMessage = req.body.messages.find(m => m.role === 'user')?.content;

  // 简单任务使用较便宜的模型
  if (userMessage && userMessage.length < 100) {
    return 'groq,llama-3.3-70b-versatile';
  }

  // 复杂任务使用默认模型
  return null;
};
```

## 示例：任务类型路由

```javascript
module.exports = async function(req, config) {
  const userMessage = req.body.messages.find(m => m.role === 'user')?.content;

  if (!userMessage) return null;

  // 代码相关任务
  if (userMessage.includes('代码') || userMessage.includes('code')) {
    return 'deepseek,deepseek-coder';
  }

  // 解释任务
  if (userMessage.includes('解释') || userMessage.includes('explain')) {
    return 'openrouter,anthropic/claude-3.5-sonnet';
  }

  // 默认
  return null;
};
```

## 测试您的路由器

通过检查日志来测试您的自定义路由器：

```bash
tail -f ~/.claude-code-router/claude-code-router.log
```

查找路由决策以查看正在选择哪个模型。

## 子代理路由

Claude Code 子代理请求可通过以下方式路由：

1. 在 system 或 message 文本中使用显式标签 `<CCR-SUBAGENT-MODEL>provider,model</CCR-SUBAGENT-MODEL>`（发送上游前会移除该标签）。也支持 `provider/model`。
2. 或在 Claude Code 将请求标记为子代理时，使用环境变量 `CLAUDE_CODE_SUBAGENT_MODEL=provider,model`。

标签优先于环境变量。若两者都未设置，则继续使用常规 Router 规则。

**示例：**

```
<CCR-SUBAGENT-MODEL>openrouter,anthropic/claude-3.5-sonnet</CCR-SUBAGENT-MODEL>
请帮我分析这段代码是否存在潜在的优化空间...
```

```bash
export CLAUDE_CODE_SUBAGENT_MODEL="openrouter,anthropic/claude-3.5-sonnet"
```

## 下一步

- [Agent](/docs/server/config/transformers) - 使用 Agent 扩展功能
- [预设](/docs/presets/intro) - 使用预定义配置