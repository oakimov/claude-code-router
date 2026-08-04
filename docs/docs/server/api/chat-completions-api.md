---
title: Chat Completions API
---

# Chat Completions API

CCR accepts OpenAI Chat Completions requests at:

- `POST /v1/chat/completions`
- `POST /chat/completions`

Both paths also work below a preset namespace. Authentication uses `Authorization: Bearer <APIKEY>` or `x-api-key: <APIKEY>`.

```bash
curl http://127.0.0.1:3456/v1/chat/completions \
  -H "Authorization: Bearer $CCR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai,gpt-5-mini",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

Use `provider,model` for an explicit destination. A bare model is routed through `Router.default` and scenario rules; CCR returns a 400 if routing does not resolve it to a provider.

The compatibility tier supports one text output, system/developer/user/assistant/tool messages, text and `image_url` input, function tools and results, sampling fields, token limits, JSON replies, and SSE. Streaming ends with `data: [DONE]`. Unsupported state, audio, structured output, multiple choices, log probabilities, and other semantics return an OpenAI-shaped 400 error.
