---
title: Responses API
---

# Responses API

CCR accepts OpenAI Responses requests at:

- `POST /v1/responses`
- `POST /responses`

Configure an OpenAI SDK with:

```shell
export OPENAI_BASE_URL=http://127.0.0.1:3456/v1
export OPENAI_API_KEY=your-router-api-key
```

The route supports string and message-array input, instructions, text/images by URL or data URL, function calls/results, function tools, reasoning effort/summary intent, token limits, usage, JSON responses, and ordered `response.*` SSE events. Function-call IDs are normalized to the Responses length and character limits while remaining correlated within the turn.

CCR does not own OpenAI conversation state. It rejects `store: true`, `previous_response_id`, `conversation`, and background execution. Provider file IDs and hosted tools that cannot be represented safely through the selected provider are also rejected. Stream success closes after `response.completed`; stream failures emit `response.failed`. Chat Completions' `[DONE]` marker is not part of the client Responses stream.
