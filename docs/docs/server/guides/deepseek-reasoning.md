---
sidebar_position: 4
---

# DeepSeek Reasoning Replay

Some DeepSeek-based providers (e.g., via OpenCode, ZenGo) require previous assistant **reasoning content** to be included in subsequent requests. This is because DeepSeek splits its output into visible content and internal reasoning/chain-of-thought, and the reasoning must be fed back into the conversation context for the model to maintain coherence across turns.

The `reasoning` transformer handles this automatically by capturing reasoning output from responses and replaying it in the next request.

## How It Works

1. On each response, the `reasoning` transformer intercepts `reasoning_content` fields (e.g., DeepSeek's `<reasoning_content>...</reasoning_content>` blocks)
2. It stores the reasoning text in the conversation context
3. On the next request to the same model, it injects the stored reasoning as a `reasoning` block in the messages array
4. This ensures the model receives its own prior reasoning as context

## Configuration

The `reasoning` transformer is typically configured per-model in your provider config:

```json
{
  "Providers": [
    {
      "name": "opencode",
      "baseUrl": "https://api.opencode.com/v1/chat/completions",
      "apiKey": "$OPENCODE_API_KEY",
      "models": ["deepseek-v4"],
      "transformer": {
        "use": ["opencode-headers"],
        "deepseek-v4": {
          "use": ["reasoning"]
        }
      }
    }
  ]
}
```

## Which Providers Need It

| Provider | Requires Reasoning Replay | Notes |
|----------|--------------------------|-------|
| DeepSeek (official) | ✅ | Must replay `reasoning_content` across turns |
| OpenCode | ✅ | Uses DeepSeek models with reasoning |
| ZenGo | ✅ | DeepSeek-based, requires reasoning replay |
| Other OpenAI-compatible | ❌ | Standard Chat Completions, no reasoning field |

## Combining with Other Transformers

The `reasoning` transformer should run **after** other request modifiers. Order matters in `transformer.use`:

```json
{
  "use": ["opencode-headers", "reasoning"]
}
```

The reasoning replay must see the final message list to inject the reasoning block in the right position.

## Troubleshooting

**Model forgets context across turns**: If the model repeats itself or loses track, the reasoning transformer may not be active. Verify it's in the model-level `use` array.

**Reasoning content visible to user**: The transformer only replays reasoning internally — it should not appear in Claude Code's output. If it does, check the `transformResponseOut` logic.
