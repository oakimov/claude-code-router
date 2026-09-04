/**
 * Encode upstream FIM responses to the **inbound** client wire.
 * Client response shape follows inbound kind (not outbound provider).
 * v1 inbound is mistral/Codestral; deepseek/qwen inbound reserved for later.
 */

import type { FimInboundKind } from "./kinds";
import { V1_FIM_INBOUND_KIND } from "./kinds";

function extractChoiceText(choice: any): string {
  if (typeof choice?.message?.content === "string") {
    return choice.message.content;
  }
  if (Array.isArray(choice?.message?.content)) {
    return choice.message.content
      .filter((p: any) => p?.type === "text" && typeof p.text === "string")
      .map((p: any) => p.text)
      .join("");
  }
  if (typeof choice?.text === "string") {
    return choice.text;
  }
  if (typeof choice?.delta?.content === "string") {
    return choice.delta.content;
  }
  if (typeof choice?.delta?.text === "string") {
    return choice.delta.text;
  }
  return "";
}

function normalizeUsage(usage: any): {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
} {
  const prompt =
    typeof usage?.prompt_tokens === "number" ? usage.prompt_tokens : 0;
  const completion =
    typeof usage?.completion_tokens === "number"
      ? usage.completion_tokens
      : 0;
  const total =
    typeof usage?.total_tokens === "number"
      ? usage.total_tokens
      : prompt + completion;
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: total,
  };
}

function encodeMistralClientJson(payload: any): Record<string, any> {
  const choicesIn = Array.isArray(payload?.choices) ? payload.choices : [];
  const choices = choicesIn.map((choice: any, index: number) => {
    const content = extractChoiceText(choice);
    const finish =
      typeof choice?.finish_reason === "string" && choice.finish_reason.length
        ? choice.finish_reason
        : "stop";
    const role =
      typeof choice?.message?.role === "string" && choice.message.role
        ? choice.message.role
        : "assistant";
    return {
      index: typeof choice?.index === "number" ? choice.index : index,
      message: { role, content },
      finish_reason: finish,
    };
  });

  const created =
    typeof payload?.created === "number"
      ? payload.created
      : Math.floor(Date.now() / 1000);

  return {
    id: typeof payload?.id === "string" ? payload.id : `fim-${created}`,
    object: "chat.completion",
    model: typeof payload?.model === "string" ? payload.model : "",
    created,
    usage: normalizeUsage(payload?.usage),
    choices,
  };
}

/** OpenAI-style text_completion (DeepSeek / Qwen completions FIM). */
function encodeTextCompletionClientJson(payload: any): Record<string, any> {
  const choicesIn = Array.isArray(payload?.choices) ? payload.choices : [];
  const choices = choicesIn.map((choice: any, index: number) => ({
    index: typeof choice?.index === "number" ? choice.index : index,
    text: extractChoiceText(choice),
    finish_reason:
      choice?.finish_reason === undefined ? null : choice.finish_reason,
  }));

  const created =
    typeof payload?.created === "number"
      ? payload.created
      : Math.floor(Date.now() / 1000);

  const out: Record<string, any> = {
    id: typeof payload?.id === "string" ? payload.id : `fim-${created}`,
    object: "text_completion",
    created,
    model: typeof payload?.model === "string" ? payload.model : "",
    choices,
  };
  if (payload?.usage != null) out.usage = payload.usage;
  return out;
}

/**
 * Normalize upstream JSON to the inbound client wire.
 * @param inboundKind — must match the kind used for inboundToUnifiedFim
 */
export function encodeFimResponseForInbound(
  payload: any,
  inboundKind: FimInboundKind = V1_FIM_INBOUND_KIND
): Record<string, any> {
  if (inboundKind === "mistral") {
    return encodeMistralClientJson(payload);
  }
  // deepseek / qwen inbound (future): text_completion + choices[].text
  return encodeTextCompletionClientJson(payload);
}

/** @deprecated Prefer encodeFimResponseForInbound(payload, inboundKind) */
export function normalizeToFimClientJson(
  payload: any,
  inboundKind: FimInboundKind = V1_FIM_INBOUND_KIND
): Record<string, any> {
  return encodeFimResponseForInbound(payload, inboundKind);
}

function encodeMistralSsePayload(parsed: any): string {
  const choices = parsed.choices.map((choice: any, index: number) => {
    const content = extractChoiceText(choice);
    return {
      index: typeof choice?.index === "number" ? choice.index : index,
      delta: { content },
      finish_reason:
        choice?.finish_reason === undefined || choice?.finish_reason === null
          ? null
          : choice.finish_reason,
    };
  });
  return JSON.stringify({
    ...parsed,
    object: "chat.completion.chunk",
    choices,
  });
}

function encodeTextCompletionSsePayload(parsed: any): string {
  const choices = parsed.choices.map((choice: any, index: number) => {
    const text = extractChoiceText(choice);
    const next: Record<string, unknown> = {
      index: typeof choice?.index === "number" ? choice.index : index,
      text,
      finish_reason:
        choice?.finish_reason === undefined || choice?.finish_reason === null
          ? null
          : choice.finish_reason,
    };
    if (choice?.delta && typeof choice.delta === "object") {
      next.delta = { text };
    }
    return next;
  });
  return JSON.stringify({
    ...parsed,
    object: "text_completion",
    choices,
  });
}

/**
 * Map one SSE data payload to the inbound client wire.
 */
export function normalizeFimSseDataPayload(
  dataStr: string,
  inboundKind: FimInboundKind = V1_FIM_INBOUND_KIND
): string {
  const trimmed = dataStr.trim();
  if (!trimmed || trimmed === "[DONE]") return dataStr;
  try {
    const parsed = JSON.parse(trimmed);
    if (!Array.isArray(parsed?.choices)) return dataStr;
    if (inboundKind === "mistral") {
      return encodeMistralSsePayload(parsed);
    }
    return encodeTextCompletionSsePayload(parsed);
  } catch {
    return dataStr;
  }
}
