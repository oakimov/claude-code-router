import type { UnifiedFimRequest } from "./types";

const DEEPSEEK_FIM_MAX_TOKENS = 4096;

/** Qwen / HF FIM markers for completions prompt. */
export function buildQwenFimPrompt(prompt: string, suffix?: string): string {
  if (typeof suffix === "string") {
    return `<|fim_prefix|>${prompt}<|fim_suffix|>${suffix}<|fim_middle|>`;
  }
  return `<|fim_prefix|>${prompt}<|fim_suffix|>`;
}

/** Sampling fields shared across FIM outbound bodies. */
export function pickFimSamplingFields(
  unified: UnifiedFimRequest
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (unified.max_tokens !== undefined) out.max_tokens = unified.max_tokens;
  if (unified.temperature !== undefined) out.temperature = unified.temperature;
  if (unified.top_p !== undefined) out.top_p = unified.top_p;
  if (unified.stream !== undefined) out.stream = unified.stream;
  if (unified.stop !== undefined) out.stop = unified.stop;
  if (unified.min_tokens !== undefined) out.min_tokens = unified.min_tokens;
  if (unified.random_seed !== undefined) out.random_seed = unified.random_seed;
  return out;
}

/** Native prompt+suffix body (Mistral / DeepSeek field shape). */
export function encodePromptSuffixBody(
  unified: UnifiedFimRequest,
  options?: { clampMaxTokens?: number; disableThinking?: boolean }
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: unified.model,
    prompt: unified.prompt,
    ...pickFimSamplingFields(unified),
  };
  if (typeof unified.suffix === "string") {
    body.suffix = unified.suffix;
  }
  if (
    options?.clampMaxTokens !== undefined &&
    typeof body.max_tokens === "number" &&
    body.max_tokens > options.clampMaxTokens
  ) {
    body.max_tokens = options.clampMaxTokens;
  }
  if (options?.disableThinking) {
    body.thinking = { type: "disabled" };
  }
  return body;
}

export function encodeDeepseekFimBody(
  unified: UnifiedFimRequest
): Record<string, unknown> {
  return encodePromptSuffixBody(unified, {
    clampMaxTokens: DEEPSEEK_FIM_MAX_TOKENS,
    disableThinking: true,
  });
}

export function encodeQwenFimBody(
  unified: UnifiedFimRequest
): Record<string, unknown> {
  const sampling = pickFimSamplingFields(unified);
  // Qwen completions: no separate suffix field; tokens live in prompt.
  delete sampling.min_tokens;
  delete sampling.random_seed;
  return {
    model: unified.model,
    prompt: buildQwenFimPrompt(unified.prompt, unified.suffix),
    ...sampling,
  };
}

export { DEEPSEEK_FIM_MAX_TOKENS };
