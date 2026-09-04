/**
 * Unified FIM intermediate (prompt + optional suffix + sampling).
 * Types live next to FIM utils — not a separate types/fim package.
 *
 * Client response wire follows **inbound** kind (see encodeFimResponseForInbound).
 * v1 inbound is mistral/Codestral chat.completion; other kinds reserved.
 */

export interface UnifiedFimRequest {
  model: string;
  prompt: string;
  suffix?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  stream?: boolean;
  min_tokens?: number;
  random_seed?: number;
}

/** Choice on mistral/Codestral inbound client wire. */
export interface UnifiedFimChoice {
  index: number;
  message: { role: string; content: string };
  finish_reason: string;
}

export interface UnifiedFimResponse {
  id: string;
  object: "chat.completion";
  model: string;
  created: number;
  choices: UnifiedFimChoice[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  [key: string]: unknown;
}
