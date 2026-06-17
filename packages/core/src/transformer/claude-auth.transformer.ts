import { UnifiedChatRequest } from "@/types/llm";
import { Transformer } from "@/types/transformer";
import { getValidAccessToken } from "../utils/claude-auth";
import { transformResponseOut } from "../utils/vertex-claude.util";

function convertUnifiedToAnthropic(request: UnifiedChatRequest): Record<string, any> {
  const system: any[] = [];
  const messages: any[] = [];

  for (const msg of request.messages) {
    if (msg.role === "system") {
      if (typeof msg.content === "string") {
        system.push({ type: "text", text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === "text") {
            const block: any = { type: "text", text: part.text };
            if ((part as any).cache_control) block.cache_control = (part as any).cache_control;
            system.push(block);
          }
        }
      }
      continue;
    }

    if (msg.role === "tool") {
      const toolResult: any = {
        type: "tool_result",
        tool_use_id: msg.tool_call_id,
        content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
      };
      // Append to the preceding user message if it already has tool_result blocks
      const last = messages[messages.length - 1];
      if (last?.role === "user" && Array.isArray(last.content)) {
        last.content.push(toolResult);
      } else {
        messages.push({ role: "user", content: [toolResult] });
      }
      continue;
    }

    if (msg.role === "assistant") {
      const content: any[] = [];
      if (typeof msg.content === "string" && msg.content) {
        content.push({ type: "text", text: msg.content });
      }
      if (msg.tool_calls?.length) {
        for (const tc of msg.tool_calls) {
          let input: Record<string, any> = {};
          try {
            input = JSON.parse(tc.function.arguments || "{}");
          } catch {}
          content.push({ type: "tool_use", id: tc.id, name: tc.function.name, input });
        }
      }
      if (content.length > 0) messages.push({ role: "assistant", content });
      continue;
    }

    if (msg.role === "user") {
      const content: any[] = [];
      if (typeof msg.content === "string") {
        content.push({ type: "text", text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === "text") {
            content.push({ type: "text", text: part.text });
          } else if (part.type === "image_url") {
            const url = (part as any).image_url?.url ?? "";
            if (url.startsWith("data:")) {
              const [meta, data] = url.split(",");
              const mediaType = meta.split(":")[1]?.split(";")[0] ?? "image/jpeg";
              content.push({ type: "image", source: { type: "base64", media_type: mediaType, data } });
            } else {
              content.push({ type: "image", source: { type: "url", url } });
            }
          }
        }
      }
      if (content.length > 0) messages.push({ role: "user", content });
    }
  }

  const body: Record<string, any> = {
    model: request.model,
    max_tokens: request.max_tokens ?? 8192,
    messages,
    stream: request.stream ?? true,
  };

  if (system.length === 1 && !system[0].cache_control) {
    body.system = system[0].text;
  } else if (system.length > 0) {
    body.system = system;
  }

  if (request.temperature !== undefined) body.temperature = request.temperature;

  if (request.tools?.length) {
    body.tools = request.tools.map((t) => ({
      name: t.function.name,
      description: t.function.description,
      input_schema: t.function.parameters,
    }));
  }

  if (request.tool_choice) {
    const tc = request.tool_choice as any;
    if (tc === "auto") {
      body.tool_choice = { type: "auto" };
    } else if (tc === "none") {
      body.tool_choice = { type: "none" };
    } else if (typeof tc === "string") {
      body.tool_choice = { type: "tool", name: tc };
    } else if (tc.type === "function") {
      body.tool_choice = { type: "tool", name: tc.function?.name };
    }
  }

  return body;
}

export class ClaudeAuthTransformer implements Transformer {
  name = "claude-auth";
  logger?: any;

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: any
  ): Promise<Record<string, any>> {
    const creds = await getValidAccessToken();
    const baseUrl = provider?.api_base_url ?? provider?.baseUrl ?? "https://api.anthropic.com";
    const url = baseUrl.endsWith("/v1/messages")
      ? baseUrl
      : `${baseUrl.replace(/\/$/, "")}/v1/messages`;

    return {
      body: convertUnifiedToAnthropic(request),
      config: {
        url,
        headers: {
          Authorization: `Bearer ${creds.access_token}`,
          "anthropic-version": "2023-06-01",
          "Content-Type": "application/json",
          "User-Agent": "claude-cli/2.1.179 (external, cli)",
        },
      },
    };
  }

  async transformResponseOut(response: Response): Promise<Response> {
    return transformResponseOut(response, this.name, this.logger);
  }
}
