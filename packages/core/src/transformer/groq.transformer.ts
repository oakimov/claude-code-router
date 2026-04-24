import { UnifiedChatRequest } from "@/types/llm";
import { Transformer } from "../types/transformer";
import { createSSEStreamReader, StreamContext, encodeSSEData, encodeSSELine } from "../utils/stream";
import { stripMessagesCacheControl } from "../utils/cacheControl";
import { v4 as uuidv4 } from "uuid";

export class GroqTransformer implements Transformer {
  name = "groq";

  async transformRequestIn(request: UnifiedChatRequest): Promise<UnifiedChatRequest> {
    request.messages = stripMessagesCacheControl(request.messages);

    if (Array.isArray(request.tools)) {
      request.tools.forEach(tool => {
        delete tool.function.parameters.$schema;
      });
    }
    return request;
  }

  async transformResponseOut(response: Response): Promise<Response> {
    if (response.headers.get("Content-Type")?.includes("application/json")) {
      const jsonResponse = await response.json();
      return new Response(JSON.stringify(jsonResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    } else if (response.headers.get("Content-Type")?.includes("stream")) {
      if (!response.body) return response;

      let hasTextContent = false;

      return createSSEStreamReader(response, (line: string, ctx: StreamContext) => {
        if (!line.trim()) {
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
          return;
        }

        if (!line.startsWith("data:") || line.trim() === "data: [DONE]") {
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
          return;
        }

        try {
          const jsonStr = line.slice(5).trim();
          const data = JSON.parse(jsonStr);
          if (data.error) {
            throw new Error(JSON.stringify(data));
          }

          if (data.choices?.[0]?.delta?.content && !hasTextContent) {
            hasTextContent = true;
          }

          if (data.choices?.[0]?.delta?.tool_calls?.length) {
            data.choices[0].delta.tool_calls.forEach((tool: any) => {
              tool.id = `call_${uuidv4()}`;
            });
          }

          if (
            data.choices?.[0]?.delta?.tool_calls?.length &&
            hasTextContent
          ) {
            if (typeof data.choices[0].index === "number") {
              data.choices[0].index += 1;
            } else {
              data.choices[0].index = 1;
            }
          }

          ctx.controller.enqueue(encodeSSEData(JSON.stringify(data), ctx.encoder));
        } catch {
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
        }
      });
    }

    return response;
  }
}
