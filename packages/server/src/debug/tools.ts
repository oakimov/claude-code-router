import type { OpenAiFunctionToolSpec } from "./types";

const TOOL_ID = /^[A-Za-z0-9_-]{1,64}$/;

export const STUB_TOOL_RESULT = { ok: true, stub: true } as const;

/**
 * Map OpenAI function-tool JSON into inspect-only descriptors.
 * User JSON is never evaluated as code — execute is always a stub.
 */
export function parseOpenAiTools(raw: unknown): OpenAiFunctionToolSpec[] {
  let value = raw;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return [];
    try {
      value = JSON.parse(trimmed);
    } catch {
      throw new Error("Tools JSON is not valid JSON");
    }
  }
  if (value == null || value === "") return [];

  let list: unknown[] = [];
  if (Array.isArray(value)) {
    list = value;
  } else if (typeof value === "object" && Array.isArray((value as any).tools)) {
    list = (value as any).tools;
  } else if (typeof value === "object") {
    list = [value];
  } else {
    throw new Error("Tools must be a JSON array of function tools");
  }

  const specs: OpenAiFunctionToolSpec[] = [];
  for (const item of list) {
    if (!item || typeof item !== "object") continue;
    const rec = item as Record<string, any>;
    const fn = rec.function && typeof rec.function === "object" ? rec.function : rec;
    const id = String(fn.name || rec.id || rec.name || "").trim();
    if (!id) continue;
    if (!TOOL_ID.test(id)) {
      throw new Error(`Invalid tool name "${id}". Use letters, digits, underscore, or hyphen.`);
    }
    const description = String(fn.description || rec.description || id);
    const parameters =
      fn.parameters && typeof fn.parameters === "object"
        ? fn.parameters
        : rec.parameters && typeof rec.parameters === "object"
          ? rec.parameters
          : { type: "object", properties: {} };
    specs.push({
      id,
      description,
      parameters: parameters as Record<string, unknown>,
    });
  }
  return specs;
}

/** Inspect-only tool result. Never runs user-supplied code. */
export function stubToolExecute(args: unknown): {
  ok: true;
  stub: true;
  args: unknown;
} {
  return { ok: true, stub: true, args };
}
