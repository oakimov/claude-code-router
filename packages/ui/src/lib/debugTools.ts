const SESSION_KEY = "ccr.debug.tools";
export const BUILTIN_WEATHER_NAME = "get_weather";

export type DebugTool = {
  type: "function";
  enabled?: boolean;
  function: {
    name: string;
    description?: string;
    parameters?: unknown;
    [key: string]: unknown;
  };
};

export const DEFAULT_WEATHER_TOOL: DebugTool = {
  type: "function",
  function: {
    name: BUILTIN_WEATHER_NAME,
    description: "Get the current weather for a city",
    parameters: {
      type: "object",
      properties: {
        city: { type: "string", description: "City name" },
      },
      required: ["city"],
    },
  },
};

export const SAMPLE_DEBUG_TOOL: DebugTool = {
  type: "function",
  function: {
    name: "sample_tool",
    description: "Example function tool. Edit the name, description, and parameters.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Input query" },
      },
      required: ["query"],
    },
  },
};

export function toolName(tool: DebugTool | null | undefined): string {
  return String(tool?.function?.name || "").trim();
}

export function isBuiltinWeatherTool(tool: DebugTool | null | undefined): boolean {
  return toolName(tool) === BUILTIN_WEATHER_NAME;
}

export function isToolEnabled(tool: DebugTool | null | undefined): boolean {
  return tool?.enabled !== false;
}

export function toolParameterNames(tool: DebugTool): string[] {
  const parameters = tool.function?.parameters;
  if (!parameters || typeof parameters !== "object") return [];
  const properties = (parameters as { properties?: Record<string, unknown> }).properties;
  if (!properties || typeof properties !== "object") return [];
  return Object.keys(properties);
}

function isDebugTool(value: unknown): value is DebugTool {
  if (!value || typeof value !== "object") return false;
  const rec = value as Record<string, any>;
  const fn = rec.function && typeof rec.function === "object" ? rec.function : rec;
  return typeof fn.name === "string" && fn.name.trim().length > 0;
}

function normalizeTool(value: unknown): DebugTool | null {
  if (!isDebugTool(value)) return null;
  const rec = value as Record<string, any>;
  const fn = rec.function && typeof rec.function === "object" ? rec.function : rec;
  return {
    type: "function",
    enabled: rec.enabled !== false,
    function: {
      ...fn,
      name: String(fn.name).trim(),
      description: fn.description || "",
      parameters: fn.parameters || rec.parameters || rec.input_schema || {
        type: "object",
        properties: {},
      },
    },
  };
}

export function parseToolDefinition(raw: string): DebugTool | null {
  try {
    return normalizeTool(JSON.parse(raw));
  } catch {
    return null;
  }
}

export function stringifyTool(tool: DebugTool): string {
  return JSON.stringify(
    {
      type: "function",
      function: tool.function,
    },
    null,
    2
  );
}

export function toRequestTools(tools: DebugTool[]): Array<{
  type: "function";
  function: DebugTool["function"];
}> {
  return tools.filter(isToolEnabled).map((tool) => ({
    type: "function",
    function: tool.function,
  }));
}

function withBuiltinWeather(tools: DebugTool[]): DebugTool[] {
  if (tools.some(isBuiltinWeatherTool)) return tools;
  return [structuredClone(DEFAULT_WEATHER_TOOL), ...tools];
}

export function parseToolsFile(raw: string): DebugTool[] {
  const parsed = JSON.parse(raw);
  const list: unknown[] = Array.isArray(parsed)
    ? parsed
    : parsed &&
        typeof parsed === "object" &&
        Array.isArray((parsed as { tools?: unknown[] }).tools)
      ? (parsed as { tools: unknown[] }).tools
      : parsed && typeof parsed === "object"
        ? [parsed]
        : (() => {
            throw new Error("Tools file must be a JSON array or object");
          })();
  return list.map(normalizeTool).filter((tool): tool is DebugTool => tool != null);
}

export function mergeDebugTools(existing: DebugTool[], incoming: DebugTool[]): DebugTool[] {
  const next = [...existing];
  const indexByName = new Map(next.map((tool, index) => [toolName(tool), index]));
  for (const tool of incoming) {
    const name = toolName(tool);
    if (!name) continue;
    const index = indexByName.get(name);
    if (index == null) {
      indexByName.set(name, next.length);
      next.push(tool);
    } else {
      next[index] = tool;
    }
  }
  return withBuiltinWeather(next);
}

export function loadDebugTools(): DebugTool[] {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return [structuredClone(DEFAULT_WEATHER_TOOL)];
    const parsed = JSON.parse(raw);
    const list = (Array.isArray(parsed) ? parsed : [])
      .map(normalizeTool)
      .filter((tool): tool is DebugTool => tool != null);
    return withBuiltinWeather(list);
  } catch {
    return [structuredClone(DEFAULT_WEATHER_TOOL)];
  }
}

export function saveDebugTools(tools: DebugTool[]): void {
  try {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(withBuiltinWeather(tools)));
  } catch {
    // Ignore quota / private-mode failures.
  }
}

export function nextSampleTool(existing: DebugTool[]): DebugTool {
  const names = new Set(existing.map(toolName));
  const sample = structuredClone(SAMPLE_DEBUG_TOOL);
  if (!names.has(sample.function.name)) return sample;
  let index = 2;
  while (names.has(`sample_tool_${index}`)) index += 1;
  sample.function.name = `sample_tool_${index}`;
  return sample;
}
