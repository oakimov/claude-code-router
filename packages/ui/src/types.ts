export interface ProviderTransformer {
  use: (string | (string | Record<string, unknown> | { max_tokens: number })[])[];
  [key: string]: any; // Allow for model-specific transformers
}

export interface Provider {
  name: string;
  api_base_url: string;
  api_key: string;
  models: string[];
  transformer?: ProviderTransformer;
  display_name?: string;
  description?: string;
  icon?: string;
  tags?: string[];
}

export interface RouterConfig {
    default: string;
    background: string;
    think: string;
    longContext: string;
    longContextThreshold: number;
    webSearch: string;
    image: string;
    fim: string;
    custom?: any;
}

export interface Transformer {
    name?: string;
    path: string;
    options?: Record<string, any>;
}

export interface StatusLineModuleConfig {
  type: string;
  icon?: string;
  text: string;
  color?: string;
  background?: string;
  scriptPath?: string; // used for script type modules, specifies the path to the Node.js script file to execute
}

export interface StatusLineThemeConfig {
  modules: StatusLineModuleConfig[];
}

export interface StatusLineConfig {
  enabled: boolean;
  currentStyle: string;
  default: StatusLineThemeConfig;
  powerline: StatusLineThemeConfig;
  fontFamily?: string;
}

export interface Config {
  Providers: Provider[];
  Router: RouterConfig;
  transformers: Transformer[];
  StatusLine?: StatusLineConfig;
  forceUseImageAgent?: boolean;
  // Top-level settings
  LOG: boolean;
  LOG_LEVEL: string;
  CLAUDE_PATH: string;
  HOST: string;
  PORT: number;
  APIKEY: string;
  API_TIMEOUT_MS: string;
  PROXY_URL: string;
  CUSTOM_ROUTER_PATH?: string;
  /**
   * Preserve arbitrary top-level keys from config.json (e.g. MISTRAL_API_KEY).
   * These are interpolated into process.env by the server and must survive UI load/save.
   */
  [key: string]: unknown;
}

export type AccessLevel = 'restricted' | 'full';

/** Current process vitals, as reported by the server's health heartbeat. */
export interface HealthSnapshot {
  uptimeMs: number;
  windowMs: number;
  memory: {
    rss: number;
    rssDelta?: number;
    heapUsed: number;
    heapTotal: number;
    external: number;
    systemTotal: number;
    systemFree: number;
    constrained: boolean;
  };
  load: {
    avg: [number, number, number];
    cpus: number;
    processCores?: number;
    eventLoopMeanMs: number;
    eventLoopP99Ms: number;
  };
  sessions: {
    running: number;
    activeInWindow: number;
  };
  requests: {
    inFlight: number;
    oldestInFlightMs?: number;
    completed: number;
    failed: number;
    p50Ms?: number;
    p95Ms?: number;
    byProvider: Array<{ provider: string; ok: number; failed: number }>;
  };
  cache: {
    promptTokens: number;
    cachedTokens: number;
    writtenTokens: number;
    hitRatio?: number;
  };
}

export interface HealthVitals {
  version: number;
  pid: number;
  node: string;
  updatedAt: number;
  intervalMs: number;
  current: HealthSnapshot;
}

/** Response of `GET /health`; `vitals` is absent on older servers. */
export interface HealthResponse {
  status: string;
  timestamp: string;
  vitals?: HealthVitals;
}
