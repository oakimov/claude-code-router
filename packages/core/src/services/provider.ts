import { TransformerConstructor } from "@/types/transformer";
import {
  LLMProvider,
  RegisterProviderRequest,
  ModelRoute,
  ProviderUpdate,
  RequestRouteInfo,
  RequestScopedErrorsCarrier,
  ConfigProvider,
  TransformerConfigEntry,
} from "../types/llm";
import {
  compiledScopedErrorRules,
  readScopedErrorRulesFromCarrier,
  SCOPED_ERROR_RULE_KEYS,
  type RequestScopedErrorRule,
} from "@/utils/request-scoped-errors";
import { ConfigService } from "./config";
import { TransformerService } from "./transformer";

/** True when the carrier sets a rule list under any accepted spelling. */
function carriesScopedErrorRules(carrier: RequestScopedErrorsCarrier): boolean {
  return readScopedErrorRulesFromCarrier(carrier) !== undefined;
}

/** Copy of `source` without any rule-list spelling. */
function withoutScopedErrorRuleKeys<T extends object>(
  source: T
): Omit<T, keyof RequestScopedErrorsCarrier> {
  const copy = { ...source } as Record<string, unknown>;
  for (const key of SCOPED_ERROR_RULE_KEYS) delete copy[key];
  return copy as Omit<T, keyof RequestScopedErrorsCarrier>;
}

export class ProviderService {
  private providers: Map<string, LLMProvider> = new Map();
  private modelRoutes: Map<string, ModelRoute> = new Map();

  constructor(private readonly configService: ConfigService, private readonly transformerService: TransformerService, private readonly logger: any) {
    this.initializeCustomProviders();
  }

  private initializeCustomProviders() {
    const providersConfig =
      this.configService.get<ConfigProvider[]>("providers");
    if (providersConfig && Array.isArray(providersConfig)) {
      this.initializeFromProvidersArray(providersConfig);
      return;
    }
  }

  private initializeFromProvidersArray(providersConfig: ConfigProvider[]) {
    providersConfig.forEach((providerConfig: ConfigProvider) => {
      try {
        if (
          !providerConfig.name ||
          !providerConfig.api_base_url ||
          !providerConfig.api_key
        ) {
          return;
        }

        const transformer: LLMProvider["transformer"] = {}

        if (providerConfig.transformer) {
          Object.keys(providerConfig.transformer).forEach(key => {
            if (key === 'use') {
              if (Array.isArray(providerConfig.transformer.use)) {
                transformer.use = providerConfig.transformer.use.map((transformer) => {
                  if (Array.isArray(transformer) && typeof transformer[0] === 'string') {
                    const Constructor = this.transformerService.getTransformer(transformer[0]);
                    if (Constructor) {
                      return this.attachTransformerLogger(
                        new (Constructor as TransformerConstructor)(transformer[1])
                      );
                    }
                  }
                  if (typeof transformer === 'string') {
                    const transformerInstance = this.transformerService.getTransformer(transformer);
                    if (typeof transformerInstance === 'function') {
                      return this.attachTransformerLogger(new transformerInstance());
                    }
                    return this.attachTransformerLogger(transformerInstance);
                  }
                }).filter((transformer) => typeof transformer !== 'undefined');
              }
            } else if (key === 'passthrough') {
              transformer.passthrough = providerConfig.transformer.passthrough;
            } else {
              if (Array.isArray(providerConfig.transformer[key]?.use)) {
                transformer[key] = {
                  use: providerConfig.transformer[key].use.map((transformer: TransformerConfigEntry) => {
                    if (Array.isArray(transformer) && typeof transformer[0] === 'string') {
                      const Constructor = this.transformerService.getTransformer(transformer[0]);
                      if (Constructor) {
                        return this.attachTransformerLogger(
                          new (Constructor as TransformerConstructor)(transformer[1])
                        );
                      }
                    }
                    if (typeof transformer === 'string') {
                      const transformerInstance = this.transformerService.getTransformer(transformer);
                      if (typeof transformerInstance === 'function') {
                        return this.attachTransformerLogger(new transformerInstance());
                      }
                      return this.attachTransformerLogger(transformerInstance);
                    }
                  }).filter((transformer: unknown) => typeof transformer !== 'undefined')
                }
              }
            }
          })
        }

        this.registerProvider({
          name: providerConfig.name,
          baseUrl: providerConfig.api_base_url,
          apiKey: providerConfig.api_key,
          models: providerConfig.models || [],
          project_id: providerConfig.project_id,
          request_scoped_errors: readScopedErrorRulesFromCarrier(
            providerConfig
          ) as RequestScopedErrorRule[] | undefined,
          transformer: providerConfig.transformer ? transformer : undefined,
        });

        this.logger.info(`${providerConfig.name} provider registered`);
      } catch (error) {
        this.logger.error(`${providerConfig.name} provider registered error: ${error}`);
      }
    });
  }

  /** Match TransformerService: instances used in provider use[] need the service logger. */
  private attachTransformerLogger<T>(instance: T): T {
    if (instance && typeof instance === "object") {
      (instance as any).logger = this.logger;
    }
    return instance;
  }

  /**
   * Store request-scoped error rules under the canonical key only, so an
   * update under any spelling cannot be shadowed by a stale alias. Rules are
   * validated here to surface config mistakes at registration time.
   */
  private withCanonicalScopedErrorRules<T extends object>(
    base: T,
    rules: unknown
  ): Omit<T, keyof RequestScopedErrorsCarrier> & {
    request_scoped_errors?: RequestScopedErrorRule[];
  } {
    const provider: Omit<T, keyof RequestScopedErrorsCarrier> & {
      request_scoped_errors?: RequestScopedErrorRule[];
    } = withoutScopedErrorRuleKeys(base);
    if (rules === undefined || rules === null) return provider;
    const name = (base as { name?: unknown }).name;
    compiledScopedErrorRules(rules, (issue) => {
      this.logger?.warn?.(
        { provider: name, index: issue.index, reason: issue.reason },
        `request_scoped_errors rule ignored for provider ${String(name)}: ${issue.reason}`
      );
    });
    provider.request_scoped_errors = rules as RequestScopedErrorRule[];
    return provider;
  }

  registerProvider(request: RegisterProviderRequest): LLMProvider {
    const provider: LLMProvider = this.withCanonicalScopedErrorRules(
      request,
      readScopedErrorRulesFromCarrier(request)
    );

    this.providers.set(provider.name, provider);

    request.models.forEach((model) => {
      const fullModel = `${provider.name},${model}`;
      const route: ModelRoute = {
        provider: provider.name,
        model,
        fullModel,
      };
      this.modelRoutes.set(fullModel, route);
      if (!this.modelRoutes.has(model)) {
        this.modelRoutes.set(model, route);
      }
    });

    return provider;
  }

  getProviders(): LLMProvider[] {
    return Array.from(this.providers.values());
  }

  getProvider(name: string): LLMProvider | undefined {
    return this.providers.get(name);
  }

  updateProvider(
    id: string,
    updates: ProviderUpdate
  ): LLMProvider | null {
    const provider = this.providers.get(id);
    if (!provider) {
      return null;
    }

    const updatedProvider = this.withCanonicalScopedErrorRules(
      {
        ...provider,
        ...updates,
        updatedAt: new Date(),
      },
      carriesScopedErrorRules(updates)
        ? readScopedErrorRulesFromCarrier(updates)
        : provider.request_scoped_errors
    );

    this.providers.set(id, updatedProvider);

    if (updates.models) {
      provider.models.forEach((model) => {
        const fullModel = `${provider.name},${model}`;
        this.modelRoutes.delete(fullModel);
        this.modelRoutes.delete(model);
      });

      updates.models.forEach((model) => {
        const fullModel = `${provider.name},${model}`;
        const route: ModelRoute = {
          provider: provider.name,
          model,
          fullModel,
        };
        this.modelRoutes.set(fullModel, route);
        if (!this.modelRoutes.has(model)) {
          this.modelRoutes.set(model, route);
        }
      });
    }

    return updatedProvider;
  }

  deleteProvider(id: string): boolean {
    const provider = this.providers.get(id);
    if (!provider) {
      return false;
    }

    provider.models.forEach((model) => {
      const fullModel = `${provider.name},${model}`;
      this.modelRoutes.delete(fullModel);
      this.modelRoutes.delete(model);
    });

    this.providers.delete(id);
    return true;
  }

  toggleProvider(name: string, _enabled: boolean): boolean {
    const provider = this.providers.get(name);
    if (!provider) {
      return false;
    }
    return true;
  }

  resolveModelRoute(modelName: string): RequestRouteInfo | null {
    const route = this.modelRoutes.get(modelName);
    if (!route) {
      return null;
    }

    const provider = this.providers.get(route.provider);
    if (!provider) {
      return null;
    }

    return {
      provider,
      originalModel: modelName,
      targetModel: route.model,
    };
  }

  getAvailableModelNames(): string[] {
    const modelNames: string[] = [];
    this.providers.forEach((provider) => {
      provider.models.forEach((model) => {
        modelNames.push(model);
        modelNames.push(`${provider.name},${model}`);
      });
    });
    return modelNames;
  }

  getModelRoutes(): ModelRoute[] {
    return Array.from(this.modelRoutes.values());
  }

  private parseTransformerConfig(transformerConfig: any): any {
    if (!transformerConfig) return {};

    if (Array.isArray(transformerConfig)) {
      return transformerConfig.reduce((acc, item) => {
        if (Array.isArray(item)) {
          const [name, config = {}] = item;
          acc[name] = config;
        } else {
          acc[item] = {};
        }
        return acc;
      }, {});
    }

    return transformerConfig;
  }

  async getAvailableModels(): Promise<{
    object: string;
    data: Array<{
      id: string;
      object: string;
      owned_by: string;
      provider: string;
    }>;
  }> {
    const models: Array<{
      id: string;
      object: string;
      owned_by: string;
      provider: string;
    }> = [];

    this.providers.forEach((provider) => {
      provider.models.forEach((model) => {
        models.push({
          id: model,
          object: "model",
          owned_by: provider.name,
          provider: provider.name,
        });

        models.push({
          id: `${provider.name},${model}`,
          object: "model",
          owned_by: provider.name,
          provider: provider.name,
        });
      });
    });

    return {
      object: "list",
      data: models,
    };
  }
}
