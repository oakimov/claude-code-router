const ENV_REFERENCE_PATTERN = /^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$/;
const ENV_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

export interface ResolveXaiApiKeyOptions {
  allowBareEnvName?: boolean;
  env?: NodeJS.ProcessEnv;
}

/**
 * Resolve an xAI API key from a literal `xai-` token or an environment reference.
 *
 * `$VAR` and `${VAR}` are always treated as references. A bare `VAR` name is
 * only resolved when the caller opts in, so placeholder values such as
 * `no-key` continue to select OAuth instead of accidentally reading an env var.
 */
export function resolveXaiApiKey(
  configuredApiKey: unknown,
  options: ResolveXaiApiKeyOptions = {}
): string | undefined {
  if (typeof configuredApiKey !== "string") return undefined;

  const value = configuredApiKey.trim();
  if (!value) return undefined;
  if (value.startsWith("xai-")) return value;

  const env = options.env ?? process.env;
  const referenceMatch = value.match(ENV_REFERENCE_PATTERN);
  const envName = referenceMatch?.[1] ??
    (options.allowBareEnvName && ENV_NAME_PATTERN.test(value) ? value : undefined);
  if (!envName) return undefined;

  const resolved = env[envName]?.trim();
  return resolved?.startsWith("xai-") ? resolved : undefined;
}
