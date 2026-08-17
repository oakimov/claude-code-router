import { Transformer } from "@/types/transformer";
import { resolveXaiApiKey } from "@caeliq/ccr-shared";
import { getValidAccessToken, loadTokens, refreshTokens } from "../utils/xai-auth";

type ResolvedXaiAuth =
  | { mode: "pat"; token: string }
  | { mode: "oauth"; token: string };

export class XaiAuthTransformer implements Transformer {
  name = "xai-auth";
  logger?: any;
  // No endPoint property — this transformer is purely an auth shim for use
  // in provider.transformer.use[]. The openai-responses transformer
  // (endPoint = "/v1/responses") owns the wire format and registers the
  // actual route.

  private async resolveAuth(provider: any): Promise<ResolvedXaiAuth> {
    const pat = resolveXaiApiKey(provider?.apiKey, { allowBareEnvName: true });
    if (pat) return { mode: "pat", token: pat };
    const tokens = await getValidAccessToken();
    return { mode: "oauth", token: tokens.access_token };
  }

  /**
   * openai-responses only converts the body — it doesn't own the outbound
   * URL (its endPoint is a client-facing inbound route registration, not an
   * outbound path). The generic pipeline falls back to a bare
   * `provider.baseUrl` when no transformer sets config.url (routes.ts:739),
   * so this transformer must build the actual `/responses` URL itself,
   * mirroring CodexTransformer's `${baseUrl}/responses`.
   */
  private buildConfig(
    auth: ResolvedXaiAuth,
    provider: any
  ): {
    url: string;
    headers: Record<string, string>;
    __authRecovery: () => Promise<Record<string, string> | null>;
  } {
    const baseUrl = provider?.baseUrl || "https://api.x.ai/v1";
    return {
      url: `${baseUrl}/responses`,
      headers: { Authorization: `Bearer ${auth.token}` },
      __authRecovery: () => this.recoverUnauthorizedAuth(auth),
    };
  }

  async transformRequestIn(request: any, provider: any): Promise<Record<string, any>> {
    const auth = await this.resolveAuth(provider);
    return {
      body: request,
      config: this.buildConfig(auth, provider),
    };
  }

  async auth(_request: any, provider: any): Promise<any> {
    const auth = await this.resolveAuth(provider);
    return { config: this.buildConfig(auth, provider) };
  }

  /**
   * 401 recovery. PAT mode has nothing to recover — a bad literal key or
   * env value can't be fixed by CCR. OAuth mode reloads the token file
   * (another process may have already refreshed it), otherwise refreshes
   * and persists, mirroring ClaudeAuthTransformer.recoverUnauthorizedAuth.
   */
  private async recoverUnauthorizedAuth(
    previous: ResolvedXaiAuth
  ): Promise<Record<string, string> | null> {
    if (previous.mode === "pat") return null;

    const reloaded = loadTokens();
    if (reloaded?.access_token && reloaded.access_token !== previous.token) {
      return { Authorization: `Bearer ${reloaded.access_token}` };
    }

    if (!reloaded?.refresh_token) return null;

    // refreshTokens is single-flight + file-locked and persists itself.
    // Do not saveTokens here: a concurrent refresh may have already written
    // a newer rotation, and overwriting it would burn the live token.
    const refreshed = await refreshTokens(reloaded.refresh_token);
    return { Authorization: `Bearer ${refreshed.access_token}` };
  }
}
