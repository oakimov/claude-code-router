import { randomUUID } from "crypto";
import { Transformer } from "@/types/transformer";
import {
  antigravityEndpointCandidates,
  buildGenerateContentUrl,
  getAntigravityHeaders,
  getPreferredEndpoint,
  getValidAccessToken,
  isEndpointDisabledError,
  markEndpointUnusable,
  parseRetryDelayMs,
  rememberEndpoint,
  resolveProjectId,
  shouldWalkEndpoint,
  wrapAntigravityRequest,
} from "../utils/antigravity-auth";
import { isProviderNetworkError } from "../utils/retry";
import {
  createSSEStreamReader,
  encodeSSEData,
  encodeSSELine,
  StreamContext,
} from "../utils/stream";
import { sendUnifiedRequest } from "../utils/request";
import { createApiError } from "../api/middleware";
import { sanitizeUpstreamErrorText } from "../utils/redact";

function unwrapEnvelopePayload(raw: any): any {
  if (raw && typeof raw === "object" && "response" in raw && raw.response != null) {
    return raw.response;
  }
  return raw;
}

function resolveModel(context: any, provider: any): string {
  const bodyModel = context?.req?.body?.model;
  if (typeof bodyModel === "string" && bodyModel) return bodyModel;
  if (typeof context?.req?.model === "string" && context.req.model) {
    return context.req.model;
  }
  return provider?.models?.[0] || "";
}

function isStreamRequest(context: any): boolean {
  return context?.req?.body?.stream === true;
}

function endpointCandidates(preferredBase?: string): string[] {
  return antigravityEndpointCandidates(preferredBase);
}

export class AntigravityAuthTransformer implements Transformer {
  name = "antigravity-auth";
  ownsTransport = true;
  requestPhase = "transport" as const;
  logger?: any;

  private async buildAuthAndEnvelope(
    geminiBody: any,
    provider: any,
    context: any,
    options?: { forceRefresh?: boolean }
  ): Promise<{
    body: Record<string, any>;
    accessToken: string;
    projectId?: string;
    model: string;
    stream: boolean;
  }> {
    const tokens = await getValidAccessToken({ force: options?.forceRefresh });
    const projectId = await resolveProjectId(provider, tokens.access_token);
    const model = resolveModel(context, provider);
    const stream = isStreamRequest(context);
    const body = wrapAntigravityRequest({
      ...(projectId ? { project: projectId } : {}),
      model,
      request: geminiBody,
    });
    // Ensure requestId uniqueness even if wrap is reused
    body.requestId = body.requestId || randomUUID();
    return {
      body,
      accessToken: tokens.access_token,
      projectId,
      model,
      stream,
    };
  }

  private buildHeaders(
    accessToken: string,
    stream: boolean
  ): Record<string, string | undefined> {
    return {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
      "x-goog-api-key": undefined,
      ...getAntigravityHeaders(),
      ...(stream ? { Accept: "text/event-stream" } : {}),
    };
  }

  /**
   * Own the upstream fetch so we can refresh on 401, walk endpoint fallbacks,
   * and surface 429 RetryInfo without leaking into the generic pipeline.
   */
  private async sendWithFallback(
    body: Record<string, any>,
    accessToken: string,
    stream: boolean,
    provider: any,
    context: any,
    retriedAuth = false
  ): Promise<Response> {
    const httpsProxy = context?.req?.server?.configService?.getHttpsProxy?.();
    const logger = this.logger ?? context?.req?.log ?? context?.req?.server?.log;
    const preferredBase =
      provider?.baseUrl || provider?.api_base_url || getPreferredEndpoint();
    const endpoints = endpointCandidates(preferredBase);
    let lastError: any;

    for (let i = 0; i < endpoints.length; i++) {
      const endpoint = endpoints[i];
      const url = buildGenerateContentUrl(endpoint, stream);
      const headers = this.buildHeaders(accessToken, stream);

      try {
        const response = await sendUnifiedRequest(
          url,
          body as any,
          {
            httpsProxy,
            headers: Object.fromEntries(
              Object.entries(headers).filter(([, v]) => v !== undefined)
            ) as Record<string, string>,
            signal: context?.signal,
          },
          context,
          logger
        );

        if (response.status === 401 && !retriedAuth) {
          const refreshed = await getValidAccessToken({ force: true });
          return this.sendWithFallback(
            body,
            refreshed.access_token,
            stream,
            provider,
            context,
            true
          );
        }

        if (response.status === 429) {
          const errorText = await response.clone().text().catch(() => "");
          const retryMs = parseRetryDelayMs(errorText);
          const headersOut: Record<string, string> = {};
          if (retryMs != null) {
            headersOut["Retry-After"] = String(Math.ceil(retryMs / 1000));
          }
          throw createApiError(
            sanitizeUpstreamErrorText(errorText) ||
              "Antigravity rate limit exceeded",
            429,
            "rate_limit_error",
            "api_error",
            headersOut
          );
        }

        if (!response.ok) {
          const errorText = await response.clone().text().catch(() => "");
          // A sandbox host whose API is not enabled on this project can never
          // serve us — stop probing it for the rest of the session.
          if (response.status === 403 && isEndpointDisabledError(errorText)) {
            markEndpointUnusable(endpoint);
            logger?.warn?.(
              { endpoint },
              "Antigravity endpoint is not enabled for this project; skipping it"
            );
          }

          if (shouldWalkEndpoint(response.status, i < endpoints.length - 1)) {
            lastError = createApiError(
              `Antigravity endpoint ${endpoint} returned ${response.status}`,
              response.status,
              "provider_error"
            );
            continue;
          }

          // Out of endpoints: report the upstream failure as itself. Returning
          // the body here would leave the response converter to fail on missing
          // choices and surface an opaque 500 with Google's error text inside.
          throw createApiError(
            sanitizeUpstreamErrorText(errorText) ||
              `Antigravity request failed with ${response.status}`,
            response.status,
            response.status === 403 ? "permission_error" : "provider_error"
          );
        }

        rememberEndpoint(endpoint);
        return response;
      } catch (error: any) {
        if (error?.statusCode === 429 || error?.status === 429) throw error;
        // Only transport failures justify trying another host. A client that
        // hung up must not have its prompt re-sent, and the errors raised above
        // are already final.
        if (!isProviderNetworkError(error)) throw error;
        lastError = error;
        if (i < endpoints.length - 1) continue;
        throw error;
      }
    }

    throw (
      lastError ||
      createApiError(
        "All Antigravity endpoints failed",
        502,
        "provider_error"
      )
    );
  }

  async transformRequestIn(
    request: any,
    provider: any,
    context: any
  ): Promise<Record<string, any>> {
    // request is already the Gemini body from GeminiTransformer
    const prepared = await this.buildAuthAndEnvelope(request, provider, context);
    const response = await this.sendWithFallback(
      prepared.body,
      prepared.accessToken,
      prepared.stream,
      provider,
      context
    );

    return {
      body: prepared.body,
      config: {
        url: buildGenerateContentUrl(
          getPreferredEndpoint(),
          prepared.stream
        ),
        headers: this.buildHeaders(prepared.accessToken, prepared.stream),
        __providerResponse: response,
      },
    };
  }

  async auth(_request: any, provider: any, context: any): Promise<any> {
    const prepared = await this.buildAuthAndEnvelope(
      _request,
      provider,
      context
    );
    return {
      config: {
        url: buildGenerateContentUrl(
          provider?.baseUrl || getPreferredEndpoint(),
          prepared.stream
        ),
        headers: this.buildHeaders(prepared.accessToken, prepared.stream),
      },
    };
  }

  async transformResponseOut(response: Response): Promise<Response> {
    const contentType = response.headers.get("Content-Type") || "";

    if (!contentType || contentType.includes("text/event-stream")) {
      if (!response.body) return response;
      return createSSEStreamReader(
        response,
        (line: string, ctx: StreamContext) => {
          if (!line.trim()) {
            ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
            return;
          }
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6);
            if (dataStr.trim() === "[DONE]") {
              // Must use encodeSSEData so events are delimited with a blank
              // line — encodeSSELine emits only one \n, and the next-stage
              // SSEParser overwrites currentEvent.data when two data: lines
              // arrive without a blank separator (drops earlier content).
              ctx.controller.enqueue(encodeSSEData("[DONE]", ctx.encoder));
              return;
            }
            try {
              const parsed = JSON.parse(dataStr);
              const unwrapped = unwrapEnvelopePayload(parsed);
              ctx.controller.enqueue(
                encodeSSEData(JSON.stringify(unwrapped), ctx.encoder)
              );
            } catch {
              ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
            }
            return;
          }
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
        },
        { logger: this.logger }
      );
    }

    if (contentType.includes("application/json")) {
      const text = await response.text();
      try {
        const parsed = JSON.parse(text);
        const unwrapped = unwrapEnvelopePayload(parsed);
        return new Response(JSON.stringify(unwrapped), {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      } catch {
        return new Response(text, {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      }
    }

    return response;
  }
}
