import { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import { dirname, join } from "path";
import { homedir } from "os";
import {
  existsSync,
  mkdirSync,
  writeFileSync,
  readFileSync,
  renameSync,
  unlinkSync,
} from "fs";
import { RATE_LIMIT_CONFIG } from "@caeliq/ccr-shared";

const DEFAULT_CODEX_AUTH_FILE = join(
  homedir(),
  ".claude-code-router",
  "codex_auth.json"
);
const DEFAULT_CODEX_VERIFIER_FILE = join(
  homedir(),
  ".claude-code-router",
  "codex_verifier.tmp"
);
const OAUTH_CONFIG = {
  client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
  token_endpoint: "https://auth.openai.com/oauth/token",
  scope:
    "openid profile email offline_access api.connectors.read api.connectors.invoke",
  redirect_uri: "http://localhost:1455/auth/callback",
};

interface JwtPayload {
  exp?: number;
  "https://api.openai.com/auth"?: {
    chatgpt_account_id?: string;
    chatgpt_account_is_fedramp?: boolean;
  };
}

function codexAuthFile(): string {
  return process.env.CCR_CODEX_AUTH_FILE || DEFAULT_CODEX_AUTH_FILE;
}

function codexVerifierFile(): string {
  return process.env.CCR_CODEX_VERIFIER_FILE || DEFAULT_CODEX_VERIFIER_FILE;
}

function decodeJwtPayload(token: unknown): JwtPayload | null {
  if (typeof token !== "string") return null;
  const parts = token.split(".");
  if (parts.length !== 3 || !parts[1]) return null;
  try {
    return JSON.parse(
      Buffer.from(parts[1], "base64url").toString("utf8")
    ) as JwtPayload;
  } catch {
    return null;
  }
}

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function saveTokens(data: any): { expires_at: number } {
  if (
    typeof data.access_token !== "string" ||
    typeof data.refresh_token !== "string" ||
    typeof data.id_token !== "string"
  ) {
    throw new Error("OAuth token exchange response omitted required credentials.");
  }

  const now = Date.now() / 1000;
  const idPayload = decodeJwtPayload(data.id_token);
  const accessPayload = decodeJwtPayload(data.access_token);
  const authClaims = idPayload?.["https://api.openai.com/auth"];
  const accountId = data.account_id || authClaims?.chatgpt_account_id;
  const tokens = {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    id_token: data.id_token,
    token_type: data.token_type || "Bearer",
    scope: data.scope,
    expires_at:
      data.expires_at ||
      accessPayload?.exp ||
      now + (typeof data.expires_in === "number" ? data.expires_in : 3600),
    ...(accountId ? { account_id: accountId } : {}),
    account_is_fedramp:
      authClaims?.chatgpt_account_is_fedramp === true,
    last_refresh: now,
  };

  const authFile = codexAuthFile();
  mkdirSync(dirname(authFile), { recursive: true });
  const tempFile = `${authFile}.${process.pid}.${Date.now()}.tmp`;
  try {
    writeFileSync(tempFile, JSON.stringify(tokens, null, 2), {
      mode: 0o600,
      encoding: "utf-8",
    });
    renameSync(tempFile, authFile);
  } finally {
    try {
      if (existsSync(tempFile)) unlinkSync(tempFile);
    } catch {
      // Cleanup is best-effort after atomic replacement.
    }
  }
  return tokens;
}

export async function registerCodexAuthRoutes(app: FastifyInstance): Promise<void> {
  app.get(
    "/auth/callback",
    { config: { rateLimit: { ...RATE_LIMIT_CONFIG } } },
    async (req: FastifyRequest, reply: FastifyReply) => {
    const query = req.query as any;
    const { code, state, error, error_description } = query;

    if (error) {
      try {
        const verifierFile = codexVerifierFile();
        if (existsSync(verifierFile)) unlinkSync(verifierFile);
      } catch {
        // Ignore cleanup errors.
      }
      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Failed</title></head>
          <body>
            <h1>Authentication Failed</h1>
            <p><strong>Error:</strong> ${escapeHtml(error)}</p>
            ${error_description ? `<p><strong>Description:</strong> ${escapeHtml(error_description)}</p>` : ""}
            <p>You can close this window and return to your terminal.</p>
          </body>
        </html>
      `);
      return;
    }

    if (!code || !state) {
      reply.type("text/html").send(`
        <html>
          <head><title>Invalid Callback</title></head>
          <body>
            <h1>Invalid Callback</h1>
            <p>Missing required parameters: code or state</p>
            <p>You can close this window and return to your terminal.</p>
          </body>
        </html>
      `);
      return;
    }

    // Read code_verifier from temp file
    const verifierFile = codexVerifierFile();
    if (!existsSync(verifierFile)) {
      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Failed</title></head>
          <body>
            <h1>Authentication Failed</h1>
            <p>Code verifier not found. Please run <code>ccr codex-auth</code> again and complete authentication within 5 minutes.</p>
            <p>You can close this window.</p>
          </body>
        </html>
      `);
      return;
    }

    let verifierData: {
      code_verifier: string;
      state: string;
      created_at?: number;
    };
    try {
      verifierData = JSON.parse(readFileSync(verifierFile, "utf-8"));
    } catch {
      try {
        unlinkSync(verifierFile);
      } catch {
        // Ignore cleanup errors.
      }
      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Failed</title></head>
          <body>
            <h1>Authentication Failed</h1>
            <p>Invalid code verifier data. Please run <code>ccr codex-auth</code> again.</p>
            <p>You can close this window.</p>
          </body>
        </html>
      `);
      return;
    }

    const verifierAgeMs =
      typeof verifierData.created_at === "number"
        ? Date.now() - verifierData.created_at
        : Number.POSITIVE_INFINITY;
    if (verifierAgeMs < 0 || verifierAgeMs > 5 * 60 * 1000) {
      try {
        unlinkSync(verifierFile);
      } catch {
        // Ignore cleanup errors.
      }
      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Failed</title></head>
          <body>
            <h1>Authentication Failed</h1>
            <p>The authorization request expired. Please run <code>ccr codex-auth</code> again.</p>
            <p>You can close this window.</p>
          </body>
        </html>
      `);
      return;
    }

    // Validate state
    if (verifierData.state !== state) {
      try {
        unlinkSync(verifierFile);
      } catch {
        // Ignore cleanup errors.
      }
      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Failed</title></head>
          <body>
            <h1>Authentication Failed</h1>
            <p>State mismatch. The callback may not match the current authorization request. Please run <code>ccr codex-auth</code> again.</p>
            <p>You can close this window.</p>
          </body>
        </html>
      `);
      return;
    }

    try {
      const tokenParams = new URLSearchParams({
        grant_type: "authorization_code",
        client_id: OAUTH_CONFIG.client_id,
        code,
        redirect_uri: OAUTH_CONFIG.redirect_uri,
        code_verifier: verifierData.code_verifier,
      });

      app.log.info("Starting Codex OAuth token exchange");

      const response = await fetch(OAUTH_CONFIG.token_endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: tokenParams,
        signal: AbortSignal.timeout(30_000),
      });

      const responseText = await response.text();
      app.log.info(
        { status: response.status },
        "Codex OAuth token exchange response"
      );

      if (!response.ok) {
        throw new Error(`Token exchange failed (${response.status})`);
      }

      const tokenData = JSON.parse(responseText);
      const savedTokens = saveTokens(tokenData);

      // Clean up verifier file
      try {
        unlinkSync(verifierFile);
      } catch {
        // Ignore cleanup errors
      }

      const expiresAt = new Date(savedTokens.expires_at * 1000);

      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Successful</title></head>
          <body>
            <h1>Authentication Successful</h1>
            <p>Your Codex OAuth tokens have been saved.</p>
            <p><strong>Access token expires:</strong> ${expiresAt.toLocaleString()}</p>
            <p>You can close this window and return to your terminal.</p>
          </body>
        </html>
      `);
    } catch (error: any) {
      try {
        unlinkSync(verifierFile);
      } catch {
        // Ignore cleanup errors.
      }
      app.log.error(
        { message: error?.message || "Unknown OAuth error" },
        "Codex OAuth token exchange failed"
      );
      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Failed</title></head>
          <body>
            <h1>Authentication Failed</h1>
            <p><strong>Error:</strong> ${escapeHtml(error?.message || "Unknown OAuth error")}</p>
            <p>Please try again by running <code>ccr codex-auth</code> in your terminal.</p>
            <p>You can close this window.</p>
          </body>
        </html>
      `);
    }
  });
}
