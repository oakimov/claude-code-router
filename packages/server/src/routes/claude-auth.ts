import { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import { join } from "path";
import { homedir } from "os";
import { existsSync, mkdirSync, writeFileSync, readFileSync, unlinkSync } from "fs";
import { RATE_LIMIT_CONFIG } from "@caeliq/ccr-shared";

const CLAUDE_AUTH_FILE = join(homedir(), ".claude-code-router", "claude_auth.json");
const CLAUDE_VERIFIER_FILE = join(homedir(), ".claude-code-router", "claude_verifier.tmp");

const OAUTH_CONFIG = {
  client_id: "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
  token_endpoint: "https://platform.claude.com/v1/oauth/token",
  redirect_uri: "http://localhost:1455/callback",
};

function saveTokens(data: any): void {
  const dir = join(homedir(), ".claude-code-router");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

  const now = Date.now() / 1000;
  const tokens = {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    id_token: data.id_token,
    token_type: data.token_type || "Bearer",
    scope: data.scope,
    expires_at: data.expires_at || now + (data.expires_in || 3600),
    last_refresh: now,
  };

  writeFileSync(CLAUDE_AUTH_FILE, JSON.stringify(tokens, null, 2), {
    mode: 0o600,
    encoding: "utf-8",
  });
}

export async function registerClaudeAuthRoutes(app: FastifyInstance): Promise<void> {
  app.get(
    "/callback",
    { config: { rateLimit: { ...RATE_LIMIT_CONFIG } } },
    async (req: FastifyRequest, reply: FastifyReply) => {
    const query = req.query as any;
    const { code, state, error, error_description } = query;

    if (error) {
      reply.type("text/html").send(`
        <html><head><title>Authentication Failed</title></head>
        <body>
          <h1>Authentication Failed</h1>
          <p><strong>Error:</strong> ${error}</p>
          ${error_description ? `<p><strong>Description:</strong> ${error_description}</p>` : ""}
          <p>You can close this window and return to your terminal.</p>
        </body></html>
      `);
      return;
    }

    if (!code || !state) {
      reply.type("text/html").send(`
        <html><head><title>Invalid Callback</title></head>
        <body>
          <h1>Invalid Callback</h1>
          <p>Missing required parameters: code or state.</p>
          <p>You can close this window and return to your terminal.</p>
        </body></html>
      `);
      return;
    }

    if (!existsSync(CLAUDE_VERIFIER_FILE)) {
      reply.type("text/html").send(`
        <html><head><title>Authentication Failed</title></head>
        <body>
          <h1>Authentication Failed</h1>
          <p>Code verifier not found. Please run <code>ccr claude-auth</code> again.</p>
        </body></html>
      `);
      return;
    }

    let verifierData: { code_verifier: string; state: string };
    try {
      verifierData = JSON.parse(readFileSync(CLAUDE_VERIFIER_FILE, "utf-8"));
    } catch {
      reply.type("text/html").send(`
        <html><head><title>Authentication Failed</title></head>
        <body>
          <h1>Authentication Failed</h1>
          <p>Invalid verifier data. Please run <code>ccr claude-auth</code> again.</p>
        </body></html>
      `);
      return;
    }

    if (verifierData.state !== state) {
      reply.type("text/html").send(`
        <html><head><title>Authentication Failed</title></head>
        <body>
          <h1>Authentication Failed</h1>
          <p>State mismatch. Please run <code>ccr claude-auth</code> again.</p>
        </body></html>
      `);
      return;
    }

    try {
      const response = await fetch(OAUTH_CONFIG.token_endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          grant_type: "authorization_code",
          client_id: OAUTH_CONFIG.client_id,
          code,
          redirect_uri: OAUTH_CONFIG.redirect_uri,
          code_verifier: verifierData.code_verifier,
          state,
        }),
      });

      const responseText = await response.text();
      app.log.info({ status: response.status }, "Claude OAuth token exchange");

      if (!response.ok) {
        throw new Error(`Token exchange failed (${response.status}): ${responseText}`);
      }

      const tokenData = JSON.parse(responseText);
      saveTokens(tokenData);

      try { unlinkSync(CLAUDE_VERIFIER_FILE); } catch {}

      const expiresAt = new Date(
        tokenData.expires_at ? tokenData.expires_at * 1000 : Date.now() + (tokenData.expires_in || 3600) * 1000
      );

      reply.type("text/html").send(`
        <html><head><title>Authentication Successful</title></head>
        <body>
          <h1>Authentication Successful</h1>
          <p>Your Claude OAuth tokens have been saved.</p>
          <p><strong>Access token expires:</strong> ${expiresAt.toLocaleString()}</p>
          <p>You can close this window and return to your terminal.</p>
        </body></html>
      `);
    } catch (err: any) {
      app.log.error({ err }, "Claude OAuth token exchange failed");
      reply.type("text/html").send(`
        <html><head><title>Authentication Failed</title></head>
        <body>
          <h1>Authentication Failed</h1>
          <p><strong>Error:</strong> ${err.message}</p>
          <p>Please try again by running <code>ccr claude-auth</code> in your terminal.</p>
        </body></html>
      `);
    }
  });
}
