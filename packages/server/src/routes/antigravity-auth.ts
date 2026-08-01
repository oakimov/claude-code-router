import { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import { join } from "path";
import { homedir } from "os";
import { existsSync, readFileSync, unlinkSync } from "fs";
import { RATE_LIMIT_CONFIG } from "@caeliq/ccr-shared";
import {
  exchangeAuthorizationCode,
  fetchUserEmail,
  resolveProjectId,
  saveTokens,
} from "@caeliq/llms";

const DEFAULT_VERIFIER_FILE = join(
  homedir(),
  ".claude-code-router",
  "antigravity_verifier.tmp"
);

function verifierFile(): string {
  return process.env.CCR_ANTIGRAVITY_VERIFIER_FILE || DEFAULT_VERIFIER_FILE;
}

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export async function registerAntigravityAuthRoutes(
  app: FastifyInstance
): Promise<void> {
  // Public OAuth callback. Host port 51121 is published to this server (3456).
  app.get(
    "/oauth-callback",
    { config: { rateLimit: { ...RATE_LIMIT_CONFIG } } },
    async (req: FastifyRequest, reply: FastifyReply) => {
    const query = req.query as any;
    const { code, state, error, error_description } = query;

    if (error) {
      try {
        if (existsSync(verifierFile())) unlinkSync(verifierFile());
      } catch {
        // ignore
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

    const verifierPath = verifierFile();
    if (!existsSync(verifierPath)) {
      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Failed</title></head>
          <body>
            <h1>Authentication Failed</h1>
            <p>Code verifier not found. Please run <code>ccr antigravity-auth</code> again and complete authentication within 5 minutes.</p>
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
      project_id?: string;
    };
    try {
      verifierData = JSON.parse(readFileSync(verifierPath, "utf-8"));
    } catch {
      try {
        unlinkSync(verifierPath);
      } catch {
        // ignore
      }
      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Failed</title></head>
          <body>
            <h1>Authentication Failed</h1>
            <p>Invalid code verifier data. Please run <code>ccr antigravity-auth</code> again.</p>
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
        unlinkSync(verifierPath);
      } catch {
        // ignore
      }
      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Failed</title></head>
          <body>
            <h1>Authentication Failed</h1>
            <p>The authorization request expired. Please run <code>ccr antigravity-auth</code> again.</p>
            <p>You can close this window.</p>
          </body>
        </html>
      `);
      return;
    }

    if (verifierData.state !== state) {
      try {
        unlinkSync(verifierPath);
      } catch {
        // ignore
      }
      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Failed</title></head>
          <body>
            <h1>Authentication Failed</h1>
            <p>State mismatch. Please run <code>ccr antigravity-auth</code> again.</p>
            <p>You can close this window.</p>
          </body>
        </html>
      `);
      return;
    }

    try {
      app.log.info("Starting Antigravity OAuth token exchange");

      const tokens = await exchangeAuthorizationCode(
        code,
        verifierData.code_verifier
      );

      app.log.info("Antigravity OAuth token exchange succeeded");

      const email = await fetchUserEmail(tokens.access_token);

      // Persist tokens before project discovery so resolveProjectId does not
      // reuse a stale project_id from a previous account's auth file.
      saveTokens({
        ...tokens,
        ...(email ? { email } : {}),
        ...(verifierData.project_id?.trim()
          ? { project_id: verifierData.project_id.trim() }
          : {}),
      });

      const projectId = await resolveProjectId(
        verifierData.project_id?.trim()
          ? { project_id: verifierData.project_id.trim() }
          : undefined,
        tokens.access_token
      );

      try {
        unlinkSync(verifierPath);
      } catch {
        // ignore
      }

      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Successful</title></head>
          <body>
            <h1>Antigravity Authentication Successful</h1>
            ${email ? `<p>Account: <strong>${escapeHtml(email)}</strong></p>` : ""}
            ${projectId ? `<p>Project: <strong>${escapeHtml(projectId)}</strong></p>` : "<p>No project id resolved — requests will proceed without one (optional).</p>"}
            <p>You can close this window and return to your terminal.</p>
          </body>
        </html>
      `);
    } catch (err: any) {
      app.log.error({ err }, "Antigravity OAuth callback failed");
      try {
        unlinkSync(verifierPath);
      } catch {
        // ignore
      }
      reply.type("text/html").send(`
        <html>
          <head><title>Authentication Failed</title></head>
          <body>
            <h1>Authentication Failed</h1>
            <p>${escapeHtml(err?.message || String(err))}</p>
            <p>Please try again by running <code>ccr antigravity-auth</code> in your terminal.</p>
          </body>
        </html>
      `);
    }
  });
}
