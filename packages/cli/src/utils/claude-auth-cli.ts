import { createHash, randomBytes } from "crypto";
import { join } from "path";
import { homedir } from "os";
import { readFileSync, writeFileSync } from "fs";

const CLAUDE_AUTH_FILE = join(homedir(), ".claude-code-router", "claude_auth.json");
const CLAUDE_VERIFIER_FILE = join(homedir(), ".claude-code-router", "claude_verifier.tmp");

const OAUTH_CONFIG = {
  client_id: "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
  authorization_endpoint: "https://claude.ai/oauth/authorize",
  redirect_uri: "http://localhost:1455/callback",
  scope: "user:profile user:inference user:sessions:claude_code user:mcp_servers",
};

function base64URLEncode(buffer: Buffer): string {
  return buffer.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");
}

function generateCodeVerifier(): string {
  return base64URLEncode(randomBytes(32));
}

function generateCodeChallenge(verifier: string): string {
  return base64URLEncode(createHash("sha256").update(verifier).digest());
}

function generateState(): string {
  return randomBytes(16).toString("hex");
}

function buildAuthorizeUrl(codeChallenge: string, state: string): string {
  const params = new URLSearchParams({
    client_id: OAUTH_CONFIG.client_id,
    redirect_uri: OAUTH_CONFIG.redirect_uri,
    response_type: "code",
    scope: OAUTH_CONFIG.scope,
    state,
    code_challenge: codeChallenge,
    code_challenge_method: "S256",
  });
  return `${OAUTH_CONFIG.authorization_endpoint}?${params.toString()}`;
}

export async function runClaudeAuth(): Promise<void> {
  const codeVerifier = generateCodeVerifier();
  const codeChallenge = generateCodeChallenge(codeVerifier);
  const state = generateState();
  const authorizeUrl = buildAuthorizeUrl(codeChallenge, state);

  console.log("Open this URL in your browser and complete sign-in:\n");
  console.log(authorizeUrl);
  console.log();

  writeFileSync(CLAUDE_VERIFIER_FILE, JSON.stringify({ code_verifier: codeVerifier, state }, null, 2), {
    mode: 0o600,
    encoding: "utf-8",
  });

  console.log("After completing sign-in, you will be redirected to the CCR server automatically.");
  console.log("Make sure the CCR server is running before proceeding.\n");
  console.log("Press Enter when you have completed authentication...");

  const readline = require("readline");
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  await new Promise<void>((resolve) => {
    rl.question("", () => {
      rl.close();
      resolve();
    });
  });

  try {
    const tokens = JSON.parse(readFileSync(CLAUDE_AUTH_FILE, "utf-8"));
    console.log("\nAuthentication successful!");
    console.log(`Access token expires: ${new Date(tokens.expires_at * 1000).toLocaleString()}`);
    console.log("\nAdd to your config:");
    console.log(`
{
  "name": "claude-subscription",
  "api_base_url": "https://api.anthropic.com",
  "api_key": "no-key",
  "models": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
  "transformer": {
    "use": ["claude-auth", "Anthropic"]
  }
}`);
  } catch {
    console.log("\nNo tokens found. Authentication may not have completed.");
    console.log("Make sure the CCR server is running and try again.");
  }
}
