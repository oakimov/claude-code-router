import { createHash, randomBytes } from "crypto";
import { join } from "path";
import { homedir } from "os";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "fs";
import { createInterface } from "readline";
import {
  ANTIGRAVITY_CLIENT_ID,
  ANTIGRAVITY_CLIENT_SECRET,
  resolveCliAntigravityProjectId,
  saveAntigravityTokens,
  type AntigravityTokens,
} from "./antigravity-auth";

const AUTH_FILE = join(homedir(), ".claude-code-router", "antigravity_auth.json");
const VERIFIER_FILE = join(
  homedir(),
  ".claude-code-router",
  "antigravity_verifier.tmp"
);

const ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauth-callback";
const ANTIGRAVITY_SCOPES = [
  "https://www.googleapis.com/auth/cloud-platform",
  "https://www.googleapis.com/auth/userinfo.email",
  "https://www.googleapis.com/auth/userinfo.profile",
  "https://www.googleapis.com/auth/cclog",
  "https://www.googleapis.com/auth/experimentsandconfigs",
];

function base64URLEncode(buffer: Buffer): string {
  return buffer
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=/g, "");
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
    client_id: ANTIGRAVITY_CLIENT_ID,
    redirect_uri: ANTIGRAVITY_REDIRECT_URI,
    response_type: "code",
    scope: ANTIGRAVITY_SCOPES.join(" "),
    state,
    code_challenge: codeChallenge,
    code_challenge_method: "S256",
    access_type: "offline",
    prompt: "consent",
  });
  return `https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`;
}

function parseCliArgs(argv: string[]): {
  manual: boolean;
  project?: string;
} {
  let manual = false;
  let project: string | undefined;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--manual") manual = true;
    else if (argv[i] === "--project" && argv[i + 1]) {
      project = argv[++i];
    }
  }
  return { manual, project };
}

async function promptForCallbackUrl(): Promise<string> {
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  try {
    return await new Promise<string>((resolve) => {
      rl.question(
        "Paste the full redirect URL from the browser:\n> ",
        (answer) => resolve(answer)
      );
    });
  } finally {
    rl.close();
  }
}

async function exchangeAuthorizationCode(
  code: string,
  codeVerifier: string
): Promise<AntigravityTokens> {
  const body = new URLSearchParams({
    client_id: ANTIGRAVITY_CLIENT_ID,
    client_secret: ANTIGRAVITY_CLIENT_SECRET,
    code,
    grant_type: "authorization_code",
    redirect_uri: ANTIGRAVITY_REDIRECT_URI,
    code_verifier: codeVerifier,
  });
  const res = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `Antigravity token exchange failed (${res.status}): ${text.slice(0, 300)}`
    );
  }
  const data = (await res.json()) as {
    access_token?: string;
    refresh_token?: string;
    expires_in?: number;
  };
  if (!data.access_token || !data.refresh_token) {
    throw new Error(
      "Token exchange did not return access_token and refresh_token."
    );
  }
  return {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    expires_at: Date.now() + (data.expires_in || 3600) * 1000,
  };
}

async function fetchUserEmail(accessToken: string): Promise<string | undefined> {
  try {
    const res = await fetch(
      "https://www.googleapis.com/oauth2/v2/userinfo",
      {
        headers: { Authorization: `Bearer ${accessToken}` },
        signal: AbortSignal.timeout(10_000),
      }
    );
    if (!res.ok) return undefined;
    const data = (await res.json()) as { email?: string };
    return data.email;
  } catch {
    return undefined;
  }
}

/**
 * Default: CLI writes PKCE verifier; CCR server handles GET /oauth-callback
 * (Docker maps 51121→3456). --manual exchanges in-process without the server.
 */
export async function runAntigravityAuth(
  argv: string[] = process.argv.slice(3)
): Promise<void> {
  const { manual, project } = parseCliArgs(argv);

  console.log(
    "\n⚠️  Terms notice: Antigravity OAuth uses the Google Antigravity IDE client\n" +
      "   credentials. Using them from a non-IDE client may violate Google's terms\n" +
      "   and can risk rate-limiting or account suspension. Proceed at your own risk.\n"
  );

  const codeVerifier = generateCodeVerifier();
  const codeChallenge = generateCodeChallenge(codeVerifier);
  const state = generateState();
  const authorizeUrl = buildAuthorizeUrl(codeChallenge, state);

  console.log("Open this URL in your browser and complete sign-in:\n");
  console.log(authorizeUrl);
  console.log();

  if (manual) {
    console.log(
      "Manual mode: after Google redirects, copy the full localhost URL and paste it below."
    );
    const callbackRaw = await promptForCallbackUrl();
    const url = new URL(callbackRaw.trim());
    const code = url.searchParams.get("code");
    const returnedState = url.searchParams.get("state");
    if (!code || !returnedState) {
      throw new Error("Callback URL missing code or state");
    }
    if (returnedState !== state) {
      throw new Error("OAuth state mismatch — aborting.");
    }

    console.log("Exchanging authorization code…");
    let tokens = await exchangeAuthorizationCode(code, codeVerifier);
    const email = await fetchUserEmail(tokens.access_token);
    if (email) tokens = { ...tokens, email };
    if (project?.trim()) {
      tokens = { ...tokens, project_id: project.trim() };
    } else {
      saveAntigravityTokens(tokens);
      const resolved = await resolveCliAntigravityProjectId(
        undefined,
        tokens.access_token
      );
      if (resolved) tokens = { ...tokens, project_id: resolved };
    }
    saveAntigravityTokens(tokens);
    printSuccess(tokens);
    return;
  }

  const dir = join(homedir(), ".claude-code-router");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

  const verifierData = {
    code_verifier: codeVerifier,
    state,
    created_at: Date.now(),
    ...(project?.trim() ? { project_id: project.trim() } : {}),
  };
  console.log("Saving code_verifier to:", VERIFIER_FILE);
  writeFileSync(VERIFIER_FILE, JSON.stringify(verifierData, null, 2), {
    mode: 0o600,
    encoding: "utf-8",
  });

  console.log(
    "After sign-in, Google redirects to http://localhost:51121/oauth-callback"
  );
  console.log(
    "(Docker: compose maps 51121→3456 onto the CCR server — keep the server running.)"
  );
  console.log("The tokens will be saved automatically.");
  console.log();
  console.log("Press Enter when you have completed authentication...");

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  await new Promise<void>((resolve) => {
    rl.question("", () => {
      rl.close();
      resolve();
    });
  });

  try {
    const tokens = JSON.parse(readFileSync(AUTH_FILE, "utf-8"));
    printSuccess(tokens);
  } catch {
    console.log(
      "\nNo tokens found. Authentication may have failed or not completed."
    );
    console.log(
      "Ensure the CCR server is running (and port 51121 is published if using Docker),"
    );
    console.log("or retry with: ccr antigravity-auth --manual");
  }
}

function printSuccess(tokens: {
  email?: string;
  project_id?: string;
  expires_at?: number;
}): void {
  console.log("\nAuthentication successful!");
  if (tokens.email) console.log(`Account: ${tokens.email}`);
  if (tokens.project_id) console.log(`Project: ${tokens.project_id}`);
  if (tokens.expires_at) {
    console.log(
      `Access token expires: ${new Date(tokens.expires_at).toLocaleString()}`
    );
  }
  console.log(
    "\nAdd an antigravity provider to config.json, then `ccr restart`."
  );
  console.log("List models with: ccr model get antigravity");
}
