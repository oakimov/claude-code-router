import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";

const QWEN_AUTH_FILE = join(homedir(), ".claude-code-router", "qwen_auth.json");

export async function runQwenAuth(): Promise<void> {
  // The CCR server (whether running locally or in Docker) hosts the auth page
  // at /qwen/auth. We default to the standard CCR port (3456), which is
  // what Docker's port mapping forwards to. Users with a non-default port
  // can edit the URL printed below.
  const port = 3456;
  const authUrl = `http://127.0.0.1:${port}/qwen/auth`;

  console.log("Open this URL in your browser and paste your Qwen token:\n");
  console.log(`  ${authUrl}`);
  console.log();
  console.log("To get a token:");
  console.log("  1. Open https://chat.qwen.ai in another tab and sign in.");
  console.log("  2. Open the browser dev tools (F12) → Console.");
  console.log("  3. Run:  copy(localStorage.getItem('token'))");
  console.log("  4. Paste the token into the form on the auth page.");
  console.log();
  console.log("Make sure the CCR server is running (locally: `ccr start`;");
  console.log("in Docker: the container is already up and serving the page).");
  console.log();
  console.log("After saving the token in the browser, press Enter here to verify...");

  const readline = await import("readline");
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  await new Promise<void>((resolve) => {
    rl.question("", () => {
      rl.close();
      resolve();
    });
  });

  try {
    if (!existsSync(QWEN_AUTH_FILE)) {
      throw new Error("file not found");
    }
    const tokens = JSON.parse(readFileSync(QWEN_AUTH_FILE, "utf-8"));
    if (!tokens.token) {
      throw new Error("token missing");
    }
    const expStr = tokens.expiresAt
      ? new Date(tokens.expiresAt).toLocaleString()
      : "unknown";
    console.log("\nAuthentication successful!");
    console.log(`Token expires: ${expStr}`);
    console.log(`Stored at: ${QWEN_AUTH_FILE}`);
  } catch {
    console.log("\nNo token found. Authentication may have failed or not completed.");
    console.log("Please try again and ensure you submitted the form on the auth page.");
  }
}
