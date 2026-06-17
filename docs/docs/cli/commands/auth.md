---
sidebar_position: 7
---

# ccr claude-auth / ccr codex-auth / ccr qwen-auth

Authentication commands for provider backends that require OAuth or JWT tokens.

## ccr claude-auth

Authenticate with Anthropic's API using your Claude Pro or Max subscription via OAuth with PKCE.

```bash
ccr claude-auth
```

### How It Works

1. The CLI generates a PKCE challenge and prints an authorization URL from `claude.ai`
2. You open the URL in your browser and sign into your Claude account
3. Claude redirects to `http://localhost:1455/callback`, where the CCR server exchanges the authorization code for tokens
4. Tokens are saved to `~/.claude-code-router/claude_auth.json`
5. You return to the terminal and press Enter — the CLI confirms the tokens were saved
6. The `claude-auth` transformer reads the token and uses it for API requests
7. When the token nears expiry, it's refreshed automatically using the refresh token

### Prerequisites

- A [Claude Pro or Max](https://claude.ai) subscription
- The CCR server must be running (it hosts the OAuth callback on port 1455)

---

## ccr codex-auth

Authenticate with the Codex (ChatGPT) backend API via OpenAI OAuth with PKCE.

```bash
ccr codex-auth
```

### How It Works

1. The CLI generates a PKCE challenge and prints an authorization URL from `auth.openai.com`
2. It starts a callback server on `http://localhost:1455/auth/callback`
3. You open the URL in your browser and sign into your OpenAI / ChatGPT account
4. OpenAI redirects to the callback server, which exchanges the authorization code for tokens
5. Tokens are saved to `~/.claude-code-router/codex_auth.json`
6. The `codex` transformer reads the token and uses it for API requests
7. When the token nears expiry, it's refreshed automatically using the refresh token

### Prerequisites

- A [ChatGPT Plus or Pro](https://chat.openai.com) subscription

## ccr qwen-auth

Opens a browser-based auth page at `http://localhost:3456/qwen/auth` for token management.

```bash
ccr qwen-auth
```

### How It Works

1. The command tells you to open `http://localhost:3456/qwen/auth` in your browser (or you can navigate there directly)
2. On the auth page, use the **bookmarklet** (recommended) or paste a token manually:
   - **Bookmarklet**: Drag "Get Qwen Token" to your bookmarks bar, open `chat.qwen.ai`, and click it — the token is sent back automatically
   - **Manual**: Run `copy(localStorage.getItem('token'))` in `chat.qwen.ai` DevTools Console, then paste on the auth page
3. The token is validated against Qwen's API and saved to `~/.claude-code-router/qwen_auth.json`
4. Automatic token rotation — expired tokens are detected on the next request

### Prerequisites

- A Qwen Chat account and access to `qwen.aikit.club`
