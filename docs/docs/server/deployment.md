---
title: Server Deployment
---

# Server Deployment

Claude Code Router Server supports multiple deployment methods, from local development to production environments.

## Docker Compose (Recommended)

The fastest way to get started is using Docker Compose from the repository root:

```yaml
# packages/server/docker-compose.yml
services:
  ccr:
    build:
      context: ../..
      dockerfile: packages/server/Dockerfile
    # Non-root (the image's `bun` user), immutable rootfs, no capabilities.
    # $HOME is a tmpfs because the bundled Claude Code layer writes
    # $HOME/.claude.json; the config volume is mounted beneath it.
    user: "1000:1000"
    read_only: true
    tmpfs:
      - /tmp
      - /home/bun:uid=1000,gid=1000,mode=0755
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
    ports:
      - "3456:3456"
    volumes:
      - ./ccr-config:/home/bun/.claude-code-router
    environment:
      - LOG_LEVEL=info
      - HOST=0.0.0.0
      - PORT=3456
    restart: unless-stopped
```

The image runs as the unprivileged `bun` user (uid/gid 1000). If you drop the
`user:` override, the container still starts as that user; only the `tmpfs`
and `read_only` settings need to stay in sync with it.

Start the service:

```bash
cd packages/server
docker compose up --build -d
```

Check logs:

```bash
docker compose logs -f ccr
```

## Configuration File Mounting

Mount a configuration directory into the container by creating a `docker-compose.yml`:

```yaml
services:
  ccr:
    build:
      context: ../..
      dockerfile: packages/server/Dockerfile
    ports:
      - "3456:3456"
    volumes:
      - ./config:/home/bun/.claude-code-router
    environment:
      - HOST=0.0.0.0
      - PORT=3456
```

Or mount a single config file:

```yaml
services:
  ccr:
    build:
      context: ../..
      dockerfile: packages/server/Dockerfile
    ports:
      - "3456:3456"
    volumes:
      - ./config.json:/home/bun/.claude-code-router/config.json
    environment:
      - HOST=0.0.0.0
      - PORT=3456
```

Start with:

```bash
docker compose up -d
```

Configuration file example:

```json5
{
  // Server configuration
  "HOST": "0.0.0.0",
  "PORT": 3456,
  "APIKEY": "your-api-key-here",

  // Logging configuration
  "LOG": true,
  "LOG_LEVEL": "info",

  // LLM provider configuration
  "Providers": [
    {
      "name": "openai",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "$OPENAI_API_KEY",
      "models": ["gpt-4", "gpt-3.5-turbo"]
    }
  ],

  // Routing configuration
  "Router": {
    "default": "openai,gpt-4"
  }
}
```

## Environment Variables

Override configuration through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Listen address | `127.0.0.1` |
| `PORT` | Listen port | `3456` |
| `APIKEY` | API key | - |
| `LOG_LEVEL` | Log level | `info` |
| `LOG` | Enable logging | `true` |

## Production Recommendations

### 1. Use Reverse Proxy

Use Nginx as reverse proxy:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3456;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 2. Configure HTTPS

Use Let's Encrypt to obtain free certificate:

```bash
sudo certbot --nginx -d your-domain.com
```

### 3. Log Management

Configure log rotation and persistence:

```yaml
services:
  ccr:
    build:
      context: ../..
      dockerfile: packages/server/Dockerfile
    volumes:
      - ./logs:/home/bun/.claude-code-router/logs
    environment:
      - LOG_LEVEL=warn
```

### 4. Health Check

Configure Docker health check:

```yaml
services:
  ccr:
    build:
      context: ../..
      dockerfile: packages/server/Dockerfile
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3456/api/config"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Access Web UI

After deployment is complete, access the Web UI:

```
http://localhost:3456/ui/
```

Through the Web UI you can:
- View and manage configuration
- Monitor logs
- Check service status

## Secondary Development

If you need to develop based on CCR Server, please see [API Reference](/docs/category/api).
