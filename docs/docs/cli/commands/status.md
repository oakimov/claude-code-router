---
sidebar_position: 3
---

# ccr status

Show the current status of the Claude Code Router server.

## Usage

```bash
ccr status
```

## Output

### Running Server

When the server is running:

```
Claude Code Router Status: Running
Version: 2.1.10
PID: 12345
Port: 8080
Uptime: 2h 34m
Configuration: /home/user/.claude-code-router/config.json
```

### Stopped Server

When the server is not running:

```
Claude Code Router Status: Stopped
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Server is running |
| 1 | Server is stopped |
| 2 | Error checking status |

## Examples

```bash
$ ccr status

Claude Code Router Status: Running
Version: 2.1.10
PID: 12345
Port: 8080
Uptime: 2h 34m
```

## Related Commands

- [ccr start](/docs/cli/commands/start) - Start the server
- [ccr stop](/docs/cli/commands/other#ccr-stop) - Stop the server
- [ccr restart](/docs/cli/commands/other#ccr-restart) - Restart the server
