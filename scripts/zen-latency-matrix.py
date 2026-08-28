#!/usr/bin/env python3
"""
Compare OpenCode Zen turnaround across runtimes, network paths, and HTTP versions.

Matrix:
  host + bun   ×  default | http1.1 | http2 | http3
  host + node  ×  default | http1.1 | http2          (http3 unsupported)
  docker-node  ×  default | http1.1 | http2          (http3 unsupported)

Optional curl check (reports negotiated %{http_version}):
  --curl-check

Usage:
  export OPENCODE_API_KEY=...
  python3 scripts/zen-latency-matrix.py --mode models --runs 3 --protocols all
  python3 scripts/zen-latency-matrix.py --mode responses --runs 2 --protocols http1.1,http2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROBE = ROOT / "zen-probe.mjs"

ALL_PROTOCOLS = ("default", "http1.1", "http2", "http3")


def run(cmd: list[str], env: dict[str, str], timeout: float) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"timeout after {timeout}s",
            "cmd": cmd[:6],
            "stdout": (exc.stdout or "")[-500:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-500:] if isinstance(exc.stderr, str) else "",
        }

    out = (proc.stdout or "").strip().splitlines()
    payload: dict[str, Any] | None = None
    for line in reversed(out):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    if payload is None:
        return {
            "ok": False,
            "error": "no JSON result",
            "exit": proc.returncode,
            "stdout": (proc.stdout or "")[-800:],
            "stderr": (proc.stderr or "")[-800:],
            "cmd": cmd[:6],
        }
    payload["exit"] = proc.returncode
    if proc.stderr and not payload.get("ok"):
        payload["stderr_tail"] = proc.stderr[-400:]
    return payload


def median(vals: list[float]) -> float | None:
    clean = [v for v in vals if isinstance(v, (int, float))]
    if not clean:
        return None
    return statistics.median(clean)


def curl_check(base: str, protocols: list[str]) -> list[dict[str, Any]]:
    curl = shutil.which("curl")
    if not curl:
        return [{"ok": False, "error": "curl not found"}]
    url = f"{base.rstrip('/')}/models"
    rows = []
    for proto in protocols:
        if proto in ("default",):
            continue
        args = [curl, "-sS", "-o", "/dev/null", "-w", "%{http_version} %{time_starttransfer} %{http_code}"]
        if proto == "http1.1":
            args.append("--http1.1")
        elif proto == "http2":
            args.append("--http2")
        elif proto == "http3":
            args.extend(["--http3", "--http3-only"])
        else:
            continue
        args.append(url)
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            rows.append({"ok": False, "tool": "curl", "protocol": proto, "error": "timeout"})
            continue
        if proc.returncode != 0:
            rows.append(
                {
                    "ok": False,
                    "tool": "curl",
                    "protocol": proto,
                    "error": (proc.stderr or proc.stdout or f"exit {proc.returncode}")[:200],
                }
            )
            continue
        parts = (proc.stdout or "").strip().split()
        rows.append(
            {
                "ok": True,
                "tool": "curl",
                "protocolRequested": proto,
                "httpVersion": parts[0] if parts else None,
                "msHeaders": float(parts[1]) * 1000 if len(parts) > 1 else None,
                "status": int(parts[2]) if len(parts) > 2 else None,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("responses", "models"), default="models")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default=os.environ.get("ZEN_MODEL", "muse-spark-1.2-contributor-free"))
    ap.add_argument("--base", default=os.environ.get("ZEN_BASE", "https://opencode.ai/zen/v1"))
    ap.add_argument("--container", default=os.environ.get("CCR_CONTAINER", "claude-code-router"))
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--protocols",
        default="all",
        help="comma list or 'all' (default,http1.1,http2,http3)",
    )
    ap.add_argument("--skip-docker", action="store_true")
    ap.add_argument("--skip-bun", action="store_true")
    ap.add_argument("--skip-node", action="store_true")
    ap.add_argument("--curl-check", action="store_true", help="also probe with curl --http1.1/--http2/--http3")
    args = ap.parse_args()

    key = os.environ.get("OPENCODE_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY")
    if not key:
        print("OPENCODE_API_KEY is unset", file=sys.stderr)
        return 2
    if not PROBE.is_file():
        print(f"missing probe: {PROBE}", file=sys.stderr)
        return 2

    if args.protocols.strip().lower() == "all":
        protocols = list(ALL_PROTOCOLS)
    else:
        protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]

    base_env = {
        **os.environ,
        "OPENCODE_API_KEY": key,
        "ZEN_BASE": args.base,
        "ZEN_MODEL": args.model,
        "ZEN_MODE": args.mode,
    }

    # (target_name, cmd, env, supports_http3)
    engines: list[tuple[str, list[str], bool]] = []
    bun = shutil.which("bun")
    node = shutil.which("node")
    if not args.skip_bun and bun:
        engines.append(("host-bun", [bun, str(PROBE)], True))
    elif not args.skip_bun:
        print("warn: bun not found; skipping host-bun", file=sys.stderr)
    if not args.skip_node and node:
        engines.append(("host-node", [node, str(PROBE)], False))
    elif not args.skip_node:
        print("warn: node not found; skipping host-node", file=sys.stderr)

    docker = shutil.which("docker")
    if not args.skip_docker and docker:
        docker_probe = "/tmp/zen-probe.mjs"
        cp = subprocess.run(
            [docker, "cp", str(PROBE), f"{args.container}:{docker_probe}"],
            capture_output=True,
            text=True,
        )
        if cp.returncode != 0:
            print(
                f"warn: docker cp failed ({cp.stderr.strip() or cp.stdout.strip()}); skipping docker-node",
                file=sys.stderr,
            )
        else:
            engines.append(
                (
                    "docker-node",
                    [
                        docker,
                        "exec",
                        "-e",
                        f"OPENCODE_API_KEY={key}",
                        "-e",
                        f"ZEN_BASE={args.base}",
                        "-e",
                        f"ZEN_MODEL={args.model}",
                        "-e",
                        f"ZEN_MODE={args.mode}",
                        args.container,
                        "node",
                        docker_probe,
                    ],
                    False,
                )
            )
    elif not args.skip_docker:
        print("warn: docker not found; skipping docker-node", file=sys.stderr)

    targets: list[tuple[str, list[str], dict[str, str]]] = []
    for engine, cmd, supports_h3 in engines:
        for proto in protocols:
            if proto == "http3" and not supports_h3:
                targets.append(
                    (
                        f"{engine}/{proto}",
                        ["python3", "-c", "print('skip')"],
                        {"__SKIP__": "node-no-http3", "ZEN_RUNTIME_LABEL": f"{engine}/{proto}"},
                    )
                )
                continue
            label = f"{engine}/{proto}"
            if engine == "docker-node":
                # inject protocol + label into docker exec env flags before container name
                # cmd layout: docker exec -e KEY=... -e ... container node probe
                new_cmd = list(cmd)
                # insert before container name (index of args.container)
                idx = new_cmd.index(args.container)
                new_cmd[idx:idx] = [
                    "-e",
                    f"ZEN_HTTP_PROTOCOL={proto}",
                    "-e",
                    f"ZEN_RUNTIME_LABEL={label}",
                ]
                targets.append((label, new_cmd, {**base_env, "ZEN_HTTP_PROTOCOL": proto, "ZEN_RUNTIME_LABEL": label}))
            else:
                targets.append(
                    (
                        label,
                        cmd,
                        {
                            **base_env,
                            "ZEN_HTTP_PROTOCOL": proto,
                            "ZEN_RUNTIME_LABEL": label,
                        },
                    )
                )

    if args.curl_check:
        for row in curl_check(args.base, protocols):
            print(json.dumps(row), flush=True)

    print(
        json.dumps(
            {
                "matrix": [t[0] for t in targets],
                "mode": args.mode,
                "model": args.model if args.mode == "responses" else None,
                "base": args.base,
                "runs": args.runs,
                "protocols": protocols,
            }
        ),
        flush=True,
    )

    by_target: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in targets}

    for name, cmd, env in targets:
        if env.get("__SKIP__"):
            skip = {
                "ok": False,
                "skipped": True,
                "target": name,
                "protocolRequested": name.split("/")[-1],
                "error": env["__SKIP__"],
            }
            by_target[name].append(skip)
            print(json.dumps(skip), flush=True)
            continue
        for i in range(args.runs):
            result = run(cmd, env, args.timeout)
            result["target"] = name
            result["run"] = i + 1
            by_target[name].append(result)
            print(json.dumps(result), flush=True)

    print("\n=== SUMMARY (median ms) ===", flush=True)
    print(
        f"{'target':28} {'ok':7} {'proto':8} {'ver':8} {'headers':>10} {'1stEvent':>10} {'complete':>10}",
        flush=True,
    )
    for name, rows in by_target.items():
        ok_rows = [r for r in rows if r.get("ok")]
        skip_rows = [r for r in rows if r.get("skipped")]
        if skip_rows and not ok_rows:
            print(f"{name:28} skip    {name.split('/')[-1]:8} {'—':8} {'—':>10} {'—':>10} {'—':>10}", flush=True)
            continue

        def col(key: str) -> float | None:
            return median([r[key] for r in ok_rows if key in r])

        vers = sorted({str(r.get("httpVersion")) for r in ok_rows if r.get("httpVersion")})
        ver = ",".join(vers) if vers else "—"
        proto = name.split("/")[-1]

        def fmt(v: float | None) -> str:
            return f"{v:10.1f}" if isinstance(v, (int, float)) else f"{'—':>10}"

        print(
            f"{name:28} {f'{len(ok_rows)}/{len(rows)}':7} {proto:8} {ver:8} "
            f"{fmt(col('msHeaders'))} {fmt(col('msFirstEvent'))} "
            f"{fmt(col('msComplete') or col('msBody'))}",
            flush=True,
        )

    print(
        "\nNotes:\n"
        "- Bun pins via fetch({ protocol: 'http1.1'|'http2'|'http3' })\n"
        "- Node pins via https ALPN http/1.1 or http2.connect (no stock HTTP/3)\n"
        "- Zen (opencode.ai) has been observed to accept h1/h2; h3 handshake may fail\n"
        "- docker-node uses the CCR container's Node to isolate NAT vs host",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
