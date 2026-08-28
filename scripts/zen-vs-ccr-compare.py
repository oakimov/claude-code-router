#!/usr/bin/env python3
"""
Compare the SAME Responses payload:
  A) direct → OpenCode Zen (OpenCode CLI headers)
  B) via CCR localhost

Zen headers match packages/opencode request.ts + CCR opencode-headers transformer:
  content-type, x-api-key (no Authorization),
  x-opencode-project / session / request / client,
  User-Agent: opencode/<version>

Usage:
  export OPENCODE_API_KEY=...
  python3 scripts/zen-vs-ccr-compare.py --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

OPENCODE_VERSION = os.environ.get("OPENCODE_VERSION", "1.18.25")
OPENCODE_UA = f"opencode/{OPENCODE_VERSION}"
BASE62 = string.digits + string.ascii_uppercase + string.ascii_lowercase

PAYLOAD = {
    "stream": True,
    "store": False,
    "max_output_tokens": 256,
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Say only the word ok."}],
        }
    ],
}


def opencode_id(prefix: str) -> str:
    """Approximate OpenCode/CCR id shape: {ses|msg}_<12 hex><14 base62>."""
    ts_hex = f"{int(time.time() * 1000) & ((1 << 48) - 1):012x}"
    suffix = "".join(secrets.choice(BASE62) for _ in range(14))
    return f"{prefix}_{ts_hex}{suffix}"


def zen_headers(api_key: str, session_id: str, request_id: str) -> dict[str, str]:
    # Mirrors OpencodeHeadersTransformer.buildHeaders + request.ts opencode branch.
    # authorization is intentionally omitted (CCR sets authorization: undefined).
    return {
        "content-type": "application/json",
        "x-api-key": api_key,
        "x-opencode-project": "global",
        "x-opencode-session": session_id,
        "x-opencode-request": request_id,
        "x-opencode-client": "cli",
        "user-agent": OPENCODE_UA,
        "User-Agent": OPENCODE_UA,
    }


def ccr_headers(ccr_api_key: str) -> dict[str, str]:
    # What a client sends to CCR itself (CCR then rewrites Zen headers upstream).
    return {
        "content-type": "application/json",
        "authorization": f"Bearer {ccr_api_key}",
        "x-api-key": ccr_api_key,
        "user-agent": OPENCODE_UA,
        "User-Agent": OPENCODE_UA,
        "x-opencode-client": "cli",
        "x-opencode-project": "global",
    }


def median(vals: list[float]) -> float | None:
    clean = [v for v in vals if isinstance(v, (int, float))]
    return statistics.median(clean) if clean else None


def curl_stream(
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Use curl so TLS/UA behavior matches a real client; parse SSE for events."""
    curl = "curl"
    hdr_args: list[str] = []
    # Prefer a single User-Agent; drop duplicate case variants for curl.
    seen = set()
    for k, v in headers.items():
        lk = k.lower()
        if lk == "user-agent":
            if "user-agent" in seen:
                continue
            seen.add("user-agent")
            hdr_args.extend(["-H", f"User-Agent: {v}"])
            continue
        if lk in seen:
            continue
        seen.add(lk)
        hdr_args.extend(["-H", f"{k}: {v}"])

    cmd = [
        curl,
        "-sS",
        "-N",
        "--max-time",
        str(int(timeout)),
        "-w",
        "\n__CURL__ %{http_code} %{time_starttransfer} %{time_total} %{size_download}",
        *hdr_args,
        "--data-binary",
        "@-",
        url,
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(body),
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "msHeaders": (time.perf_counter() - t0) * 1000}

    out = proc.stdout or ""
    meta = {}
    body_text = out
    if "\n__CURL__ " in out:
        body_text, _, tail = out.rpartition("\n__CURL__ ")
        parts = tail.strip().split()
        if len(parts) >= 4:
            meta = {
                "status": int(parts[0]),
                "msHeaders": float(parts[1]) * 1000,
                "msComplete": float(parts[2]) * 1000,
                "bytes": int(float(parts[3])),
            }

    if proc.returncode != 0 and not meta:
        return {
            "ok": False,
            "error": (proc.stderr or out)[:300],
            "msHeaders": (time.perf_counter() - t0) * 1000,
        }

    status = meta.get("status", 0)
    first_event = first_event_type = None
    first_delta = first_delta_type = None
    texts: list[str] = []
    events: list[str] = []
    for line in body_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        et = ev.get("type")
        if isinstance(et, str):
            events.append(et)
            if first_event is None:
                first_event = meta.get("msHeaders")
                first_event_type = et
            if et.endswith(".delta") and first_delta is None:
                # curl can't timestamp mid-stream precisely; approximate with headers
                # if delta is in the first buffered chunk, else leave as headers+.
                first_delta = meta.get("msHeaders")
                first_delta_type = et
            if et == "response.output_text.delta":
                texts.append(ev.get("delta") or "")
            if et == "response.completed":
                for item in (ev.get("response") or {}).get("output") or []:
                    if item.get("type") != "message":
                        continue
                    for part in item.get("content") or []:
                        if part.get("type") == "output_text":
                            texts.append(part.get("text") or "")

    text = "".join(texts)
    ok = 200 <= int(status) < 300 and (
        bool(text.strip()) or "response.completed" in events or "response.incomplete" in events
    )
    return {
        "ok": ok,
        "status": status,
        "msHeaders": None if meta.get("msHeaders") is None else round(meta["msHeaders"], 1),
        "msFirstEvent": None if first_event is None else round(float(first_event), 1),
        "firstEventType": first_event_type,
        "msFirstDelta": None if first_delta is None else round(float(first_delta), 1),
        "firstDeltaType": first_delta_type,
        "msComplete": None if meta.get("msComplete") is None else round(meta["msComplete"], 1),
        "bytes": meta.get("bytes"),
        "eventCount": len(events),
        "events": events[:16],
        "text": text[:80],
        "stderr": (proc.stderr or "")[:200] if proc.returncode else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--zen-base", default="https://opencode.ai/zen/v1")
    ap.add_argument("--ccr-base", default="http://127.0.0.1:3456")
    ap.add_argument("--zen-model", default="muse-spark-1.2-contributor-free")
    ap.add_argument(
        "--ccr-model",
        default="opencode-responses,muse-spark-1.2-contributor-free",
    )
    ap.add_argument("--ccr-api-key", default=os.environ.get("CCR_API_KEY", "dummy"))
    args = ap.parse_args()

    zen_key = os.environ.get("OPENCODE_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY")
    if not zen_key:
        print("OPENCODE_API_KEY unset", file=sys.stderr)
        return 2

    # Sticky session across Zen runs (like a real OpenCode conversation).
    session_id = opencode_id("ses")

    print(
        json.dumps(
            {
                "runs": args.runs,
                "zen_model": args.zen_model,
                "ccr_model": args.ccr_model,
                "user_agent": OPENCODE_UA,
                "zen_session": session_id,
                "payload_text": PAYLOAD["input"][0]["content"][0]["text"],
            }
        ),
        flush=True,
    )

    by: dict[str, list[dict[str, Any]]] = {"zen-direct": [], "ccr": []}
    for i in range(args.runs):
        req_id = opencode_id("msg")
        zen_body = {**PAYLOAD, "model": args.zen_model}
        ccr_body = {**PAYLOAD, "model": args.ccr_model}

        z = curl_stream(
            url=f"{args.zen_base.rstrip('/')}/responses",
            headers=zen_headers(zen_key, session_id, req_id),
            body=zen_body,
            timeout=args.timeout,
        )
        z.update({"target": "zen-direct", "run": i + 1, "request_id": req_id})
        by["zen-direct"].append(z)
        print(json.dumps(z), flush=True)

        c = curl_stream(
            url=f"{args.ccr_base.rstrip('/')}/v1/responses",
            headers=ccr_headers(args.ccr_api_key),
            body=ccr_body,
            timeout=args.timeout,
        )
        c.update({"target": "ccr", "run": i + 1})
        by["ccr"].append(c)
        print(json.dumps(c), flush=True)

    print("\n=== SUMMARY (median ms, curl time_starttransfer) ===", flush=True)
    print(
        f"{'target':12} {'ok':7} {'headers':>10} {'complete':>10} {'bytes':>8}  sample text",
        flush=True,
    )
    stats: dict[str, dict[str, float | None]] = {}
    for name, rows in by.items():
        ok_rows = [r for r in rows if r.get("ok")]
        stats[name] = {
            "headers": median([r["msHeaders"] for r in ok_rows if r.get("msHeaders") is not None]),
            "complete": median([r["msComplete"] for r in ok_rows if r.get("msComplete") is not None]),
        }
        sample = next((r.get("text") for r in ok_rows if r.get("text")), "")
        b = median([float(r["bytes"]) for r in ok_rows if r.get("bytes") is not None])

        def fmt(v: float | None) -> str:
            return f"{v:10.1f}" if isinstance(v, (int, float)) else f"{'—':>10}"

        print(
            f"{name:12} {f'{len(ok_rows)}/{len(rows)}':7} "
            f"{fmt(stats[name]['headers'])} {fmt(stats[name]['complete'])} "
            f"{(f'{b:.0f}' if b is not None else '—'):>8}  {sample!r}",
            flush=True,
        )

    zd, ccr = stats.get("zen-direct", {}), stats.get("ccr", {})
    if zd.get("headers") is not None and ccr.get("headers") is not None:
        delta = float(ccr["headers"]) - float(zd["headers"])
        print(f"\nCCR − Zen (median headers/TTFB): {delta:+.1f} ms", flush=True)
        if zd.get("complete") is not None and ccr.get("complete") is not None:
            print(
                f"CCR − Zen (median complete): {float(ccr['complete']) - float(zd['complete']):+.1f} ms",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
