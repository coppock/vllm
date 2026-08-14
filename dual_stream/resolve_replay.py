# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arrival-timed replay of a bounded Resolve traffic sample against vLLM.

The input is JSONL with one object per line:

    {"timestamp": "...", "request": {<OpenAI chat request>}}

Requests with ``service_tier=flex`` are sent at priority 0 (throughput);
all other requests are sent at priority -1 (latency). Response text is never
retained or written to the result file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return round(ordered[low], 3)
    result = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return round(result, 3)


def _distribution(values: Iterable[float]) -> dict[str, float | None]:
    data = list(values)
    return {
        "p50": _percentile(data, 0.50),
        "p90": _percentile(data, 0.90),
        "p95": _percentile(data, 0.95),
        "max": round(max(data), 3) if data else None,
    }


def _meaningful_delta(chunk: dict[str, Any]) -> bool:
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if any(
            delta.get(field) for field in ("content", "reasoning_content", "tool_calls")
        ):
            return True
    return False


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("request"), dict):
                raise ValueError(f"line {line_number}: request is not an object")
            record["_timestamp"] = _parse_timestamp(record["timestamp"])
            records.append(record)
    return sorted(records, key=lambda record: record["_timestamp"])


def _summary(rows: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    ok = [
        row
        for row in rows
        if row["status"] == 200
        and row["error"] is None
        and row["input_tokens"] is not None
    ]
    output_tokens = sum(row.get("output_tokens") or 0 for row in ok)
    input_tokens = sum(row.get("input_tokens") or 0 for row in ok)
    return {
        "requests": len(rows),
        "succeeded": len(ok),
        "failed": len(rows) - len(ok),
        "elapsed_s": round(elapsed_s, 3),
        "request_rate": round(len(ok) / elapsed_s, 3) if elapsed_s else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tokens_per_s": round(output_tokens / elapsed_s, 3)
        if elapsed_s
        else None,
        "start_lag_s": _distribution(row["start_lag_s"] for row in rows),
        "ttft_s": _distribution(
            row["ttft_s"] for row in ok if row.get("ttft_s") is not None
        ),
        "e2e_s": _distribution(row["e2e_s"] for row in ok),
    }


async def _send_one(
    client: httpx.AsyncClient,
    record: dict[str, Any],
    index: int,
    due_s: float,
    run_start: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    await asyncio.sleep(max(0.0, run_start + due_s - time.monotonic()))
    started = time.monotonic()
    body = dict(record["request"])
    tier = "throughput" if body.get("service_tier") == "flex" else "latency"
    body["model"] = args.model
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    if args.route_priority:
        body["priority"] = 0 if tier == "throughput" else -1
    else:
        body.pop("priority", None)
    if args.max_output_tokens is not None:
        body.pop("max_completion_tokens", None)
        body["max_tokens"] = args.max_output_tokens

    result: dict[str, Any] = {
        "index": index,
        "timestamp": record["timestamp"],
        "tier": tier,
        "priority": body.get("priority"),
        "scheduled_s": round(due_s, 3),
        "start_lag_s": round(max(0.0, started - run_start - due_s), 3),
        "status": None,
        "ttft_s": None,
        "e2e_s": None,
        "input_tokens": None,
        "output_tokens": None,
        "finish_reason": None,
        "error": None,
    }
    first_token_at = None
    usage = None
    finish_reason = None
    try:
        async with client.stream("POST", args.url, json=body) as response:
            result["status"] = response.status_code
            if response.status_code != 200:
                await response.aread()
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    now = time.monotonic()
                    if first_token_at is None and _meaningful_delta(chunk):
                        first_token_at = now
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        finish_reason = choice.get("finish_reason") or finish_reason
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    finished = time.monotonic()
    result["e2e_s"] = round(finished - started, 3)
    result["ttft_s"] = (
        round(first_token_at - started, 3) if first_token_at is not None else None
    )
    if usage:
        result["input_tokens"] = usage.get("prompt_tokens")
        result["output_tokens"] = usage.get("completion_tokens")
    elif result["status"] == 200 and result["error"] is None:
        result["error"] = "stream ended without usage"
    result["finish_reason"] = finish_reason
    print(
        f"[{index:02d}] {tier:10s} status={result['status']} "
        f"in={result['input_tokens']} out={result['output_tokens']} "
        f"ttft={result['ttft_s']}s e2e={result['e2e_s']}s",
        flush=True,
    )
    return result


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    records = _load_records(args.input)
    if not records:
        raise ValueError("no records found")
    first_timestamp = records[0]["_timestamp"]
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    limits = httpx.Limits(
        max_connections=args.max_in_flight,
        max_keepalive_connections=args.max_in_flight,
    )
    run_start = time.monotonic()
    semaphore = asyncio.Semaphore(args.max_in_flight)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

        async def limited_send(record: dict[str, Any], index: int) -> dict[str, Any]:
            original_delay = (record["_timestamp"] - first_timestamp).total_seconds()
            due_s = original_delay / args.time_scale
            await asyncio.sleep(max(0.0, run_start + due_s - time.monotonic()))
            async with semaphore:
                return await _send_one(client, record, index, due_s, run_start, args)

        rows = await asyncio.gather(
            *(limited_send(record, index) for index, record in enumerate(records))
        )

    elapsed_s = time.monotonic() - run_start
    rows.sort(key=lambda row: row["index"])
    by_tier = {
        tier: _summary([row for row in rows if row["tier"] == tier], elapsed_s)
        for tier in ("latency", "throughput")
    }
    return {
        "configuration": {
            "url": args.url,
            "model": args.model,
            "time_scale": args.time_scale,
            "max_in_flight": args.max_in_flight,
            "max_output_tokens": args.max_output_tokens,
            "route_priority": args.route_priority,
            "source": str(args.input),
        },
        "all": _summary(rows, elapsed_s),
        "by_tier": by_tier,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-in-flight", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument(
        "--no-route-priority", action="store_false", dest="route_priority"
    )
    args = parser.parse_args()
    if args.time_scale <= 0:
        parser.error("--time-scale must be positive")
    if args.max_in_flight <= 0:
        parser.error("--max-in-flight must be positive")

    result = asyncio.run(_run(args))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all": result["all"], "by_tier": result["by_tier"]}, indent=2))


if __name__ == "__main__":
    main()
