# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reproducible OpenAI-compatible stress workload for dual-stream vLLM."""

import argparse
import asyncio
import statistics
import time

import httpx

URL = "http://127.0.0.1:8000/v1/completions"


async def request(
    client: httpx.AsyncClient,
    scenario_started: float,
    role: str,
    index: int,
    max_tokens: int,
    delay: float = 0.0,
) -> dict[str, float | int | str]:
    await asyncio.sleep(delay)
    started = time.perf_counter()
    payload = {
        "model": "small",
        "prompt": f"{role} request {index}: Write a detailed response.",
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "priority": -1 if role == "latency" else 0,
    }
    response = await client.post(URL, json=payload)
    completed = time.perf_counter()
    response.raise_for_status()
    result = response.json()
    return {
        "role": role,
        "index": index,
        "latency_s": completed - started,
        "completed_at_s": completed - scenario_started,
        "tokens": result["usage"]["completion_tokens"],
    }


async def run(scenario: str) -> None:
    async with httpx.AsyncClient(timeout=180) as client:
        scenario_started = time.perf_counter()
        tasks = []
        if scenario in ("mixed", "throughput"):
            tasks.extend(
                request(client, scenario_started, "throughput", index, 160)
                for index in range(8)
            )
        if scenario in ("mixed", "latency"):
            tasks.extend(
                request(
                    client,
                    scenario_started,
                    "latency",
                    index,
                    16,
                    (1.0 + index * 0.15) if scenario == "mixed" else 0.0,
                )
                for index in range(4)
            )
        results = await asyncio.gather(*tasks)
        makespan = time.perf_counter() - scenario_started

    for result in sorted(results, key=lambda item: (item["role"], item["index"])):
        print(
            result["role"],
            result["index"],
            "latency_s",
            round(float(result["latency_s"]), 3),
            "completed_at_s",
            round(float(result["completed_at_s"]), 3),
            "tokens",
            result["tokens"],
        )
    for role in ("latency", "throughput"):
        latencies = [
            float(item["latency_s"]) for item in results if item["role"] == role
        ]
        if latencies:
            print(
                role,
                "count",
                len(latencies),
                "mean_s",
                round(statistics.mean(latencies), 3),
                "max_s",
                round(max(latencies), 3),
            )
    total_tokens = sum(int(item["tokens"]) for item in results)
    print(
        "makespan_s",
        round(makespan, 3),
        "output_tokens",
        total_tokens,
        "tokens_per_s",
        round(total_tokens / makespan, 2),
    )


parser = argparse.ArgumentParser()
parser.add_argument("scenario", choices=("mixed", "latency", "throughput"))
args = parser.parse_args()
asyncio.run(run(args.scenario))
