#!/usr/bin/env python3
"""Harness de generación sintética determinista de errores para Agent-R.

Genera N errores con seed fija (SplitMix64 sobre el índice) y los envía por
POST /agent-r/record-error con concurrencia controlada (asyncio, semáforo).
Determinista: mismo seed => mismo stream de payloads (salvo timestamps del server).

Uso:
  gen_errors.py --url http://127.0.0.1:19003 --n 10000 --seed 20261012 \
      --concurrency 10 [--out-dir DIR] [--validate-only]

Salida: JSON en stdout {n, seed, elapsed_s, throughput_rps, codes:{200:n,...}}
y archivo <out-dir>/manifest-N-seed-S.json con el manifiesto reproducible.
"""
import argparse
import asyncio
import json
import time
import urllib.request
import urllib.error

CATEGORIES = [
    "fabrication", "circular_reasoning", "output_drift", "bad_assumption",
    "skipped_validation", "timeout", "state_corruption", "recurring_pattern", "other",
]
SEVERITIES = ["warning", "error", "critical"]
OPERATIONS = [
    "reindex_collection", "store_memory", "search_similar", "rebuild_hnsw_index",
    "merge_splats", "validate_output_chain", "promote_rule", "distill_lessons",
    "snapshot_metadata", "gc_expired_splats", "preflight_mutation", "audit_receipts",
    "consolidate_memory", "expand_context", "route_query",
]
TAG_POOL = [
    "gpu", "cpu", "nightly", "ci", "prod", "index", "benchmark", "large-batch",
    "cold-start", "retry", "timeout-window", "sqlite", "hot-loop", "high-recall",
    "precision@10", "hnsw", "warming", "crash-recovery", "sha256", "receipt",
]


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def gen_record(i: int, seed: int) -> dict:
    """Registro i-ésimo 100% determinista a partir de (i, seed)."""
    r = splitmix64(seed ^ (i * 0x8000000000000001 if i else 1))
    r2 = splitmix64(r)
    r3 = splitmix64(r2)
    r4 = splitmix64(r3)
    cat = CATEGORIES[r % len(CATEGORIES)]
    sev = SEVERITIES[(r2 >> 17) % len(SEVERITIES)]
    op = OPERATIONS[(r3 >> 31) % len(OPERATIONS)]
    n_tags = 1 + ((r4 >> 47) % 4)  # 1..4 tags
    tags = [TAG_POOL[(r >> (8 * (k + 1))) % len(TAG_POOL)] for k in range(n_tags)]
    # Longitudes de texto controladas y reproducibles (sin timestamps/aleatoriedad real)
    what = f"synthetic failure mode #{i}: {op} produced inconsistent state hash {r & 0xFFFFFF:06x}"
    why = f"root cause {r2 & 0xFFFF:04x}: precondition violated before {op} (deterministic harness seed {seed})"
    prevention = f"verify preconditions of {op} and re-run with receipt {r3 & 0xFFFF:04x} before commit"
    return {
        "category": cat,
        "severity": sev,
        "operation": op,
        "what": what,
        "why": why,
        "prevention": prevention,
        "context_tags": tags,
    }


async def post(session_sem, url: str, payload: dict) -> tuple:
    loop = asyncio.get_event_loop()

    def do():
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:  # noqa: BLE001
            return -1, str(e).encode()

    async with session_sem:
        return await loop.run_in_executor(None, do)


async def post_chunk(sem, url: str, chunk: list) -> list:
    """Envía un chunk de payloads; devuelve (status, body) por payload."""
    loop = asyncio.get_event_loop()
    tasks = [post(sem, url, p) for p in chunk]
    return await asyncio.gather(*tasks)


async def run(args):
    sem = asyncio.Semaphore(args.concurrency)
    url = args.url.rstrip("/") + "/agent-r/record-error"
    codes = {}
    n_ok = 0
    t0 = time.perf_counter()
    first_record = None
    last_record = None
    CHUNK = 10_000  # materializa 10k payloads por vez (RAM del generador acotada)
    done = 0
    while done < args.n:
        m = min(CHUNK, args.n - done)
        chunk = [gen_record(done + k, args.seed) for k in range(m)]
        if first_record is None:
            first_record = chunk[0]
        last_record = chunk[-1]
        results = await post_chunk(sem, url, chunk)
        for status, _ in results:
            key = str(status)
            codes[key] = codes.get(key, 0) + 1
            if status == 200:
                n_ok += 1
        done += m
    elapsed = time.perf_counter() - t0
    summary = {
        "n": args.n,
        "ok_200": n_ok,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "url": url,
        "elapsed_s": round(elapsed, 3),
        "throughput_rps": round(args.n / elapsed, 1),
        "codes": codes,
        "generator": "splitmix64(idx,seed)",
        "payload_shape": {
            "categories": len(CATEGORIES),
            "severities": len(SEVERITIES),
            "operations": len(OPERATIONS),
            "tag_pool": len(TAG_POOL),
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out_dir:
        import os
        os.makedirs(args.out_dir, exist_ok=True)
        manifest = dict(summary)
        manifest["first_record"] = first_record
        manifest["last_record"] = last_record
        path = os.path.join(args.out_dir, f"manifest-n{args.n}-seed{args.seed}.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"manifest: {path}", file=__import__('sys').stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:19003")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
