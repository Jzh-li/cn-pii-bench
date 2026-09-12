#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_runner_adversarial.py —— 真对抗语料 F1 评估器（独立脚本）。

与 bench/runner.py（合成语料）的区别：
- 不走 /_api/detect（不存在），改走 /v1/privacy/redact 的 gate_only=true 模式
- 只跑 fixtures/cases_adversarial.jsonl
- 输出真实 F1 / by-subset / FN 列表，便于 README 引用

⚠️ 与合成语料并存，不是替代。本评估器跑出的是 **真对抗 F1**。

用法：
    python3 bench/bench_runner_adversarial.py \
        --endpoint http://127.0.0.1:8401/v1/privacy/redact \
        --cases bench/fixtures/cases_adversarial.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "fixtures" / "cases_adversarial.jsonl"


@dataclass
class Expect:
    type: str
    value: str
    start: int
    end: int


@dataclass
class Detect:
    type: str
    value: str
    start: int
    end: int


def call_redact(endpoint: str, text: str, timeout: float = 5.0) -> tuple[list[Detect], int, str]:
    """POST gate_only=true redact，返回 (entities, latency_ms, error).

    注：gate_only 模式的 entitySummary 只含 type/value/score，**不返回 start/end**。
    本评估器改用 (type, value) 严格匹配（offset 仅做内部 sanity check）。
    """
    body = json.dumps({"text": text, "gate_only": True}).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            latency_ms = int((time.perf_counter() - t0) * 1000)
            payload = json.loads(data)
            entities = [
                Detect(
                    type=e["type"],
                    value=e["value"],
                    # gate_only 不返回 offset，用 0/0 占位（runner 走 type+value 匹配）
                    start=0,
                    end=0,
                )
                for e in payload.get("entities", [])
            ]
            return entities, latency_ms, ""
    except urllib.error.URLError as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return [], latency_ms, f"urllib.URLError: {e.reason}"
    except (KeyError, json.JSONDecodeError) as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return [], latency_ms, f"decode error: {e}"


def strict_match(expected: list[Expect], detected: list[Detect]) -> tuple[int, int, int]:
    """(type, value) 二元组严格匹配 → (TP, FP, FN)。

    与 runner.py 区别：本评估器不强求 offset（gate_only 不暴露）；
    严格的 (type, value, start, end) 四元组匹配需要等 detector 暴露 offset 字段。
    """
    expected_set = {(e.type, e.value) for e in expected}
    detected_set = {(d.type, d.value) for d in detected}
    tp = len(expected_set & detected_set)
    fp = len(detected_set - expected_set)
    fn = len(expected_set - detected_set)
    return tp, fp, fn


def load_cases(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8401/v1/privacy/redact")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--report", action="store_true", help="print subset report")
    args = parser.parse_args()

    cases = list(load_cases(Path(args.cases)))

    # by-subset 统计
    subset_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_cases": 0})
    latencies = []
    fn_dump = []  # 漏报样本
    err_count = 0

    for c in cases:
        text = c["text"]
        expects = [
            Expect(type=e["type"], value=e["value"], start=e["start"], end=e["end"])
            for e in c.get("expect", [])
        ]
        detected, latency_ms, err = call_redact(args.endpoint, text)
        if err:
            err_count += 1
            print(f"[ERR] {c['id']}: {err}", file=sys.stderr)
            continue
        latencies.append(latency_ms)
        tp, fp, fn = strict_match(expects, detected)
        subset_stats[c["subset"]]["tp"] += tp
        subset_stats[c["subset"]]["fp"] += fp
        subset_stats[c["subset"]]["fn"] += fn
        subset_stats[c["subset"]]["n_cases"] += 1
        if fn > 0:
            fn_dump.append({
                "id": c["id"],
                "subset": c["subset"],
                "fn": fn,
                "expected": [e.value for e in expects],
                "detected": [(d.type, d.value) for d in detected],
            })

    # 汇总
    total_tp = sum(s["tp"] for s in subset_stats.values())
    total_fp = sum(s["fp"] for s in subset_stats.values())
    total_fn = sum(s["fn"] for s in subset_stats.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if not args.report:
        print(json.dumps({
            "n_cases": len(cases),
            "err_count": err_count,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "latency_p50_ms": int(statistics.median(latencies)) if latencies else 0,
            "latency_p95_ms": int(sorted(latencies)[int(len(latencies) * 0.95)]) if latencies else 0,
            "latency_p99_ms": int(sorted(latencies)[int(len(latencies) * 0.99)]) if latencies else 0,
        }, ensure_ascii=False, indent=2))
        return

    # Markdown 报告
    print(f"\n# 真对抗语料评估报告\n")
    print(f"- **Endpoint**: `{args.endpoint}`")
    print(f"- **Cases**: `{args.cases}`")
    print(f"- **Total cases**: {len(cases)}  |  **Errors**: {err_count}")
    print(f"- **Entities**: TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"- **Precision**: {precision:.4f}  **Recall**: {recall:.4f}  **F1**: {f1:.4f}")
    if latencies:
        sorted_lat = sorted(latencies)
        print(f"- **Latency p50/p95/p99 (ms)**: "
              f"{statistics.median(latencies):.0f}/"
              f"{sorted_lat[int(len(latencies) * 0.95)]:.0f}/"
              f"{sorted_lat[int(len(latencies) * 0.99)]:.0f}")

    print(f"\n## by-subset\n")
    print(f"| Subset | n_cases | TP | FP | FN | P | R | F1 |")
    print(f"|---|---|---|---|---|---|---|---|")
    for k in sorted(subset_stats):
        s = subset_stats[k]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        f1_sub = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"| {k} | {s['n_cases']} | {s['tp']} | {s['fp']} | {s['fn']} | {p:.4f} | {r:.4f} | {f1_sub:.4f} |")

    if fn_dump:
        print(f"\n## 漏报样本（前 10）\n")
        for x in fn_dump[:10]:
            print(f"- **{x['id']}** ({x['subset']})  漏 {x['fn']} 个；"
                  f"期望 {x['expected']}；检出 {x['detected']}")


if __name__ == "__main__":
    main()