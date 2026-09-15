#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_runner_adversarial.py —— 真对抗语料 F1 评估器（独立脚本）。

与 runner.py（合成语料）的区别：
- 不走 /_api/detect（不存在），改走 /v1/privacy/redact 的 gate_only=true 模式
- 只跑 fixtures/cases_adversarial.jsonl
- 输出真实 F1 / by-subset / FN 列表，便于 README 引用

⚠️ 与合成语料并存，不是替代。本评估器跑出的是 **真对抗 F1**。

## 两处刻意设计（2026-09-15）

**1. expect_miss（已知弱点台账）**
语料里有一类样本是「我们知道检测器目前抓不到、但它是真 PII」。早期写法是
`"expect": []` + 一句 note —— 后果是这种样本在 P/R/F1 里**完全不可见**（空期望既
不产生 TP/FN，检出了反而算 FP）。等于先把弱点写进语料、再从评分里删掉，标题 F1 会
系统性高估真实能力。

现在改为显式 `expect_miss: [{"type","value","reason"}]`：
- 命中 expect_miss 的检出**不计 FP**（它本来就是真 PII，抓到了是好事）；
- 单独输出「已知弱点台账」，逐条标注 已恢复 / 仍漏报；
- 报告同时给出**悲观口径 F1**（把每条 expect_miss 都当成 FN），两个数字并排看，
  读者能自己判断标题数字里有多少是「已知缺口没算进去」。

**2. 语料规模口径**
标题 F1 是**行口径**。行数不等于独立样本数：同一句话抄两遍只增加权重、不增加覆盖。
因此报告里固定输出 `unique_texts` / `duplicate_groups` / `unique_shapes`（数字归一后
的文本骨架数），让「N 行到底代表多少独立句法形态」一眼可见。fixture 已去重，重复组
应为 0；validate.py 会把新增重复判为错误，防止再退化。

用法：
    python3 bench_runner_adversarial.py \\
        --endpoint http://127.0.0.1:8401/v1/privacy/redact \\
        --cases fixtures/cases_adversarial.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "fixtures" / "cases_adversarial.jsonl"

# 数字归一：把连续数字折叠成单个 D，用于得到「文本骨架」。
# 「13800138000」与「13912345678」骨架相同 —— 它们只差一个取值，句法形态是一条。
_DIGITS = re.compile(r"\d+")


@dataclass
class Expect:
    type: str
    value: str
    start: int
    end: int


@dataclass
class Miss:
    """已知弱点：真 PII，但当前检测器抓不到。"""

    type: str
    value: str
    reason: str


@dataclass
class Detect:
    type: str
    value: str
    start: int
    end: int


# 【2026-09-15】无代理 opener：bench 永远打本地网关，绝不能走系统代理。
# runner.py 早已踩过这个坑（Windows 下 urllib 读注册表代理，127.0.0.1 被劫持导致
# 挂起且 timeout 不生效）；本评估器原先用裸 urlopen，在设了 http_proxy 的环境里
# 同样会把 127.0.0.1 交给代理。统一改走显式空代理 opener。
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call_redact(endpoint: str, text: str, timeout: float = 5.0) -> tuple[list[Detect], int, str]:
    """POST gate_only=true redact，返回 (entities, latency_ms, error)。

    注：gate_only 模式的 entitySummary 只含 type/value/score，**不返回 start/end**。
    本评估器改用 (type, value) 严格匹配（offset 仅做内部 sanity check）。
    """
    body = json.dumps({"text": text, "gate_only": True}).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
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


def strict_match(
    expected: list[Expect], detected: list[Detect], misses: list[Miss]
) -> tuple[int, int, int, list[Detect]]:
    """(type, value) 二元组严格匹配 → (TP, FP, FN, 被豁免的检出)。

    与 runner.py 区别：本评估器不强求 offset（gate_only 不暴露）；
    严格的 (type, value, start, end) 四元组匹配需要等 detector 暴露 offset 字段。

    豁免规则（本评估器特有）：检出的值若出现在本 case 的 expect_miss 里，就不算 FP。
    这类值是**真 PII**，只是语料预先声明「我们抓不到」。抓到了应当加分而不是扣分；
    旧实现把它们算成 FP，等于惩罚「检测器比语料作者预期更强」。
    按 value 豁免而非 (type,value)：真值是同一个串，类型标注分歧不该让豁免失效。
    """
    expected_set = {(e.type, e.value) for e in expected}
    detected_set = {(d.type, d.value) for d in detected}
    miss_values = {m.value for m in misses}

    excused = [d for d in detected if d.value in miss_values]
    # 已计入 expected 的检出不算豁免（避免同时算 TP 与豁免）
    excused = [d for d in excused if (d.type, d.value) not in expected_set]

    tp = len(expected_set & detected_set)
    fp = len(detected_set - expected_set - {(d.type, d.value) for d in excused})
    fn = len(expected_set - detected_set)
    return tp, fp, fn, excused


def corpus_quality(cases: list[dict]) -> dict:
    """语料规模口径：行数 / 唯一文本数 / 重复组 / 文本骨架数。

    行口径的 F1 会被重复句子的权重带偏，因此报告必须同时给出这几个数。
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for c in cases:
        key = json.dumps(
            {"text": c["text"], "expect": c.get("expect", [])},
            ensure_ascii=False,
            sort_keys=True,
        )
        groups[key].append(c.get("id", "?"))
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    shapes = {_DIGITS.sub("D", c["text"]) for c in cases}
    return {
        "n_cases": len(cases),
        "unique_texts": len({c["text"] for c in cases}),
        "duplicate_groups": len(dup),
        "duplicate_ids": [v for v in dup.values()],
        "unique_shapes": len(shapes),
    }


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
    parser.add_argument("--out", default=str(ROOT / "reports"),
                        help="报告落盘目录（跟 runner.py / carriers.py 口径一致）")
    parser.add_argument("--no-write", action="store_true",
                        help="只打印，不落盘（--report 之外不需要时用）")
    args = parser.parse_args()

    cases = list(load_cases(Path(args.cases)))
    quality = corpus_quality(cases)

    # by-subset 统计
    subset_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_cases": 0})
    latencies = []
    fn_dump = []  # 漏报样本
    err_count = 0
    # 已知弱点台账：id → 该 case 的每条 expect_miss 及是否被恢复
    ledger: list[dict] = []
    n_excused = 0

    for c in cases:
        text = c["text"]
        expects = [
            Expect(type=e["type"], value=e["value"], start=e["start"], end=e["end"])
            for e in c.get("expect", [])
        ]
        misses = [
            Miss(type=m["type"], value=m["value"], reason=m.get("reason", ""))
            for m in c.get("expect_miss", [])
        ]
        detected, latency_ms, err = call_redact(args.endpoint, text)
        if err:
            err_count += 1
            print(f"[ERR] {c['id']}: {err}", file=sys.stderr)
            continue
        latencies.append(latency_ms)
        tp, fp, fn, excused = strict_match(expects, detected, misses)
        n_excused += len(excused)
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
        if misses:
            det_values = {d.value for d in detected}
            ledger.append({
                "id": c["id"],
                "subset": c["subset"],
                "items": [
                    {
                        "type": m.type,
                        "value": m.value,
                        "reason": m.reason,
                        "recovered": m.value in det_values,
                    }
                    for m in misses
                ],
            })

    # 汇总（严格口径：expect 定 P/R/F1，已知弱点不计 FP 也不计 FN）
    total_tp = sum(s["tp"] for s in subset_stats.values())
    total_fp = sum(s["fp"] for s in subset_stats.values())
    total_fn = sum(s["fn"] for s in subset_stats.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 悲观口径：每条 expect_miss 都当成 FN。用于让「已知缺口」在标题数字外可见。
    n_miss_total = sum(len(x["items"]) for x in ledger)
    n_miss_open = sum(1 for x in ledger for it in x["items"] if not it["recovered"])
    pess_fn = total_fn + n_miss_total
    pess_recall = total_tp / (total_tp + pess_fn) if (total_tp + pess_fn) > 0 else 0.0
    pess_f1 = (
        2 * precision * pess_recall / (precision + pess_recall)
        if (precision + pess_recall) > 0
        else 0.0
    )

    summary = {
        "n_cases": len(cases),
        "corpus": quality,
        "err_count": err_count,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "expect_miss_total": n_miss_total,
        "expect_miss_open": n_miss_open,
        "expect_miss_recovered": n_miss_total - n_miss_open,
        "excused_detections": n_excused,
        "pessimistic_fn": pess_fn,
        "pessimistic_recall": round(pess_recall, 4),
        "pessimistic_f1": round(pess_f1, 4),
        "latency_p50_ms": int(statistics.median(latencies)) if latencies else 0,
        "latency_p95_ms": int(sorted(latencies)[int(len(latencies) * 0.95)]) if latencies else 0,
        "latency_p99_ms": int(sorted(latencies)[int(len(latencies) * 0.99)]) if latencies else 0,
        "by_subset": {k: dict(v) for k, v in subset_stats.items()},
    }

    if not args.report:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not args.no_write:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            js = out_dir / f"adversarial_{time.strftime('%Y%m%d-%H%M%S')}.json"
            js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[adversarial] json-> {js}", file=sys.stderr)
        return

    # Markdown 报告
    out: list[str] = []
    out.append("# 真对抗语料评估报告")
    out.append("")
    out.append(f"- **Endpoint**: `{args.endpoint}`")
    out.append(f"- **Cases**: `{args.cases}`")
    out.append(f"- **Total cases**: {len(cases)}  |  **Errors**: {err_count}")
    out.append(f"- **Corpus**: {quality['unique_texts']} unique texts / "
               f"{quality['duplicate_groups']} duplicate groups / "
               f"{quality['unique_shapes']} unique shapes（数字归一后的句法形态数）")
    if quality["duplicate_groups"]:
        for ids in quality["duplicate_ids"]:
            out.append(f"  - ⚠️ 重复组: {', '.join(ids)}")
    out.append(f"- **Entities（严格口径，expect 定分）**: TP={total_tp}  FP={total_fp}  FN={total_fn}")
    out.append(f"- **Precision**: {precision:.4f}  **Recall**: {recall:.4f}  **F1**: {f1:.4f}")
    out.append(f"- **悲观口径（把每条 expect_miss 计入 FN）**: "
               f"FN={pess_fn}  Recall={pess_recall:.4f}  **F1={pess_f1:.4f}**")
    if latencies:
        sorted_lat = sorted(latencies)
        out.append(f"- **Latency p50/p95/p99 (ms)**: "
                   f"{statistics.median(latencies):.0f}/"
                   f"{sorted_lat[int(len(latencies) * 0.95)]:.0f}/"
                   f"{sorted_lat[int(len(latencies) * 0.99)]:.0f}")

    out.append("")
    out.append("## 已知弱点台账（expect_miss）")
    out.append("")
    if not ledger:
        out.append("（本语料未声明已知弱点）")
    else:
        out.append(f"- 共 {n_miss_total} 条；**已恢复 {n_miss_total - n_miss_open}**、仍漏报 {n_miss_open}")
        if n_excused:
            out.append(f"- 另有 {n_excused} 次检出命中已知弱点 → 不计 FP（真 PII，抓到是加分）")
        out.append("")
        out.append("| Case | 类型 | 值 | 原因 | 状态 |")
        out.append("|---|---|---|---|---|")
        for x in ledger:
            for it in x["items"]:
                status = "✅ 已恢复" if it["recovered"] else "❌ 仍漏报"
                out.append(f"| {x['id']} | {it['type']} | `{it['value']}` | "
                           f"{it['reason']} | {status} |")

    out.append("")
    out.append("## by-subset")
    out.append("")
    out.append("| Subset | n_cases | TP | FP | FN | P | R | F1 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k in sorted(subset_stats):
        s = subset_stats[k]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        f1_sub = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out.append(f"| {k} | {s['n_cases']} | {s['tp']} | {s['fp']} | {s['fn']} | "
                   f"{p:.4f} | {r:.4f} | {f1_sub:.4f} |")

    if fn_dump:
        out.append("")
        out.append("## 漏报样本（前 10）")
        out.append("")
        for x in fn_dump[:10]:
            out.append(f"- **{x['id']}** ({x['subset']})  漏 {x['fn']} 个；"
                       f"期望 {x['expected']}；检出 {x['detected']}")

    text = "\n".join(out)
    print(text)

    # 落盘：md + json（与 runner.py / carriers.py 一致，供 CI / 第三方消费）
    if not args.no_write:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        md_path = out_dir / f"adversarial_{ts}.md"
        js_path = out_dir / f"adversarial_{ts}.json"
        md_path.write_text(text + "\n", encoding="utf-8")
        js_path.write_text(json.dumps(
            {"endpoint": args.endpoint, "cases": str(args.cases), **summary},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[adversarial] md  -> {md_path}", file=sys.stderr)
        print(f"[adversarial] json-> {js_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
