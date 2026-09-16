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

**3. 双口径（2026-09-16 定案）**
新增 `--match strict|span|both`（默认 `both`）与 `--gate-on strict|span`（默认 `span`）：

- **跨度口径（span）= 主口径**：同类型且值互为子串即算命中，衡量「有没有找到」。
- **严格口径（strict）= 最保守下界**：值逐字相等，衡量「跨度和语料是否逐字一致」。

动因：严格口径对 `zh_address` 这类可变长实体过苛 —— 检出了 `北京市海淀区中关村南大街`
而 GT 是 `…南大街 5 号`，被记 **1 FP + 1 FN**（一次找到却被罚两次）。实测两口径 F1
相差约 0.09，差距 100% 来自地址跨度。**这不是能力差异，是口径差异**，因此两个数字
必须并排出现，只报其中一个都是误导。细节见 `span_match()`。

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

    ⚠️ 必须显式带 `include_values: true`（2026-09-16）。
    网关侧 `entities[].value` 的默认值已改为**不回显**：默认开着会让网关同时成为
    「提交一段文本 → 取出其中 PII 原文」的提取接口，而判定拦放只需要 type。
    本评估器是本地可信工具，且匹配口径就是 (type, value)，因此显式opt in。
    不带这个字段时 entities 里没有 value，每条有检出的样本都会 KeyError →
    被记成 err 而不是 FN，指标会以一种「看起来很安静」的方式塌成 0。
    """
    body = json.dumps({
        "text": text,
        "gate_only": True,
        "include_values": True,
    }).encode("utf-8")
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


def span_match(
    expected: list[Expect], detected: list[Detect], misses: list[Miss]
) -> tuple[int, int, int, list[Detect]]:
    """跨度口径：(type 相同) 且 (值相等 或 互为子串) 即算命中。

    【2026-09-16 新增】为什么需要这个口径：

    严格口径对 `zh_address` 这类**可变长**实体过苛。实测例子：
      文本  `…地址北京市海淀区中关村南大街 5 号…`
      GT    `北京市海淀区中关村南大街 5 号`
      检出  `北京市海淀区中关村南大街`      ← 少了一个门牌
    人看这是「找到了地址，只是略短」；严格口径却记 1 FP + 1 FN —— 一次漏了
    一个实体，却被罚两次。反过来检测器标得**更全**（多吞一个后缀）也一样被罚。

    所以地址类的严格相等本质上在测「跨度和语料作者的习惯是否一致」，而不是
    「有没有找到 PII」。跨度口径把这一层与真实能力分离出来。

    ⚠️ 这个口径**更宽**，不能单独用来宣称性能。按项目定案：
      - **跨度口径为主口径**（衡量「有没有找到」）
      - **严格口径作为最保守下界**（衡量「跨度和语料是否逐字一致」）
    两个数字必须并排出现，只报其中一个都是误导。

    已知的宽口径风险：若检测器输出极短的值（如单字），可能子串命中大量 GT。
    实践中检测器输出的是实体跨度，且同类型才比较，风险可控；如后续引入
    单字级检测器，需给子串方向加长度比例下限。
    """
    expected_set = {(e.type, e.value) for e in expected}
    detected_set = {(d.type, d.value) for d in detected}
    miss_values = {m.value for m in misses}

    hit_exp: set[tuple[str, str]] = set()
    matched_det: set[tuple[str, str]] = set()
    for t, v in expected_set:
        if not v:
            continue
        for dt, dv in detected_set:
            if dt != t or not dv:
                continue
            if v == dv or v in dv or dv in v:
                hit_exp.add((t, v))
                matched_det.add((dt, dv))
                break

    tp = len(hit_exp)
    fn = len(expected_set - hit_exp)
    excused = [d for d in detected
               if d.value in miss_values and (d.type, d.value) not in matched_det]
    fp = len(detected_set - matched_det - {(d.type, d.value) for d in excused})
    return tp, fp, fn, excused


MATCH_MODES = ("strict", "span")


def match(mode: str, expected, detected, misses):
    """口径分发。`strict` 保持历史行为，`span` 见 span_match 文档。"""
    if mode == "strict":
        return strict_match(expected, detected, misses)
    if mode == "span":
        return span_match(expected, detected, misses)
    raise ValueError(f"unknown match mode: {mode}")


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


def gate_exit_code(args, precision: float, recall: float, f1: float) -> int:
    """按 --min-* 阈值判定是否放行。未指定任何阈值时恒为 0（纯报告模式）。

    三条阈值的语义（用于 CI 守门）：

    - `precision`：**必须**不低于下限。这是 PII 网关最不能退的指标 ——
      误报会破坏正常对话内容（把非 PII 换成占位符），比漏检更容易被用户察觉。
    - `recall` / `f1`：用**门禁口径**（`--gate-on`，默认 `span` = 主口径）。
      弱点台账另有悲观口径，不作为门禁 —— 悲观口径包含「已声明且接受」的缺口，
      拿它做门禁会永远红。
    - 门禁口径必须与 `--match` 选定的范围相容：只跑 `--match strict` 时不能再指定
      `--gate-on span`，main() 会直接报错退出，而不是静默换口径。
    """
    limits = [
        ("precision", args.min_precision, precision),
        ("recall", args.min_recall, recall),
        ("f1", args.min_f1, f1),
    ]
    bad = [(n, lim, got) for n, lim, got in limits if lim is not None and got < lim]
    if not bad:
        if any(lim is not None for _, lim, _ in limits):
            print("[adversarial] 阈值守门：通过", file=sys.stderr)
        return 0
    for name, lim, got in bad:
        print(f"[adversarial] 阈值守门：未通过 —— {name}={got:.4f} < {lim:.4f}",
              file=sys.stderr)
    print("[adversarial] 说明：低召回若是「已知弱点」所致，请查报告里的 expect_miss 台账；"
          "低精确率通常是检测层误报，属回归。", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8401/v1/privacy/redact")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--report", action="store_true", help="print subset report")
    parser.add_argument("--out", default=str(ROOT / "reports"),
                        help="报告落盘目录（跟 runner.py / carriers.py 口径一致）")
    parser.add_argument("--no-write", action="store_true",
                        help="只打印，不落盘（--report 之外不需要时用）")
    # 阈值守门（供 CI 用）。阈值刻意放在命令行而不是代码里 —— 让「当前承诺的底线」
    # 在 CI 配置文件里一眼可见，改阈值必须在 review 里显式出现。
    parser.add_argument("--min-precision", type=float, default=None,
                        help="精确率下限（门禁口径，见 --gate-on），低于则退出码 2")
    parser.add_argument("--min-recall", type=float, default=None,
                        help="召回率下限（门禁口径，见 --gate-on），低于则退出码 2")
    parser.add_argument("--min-f1", type=float, default=None,
                        help="F1 下限（门禁口径，见 --gate-on），低于则退出码 2")
    # 匹配口径（2026-09-16 定案）：跨度口径为主、严格口径作最保守下界。
    # 两个数字必须并排出现；只报其中一个都是误导。详见 span_match() 文档。
    parser.add_argument("--match", choices=["strict", "span", "both"], default="both",
                        help="匹配口径：strict=值逐字相等（最保守下界）；"
                             "span=同类型且互为子串（主口径）；both=两个都算都出（默认）")
    parser.add_argument("--gate-on", choices=["strict", "span"], default=None,
                        help="用哪个口径做阈值门禁；默认 span（主口径），需在 --match 选定的范围内")
    args = parser.parse_args()

    cases = list(load_cases(Path(args.cases)))
    quality = corpus_quality(cases)

    # 口径：strict（最保守下界）/ span（主口径，见 span_match 文档）/ both（默认，两个都出）
    modes = MATCH_MODES if args.match == "both" else (args.match,)

    # by-subset 统计（按口径分开累计）
    subset_stats = {m: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_cases": 0})
                    for m in modes}
    latencies = []
    fn_dump = {m: [] for m in modes}  # 漏报样本（分口径）
    err_count = 0
    # 已知弱点台账：id → 该 case 的每条 expect_miss 及是否被恢复
    ledger: list[dict] = []
    n_excused = {m: 0 for m in modes}

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
        for m in modes:
            tp, fp, fn, excused = match(m, expects, detected, misses)
            n_excused[m] += len(excused)
            subset_stats[m][c["subset"]]["tp"] += tp
            subset_stats[m][c["subset"]]["fp"] += fp
            subset_stats[m][c["subset"]]["fn"] += fn
            subset_stats[m][c["subset"]]["n_cases"] += 1
            if fn > 0:
                fn_dump[m].append({
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

    # 汇总：**按口径分别计算**（expect 定 P/R/F1，已知弱点不计 FP 也不计 FN）
    totals: dict[str, dict] = {}
    for m in modes:
        t_tp = sum(s["tp"] for s in subset_stats[m].values())
        t_fp = sum(s["fp"] for s in subset_stats[m].values())
        t_fn = sum(s["fn"] for s in subset_stats[m].values())
        p = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
        r = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        totals[m] = {"tp": t_tp, "fp": t_fp, "fn": t_fn,
                     "precision": p, "recall": r, "f1": f}

    # 门禁用哪个口径：默认**主口径 = span**（项目 2026-09-16 定案），
    # `--gate-on strict` 可切回最保守口径。
    primary = args.gate_on or ("span" if "span" in modes else modes[0])
    if primary not in modes:
        raise SystemExit(f"--gate-on {primary} 不在 --match {args.match} 选定的口径 {modes} 里")

    total_tp = totals[primary]["tp"]
    total_fp = totals[primary]["fp"]
    total_fn = totals[primary]["fn"]
    precision, recall, f1 = (totals[primary]["precision"], totals[primary]["recall"],
                             totals[primary]["f1"])

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
        "match_mode": args.match,
        "primary_mode": primary,
        "err_count": err_count,
        # 顶层字段 = 主口径，保持与历史报告的键名兼容
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        # 两个口径并排：只报其中一个都是误导
        "by_match_mode": {
            m: {
                "tp": totals[m]["tp"], "fp": totals[m]["fp"], "fn": totals[m]["fn"],
                "precision": round(totals[m]["precision"], 4),
                "recall": round(totals[m]["recall"], 4),
                "f1": round(totals[m]["f1"], 4),
            }
            for m in modes
        },
        "expect_miss_total": n_miss_total,
        "expect_miss_open": n_miss_open,
        "expect_miss_recovered": n_miss_total - n_miss_open,
        "excused_detections": n_excused[primary],
        "pessimistic_fn": pess_fn,
        "pessimistic_recall": round(pess_recall, 4),
        "pessimistic_f1": round(pess_f1, 4),
        "latency_p50_ms": int(statistics.median(latencies)) if latencies else 0,
        "latency_p95_ms": int(sorted(latencies)[int(len(latencies) * 0.95)]) if latencies else 0,
        "latency_p99_ms": int(sorted(latencies)[int(len(latencies) * 0.99)]) if latencies else 0,
        "by_subset": {k: dict(v) for k, v in subset_stats[primary].items()},
        "by_subset_all_modes": {m: {k: dict(v) for k, v in subset_stats[m].items()}
                                for m in modes},
    }

    if not args.report:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not args.no_write:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            js = out_dir / f"{Path(args.cases).stem}_{time.strftime('%Y%m%d-%H%M%S')}.json"
            js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[adversarial] json-> {js}", file=sys.stderr)
        return gate_exit_code(args, precision, recall, f1)

    # Markdown 报告
    out: list[str] = []
    out.append("# 真对抗语料评估报告")
    out.append("")
    out.append(f"- **Endpoint**: `{args.endpoint}`")
    out.append(f"- **Cases**: `{args.cases}`")
    out.append(f"- **Total cases**: {len(cases)}  |  **Errors**: {err_count}")
    if err_count:
        # 请求/解码失败的样本不产生 TP/FP/FN，指标会**安静地**塌掉而不是显式报警。
        # 2026-09-16 就真实发生过一次：网关把 gate_only 的 entities[].value 默认省略后，
        # 每条有检出的样本都 KeyError → err_count=17，P/R/F1 全变 0。
        # 当时误以为「指标变差」，实际是度量工具坏了。
        out.append("")
        out.append(f"> ⚠️ **有 {err_count} 条样本因请求/解码失败被跳过**，"
                   f"它们既不计 TP 也不计 FN。此报告的比例指标**不可直接与历史对比**，"
                   f"请先查明 err 原因（常见：响应字段缺失、endpoint 未就绪）。")
        out.append("")
    out.append(f"- **Corpus**: {quality['unique_texts']} unique texts / "
               f"{quality['duplicate_groups']} duplicate groups / "
               f"{quality['unique_shapes']} unique shapes（数字归一后的句法形态数）")
    if quality["duplicate_groups"]:
        for ids in quality["duplicate_ids"]:
            out.append(f"  - ⚠️ 重复组: {', '.join(ids)}")
    out.append(f"- **Entities（{primary} 口径 = 主口径，expect 定分）**: "
               f"TP={total_tp}  FP={total_fp}  FN={total_fn}")
    out.append(f"- **Precision**: {precision:.4f}  **Recall**: {recall:.4f}  **F1**: {f1:.4f}")
    if len(modes) > 1:
        out.append("")
        out.append("### 双口径对照（只报其中一个都是误导）")
        out.append("")
        out.append("| 口径 | 含义 | TP | FP | FN | P | R | F1 |")
        out.append("|---|---|---|---|---|---|---|---|")
        _desc = {
            "span": "同类型且互为子串 —— **主口径**，衡量「有没有找到」",
            "strict": "值逐字相等 —— **最保守下界**，衡量「跨度和语料是否一致」",
        }
        for mm in ("span", "strict"):
            if mm not in modes:
                continue
            tt = totals[mm]
            star = " ⬅️ 门禁" if mm == primary else ""
            out.append(f"| `{mm}`{star} | {_desc[mm]} | {tt['tp']} | {tt['fp']} | {tt['fn']} | "
                       f"{tt['precision']:.4f} | {tt['recall']:.4f} | **{tt['f1']:.4f}** |")
        gap = totals["span"]["f1"] - totals["strict"]["f1"] \
            if "span" in modes and "strict" in modes else 0.0
        if abs(gap) > 1e-9:
            out.append("")
            out.append(f"- 两口径 F1 差 **{gap:+.4f}**。差距来自「找到了但跨度与语料不完全一致」"
                       f"（典型是地址少标/多标一个门牌）。**这不是能力差异，是口径差异** ——"
                       f"标称性能必须同时给出这两个数。")
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
        if n_excused[primary]:
            out.append(f"- 另有 {n_excused[primary]} 次检出命中已知弱点 "
                       f"→ 不计 FP（真 PII，抓到是加分）")
        out.append("")
        out.append("| Case | 类型 | 值 | 原因 | 状态 |")
        out.append("|---|---|---|---|---|")
        for x in ledger:
            for it in x["items"]:
                status = "✅ 已恢复" if it["recovered"] else "❌ 仍漏报"
                out.append(f"| {x['id']} | {it['type']} | `{it['value']}` | "
                           f"{it['reason']} | {status} |")

    out.append("")
    out.append(f"## by-subset（{primary} 口径）")
    out.append("")
    out.append("| Subset | n_cases | TP | FP | FN | P | R | F1 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k in sorted(subset_stats[primary]):
        s = subset_stats[primary][k]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        f1_sub = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out.append(f"| {k} | {s['n_cases']} | {s['tp']} | {s['fp']} | {s['fn']} | "
                   f"{p:.4f} | {r:.4f} | {f1_sub:.4f} |")

    if fn_dump[primary]:
        out.append("")
        out.append(f"## 漏报样本（{primary} 口径，前 10）")
        out.append("")
        for x in fn_dump[primary][:10]:
            out.append(f"- **{x['id']}** ({x['subset']})  漏 {x['fn']} 个；"
                       f"期望 {x['expected']}；检出 {x['detected']}")

    text = "\n".join(out)
    print(text)

    # 落盘：md + json（与 runner.py / carriers.py 一致，供 CI / 第三方消费）
    if not args.no_write:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        # ⚠️ 文件名必须带语料名（2026-09-16 修）。时间戳只有秒级精度，不带语料名时
        # 同一目录里跑两个语料会**静默互相覆盖** —— 而「先跑 28 条基线、再跑 318 条
        # 矩阵」正是最常见的用法。runner.py 早已踩过这个坑（240 条报告曾被 180 条
        # 报告覆盖掉），此处与它的 `phase0_<engine>_<corpus>_<ts>` 口径对齐。
        corpus = Path(args.cases).stem
        md_path = out_dir / f"{corpus}_{ts}.md"
        js_path = out_dir / f"{corpus}_{ts}.json"
        md_path.write_text(text + "\n", encoding="utf-8")
        js_path.write_text(json.dumps(
            {"endpoint": args.endpoint, "cases": str(args.cases), **summary},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[adversarial] md  -> {md_path}", file=sys.stderr)
        print(f"[adversarial] json-> {js_path}", file=sys.stderr)

    return gate_exit_code(args, precision, recall, f1)


if __name__ == "__main__":
    raise SystemExit(main())
