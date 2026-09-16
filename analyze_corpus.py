#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_corpus.py —— 语料质量分析（不落报告，只出诊断表）。

与 bench_runner_adversarial.py 的分工：
- 评估器回答「多少分」；
- 本工具回答「**分数丢在哪**」，供定位优化方向。

两个子命令（都复用评估器的 `call_redact` 与匹配口径，不另写一份 —— 口径漂移是
这类分析最容易出的错）：

1. `forms` —— 按 类型 × 表面形态 × 载体 聚合漏检分布。
   典型用途：表面形式矩阵（`cases_adversarial_ext.jsonl`）跑完后，
   回答「漏检集中在哪些书写形态上」。case id 形如
   `zh_phone-dashed-chat_sf-011`，拆出三段聚合；非矩阵 case 归入 subset 分组。

2. `sanity` —— 核验语料里的 GT 值是否**结构自洽**。
   检测器对手机号 / 身份证 / 银行卡都带校验（号段白名单 / GB11643 校验位 / Luhn）。
   语料里放一个**不过校验**的假号码，测出来的是「检测器会不会对非法号码误报」，
   与语料意图相反 —— 而且失败长得像一个真实的形态缺口，会把优化方向带偏。
   本命令把漏检一分为二：语料缺陷（应改语料）vs 真缺口（应改检测器）。
   生成器的 `self_check()` 已内建同样检查（约定 5）；本命令用于
   审计**已生成**的语料与将来的第三方标注集。

用法：
    python3 analyze_corpus.py forms  --endpoint http://127.0.0.1:8413/v1/privacy/redact \\
        --cases fixtures/cases_adversarial_ext.jsonl
    python3 analyze_corpus.py sanity --endpoint http://127.0.0.1:8413/v1/privacy/redact \\
        --cases fixtures/cases_adversarial_ext.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from bench_runner_adversarial import Expect, Miss, call_redact

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "fixtures" / "cases_adversarial_ext.jsonl"
MATRIX_ID = re.compile(r"^(.+?)-([a-z_]+)-([a-z_]+)_sf-(\d+)$")


# ---- 结构校验（与网关 pkg/cn 的口径对齐，用于审计）-------------------------
_FW_HALF = {chr(0xFF10 + d): str(d) for d in range(10)}
_PHONE_PREFIXES = {
    "130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
    "145", "146", "147", "148", "149", "150", "151", "152", "153", "155",
    "156", "157", "158", "159", "166", "167", "170", "171", "172", "173",
    "174", "175", "176", "177", "178", "179", "180", "181", "182", "183",
    "184", "185", "186", "187", "188", "189", "190", "191", "192", "193",
    "195", "196", "197", "198", "199",
}


def _digits(v: str, keep_x: bool = False) -> str:
    out = []
    for c in v:
        c = _FW_HALF.get(c, c)
        if c.isdigit() or (keep_x and c in "Xx"):
            out.append(c)
    return "".join(out)


def _luhn_ok(v: str) -> bool:
    if not v or not v.isdigit():
        return False
    s, alt = 0, False
    for c in reversed(v):
        d = int(c)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        s += d
        alt = not alt
    return s % 10 == 0


_W = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_MAP = "10X98765432"


def _id_ok(v: str) -> bool:
    d = _digits(v, keep_x=True)
    if len(d) == 15:
        return True
    if len(d) != 18 or not d[:17].isdigit():
        return False
    return d[17].upper() == _MAP[sum(int(d[i]) * _W[i] for i in range(17)) % 11]


def _phone_ok(v: str) -> bool:
    d = _digits(v)
    if len(d) == 13 and d.startswith("86"):
        d = d[2:]
    return len(d) == 11 and d[0] == "1" and d[:3] in _PHONE_PREFIXES


def _bank_ok(v: str) -> bool:
    d = _digits(v)
    return 16 <= len(d) <= 19 and _luhn_ok(d)


def _ip_ok(v: str) -> bool:
    parts = v.split(":")[0].split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 and str(int(p)) == p for p in parts)


SANITY_CHECKS = {
    "zh_phone": _phone_ok,
    "zh_id_card": _id_ok,
    "zh_bank_card": _bank_ok,
    "ip_address": _ip_ok,
}


# ---- 采集 ------------------------------------------------------------------
def collect(cases_path: Path, endpoint: str) -> list[dict]:
    """跑一遍语料，返回每条 case 的 GT / 检出 / 漏检明细。"""
    rows = []
    for c in (json.loads(l) for l in cases_path.read_text(encoding="utf-8").splitlines() if l.strip()):
        expects = [Expect(e["type"], e["value"], e.get("start", 0), e.get("end", 0))
                   for e in c.get("expect", [])]
        misses = [Miss(m["type"], m["value"], m.get("reason", ""))
                  for m in c.get("expect_miss", [])]
        det, _lat, err = call_redact(endpoint, c["text"])
        if err:
            print(f"[ERR] {c['id']}: {err}", file=sys.stderr)
            rows.append({"id": c["id"], "err": err})
            continue
        det_set = {(d.type, d.value) for d in det}
        missing, fp = [], []
        for e in expects:
            hit = any(dt == e.type and dv and e.value and
                      (dv == e.value or dv in e.value or e.value in dv)
                      for dt, dv in det_set)
            if not hit:
                missing.append((e.type, e.value))
        # span 口径下与任何 GT 都无子串关系的检出 = FP
        for dt, dv in det_set:
            if not any(dt == e.type and dv and e.value and
                       (dv == e.value or dv in e.value or e.value in dv)
                       for e in expects):
                fp.append((dt, dv))
        row = {
            "id": c["id"],
            "subset": c.get("subset", "?"),
            "text": c["text"],
            "n_expect": len(expects),
            "expects": [(e.type, e.value) for e in expects],
            "detected": sorted(det_set),
            "missing": missing,
            "fp": fp,
            "err": "",
        }
        m = MATRIX_ID.match(c["id"])
        if m:
            row["typ"], row["form"], row["carrier"] = m.group(1), m.group(2), m.group(3)
        rows.append(row)
    return rows


def _agg(rows, keyf):
    g = defaultdict(lambda: {"cases": 0, "gt": 0, "miss": 0, "fp": 0})
    for r in rows:
        if r.get("err"):
            continue
        k = keyf(r)
        if k is None:
            continue
        g[k]["cases"] += 1
        g[k]["gt"] += r["n_expect"]
        g[k]["miss"] += len(r["missing"])
        g[k]["fp"] += len(r["fp"])
    return g


def _table(g, title, top=None):
    items = sorted(g.items(), key=lambda kv: (-kv[1]["miss"], kv[0]))
    if top:
        items = items[:top]
    out = [f"### {title}", "", "| key | cases | GT | 漏 | 漏检率 | FP |", "|---|---|---|---|---|---|"]
    for k, v in items:
        out.append(f"| {k} | {v['cases']} | {v['gt']} | {v['miss']} | "
                   f"{v['miss'] / v['gt'] if v['gt'] else 0.0:.3f} | {v['fp']} |")
    out.append("")
    return "\n".join(out)


# ---- 子命令 ----------------------------------------------------------------
def cmd_forms(rows: list[dict]) -> str:
    mat = [r for r in rows if not r.get("err") and "typ" in r]
    hand = [r for r in rows if not r.get("err") and "typ" not in r]
    gt = sum(r["n_expect"] for r in mat)
    miss = sum(len(r["missing"]) for r in mat)
    fp = sum(len(r["fp"]) for r in mat)

    out = ["# 形态缺口聚合（span 口径）", "",
           f"- 矩阵 case: {len(mat)}　手写/其他: {len(hand)}　err: {sum(1 for r in rows if r.get('err'))}",
           f"- 矩阵 GT: {gt}　漏: {miss}　漏检率: {miss / gt if gt else 0:.4f}　FP: {fp}", ""]
    out.append(_table(_agg(mat, lambda r: r["typ"]), "按类型（type）"))
    out.append(_table(_agg(mat, lambda r: r["form"]), "按表面形态（form）"))
    out.append(_table(_agg(mat, lambda r: r["carrier"]), "按载体（carrier）"))
    out.append(_table(_agg(mat, lambda r: f"{r['typ']}/{r['form']}"), "按 类型×形态", top=30))
    if hand:
        out.append(_table(_agg(hand, lambda r: "手写基线/全部"), "手写基线（汇总）", top=3))

    zero = defaultdict(int)
    for r in mat:
        if r["n_expect"] and len(r["missing"]) == r["n_expect"]:
            zero[f"{r['typ']}/{r['form']}"] += 1
    out += ["### 完全漏检（该 case 的 GT 一个都没抓到）", "", "| type/form | cases |", "|---|---|"]
    for k, v in sorted(zero.items(), key=lambda kv: -kv[1]):
        out.append(f"| {k} | {v} |")
    out.append("")

    seen = set()
    out += ["### 漏报样本示例（每个 type/form 取一条）", ""]
    for r in mat:
        if not r["missing"]:
            continue
        k = f"{r['typ']}/{r['form']}"
        if k in seen:
            continue
        seen.add(k)
        out.append(f"- `{r['id']}`　文本={r['text']!r}　期望={r['expects']}　检出={r['detected']}")
    out.append("")

    fps = [(r["id"], r["fp"]) for r in mat if r["fp"]]
    if fps:
        out += ["### 假阳性（span 口径）", ""]
        for cid, f in fps[:20]:
            out.append(f"- `{cid}`　{f}")
        out.append("")
    return "\n".join(out)


def cmd_sanity(rows: list[dict]) -> str:
    bad = defaultdict(list)
    gaps = defaultdict(list)
    ok_hit = defaultdict(int)
    bad_hit = defaultdict(int)
    gt_kind = defaultdict(int)

    for r in rows:
        if r.get("err") or "typ" not in r:
            continue
        key = f"{r['typ']}/{r['form']}"
        miss_set = {v for _t, v in r["missing"]}
        for typ, value in r["expects"]:
            check = SANITY_CHECKS.get(typ)
            valid = check(value) if check else True
            hit = value not in miss_set
            gt_kind[key] += 1
            if valid and hit:
                ok_hit[key] += 1
            elif valid and not hit:
                gaps[key].append((r["id"], value, r["text"]))
            elif not valid and hit:
                bad_hit[key] += 1
            else:
                bad[key].append((r["id"], value))

    out = ["# GT 结构自洽性核验", "",
           "| type/form | GT | 合法且命中 | 合法但漏（真缺口） | 非法且漏（语料缺陷） | 非法却命中 |",
           "|---|---|---|---|---|---|"]
    for k in sorted(gt_kind):
        out.append(f"| {k} | {gt_kind[k]} | {ok_hit[k]} | {len(gaps[k])} | "
                   f"{len(bad[k])} | {bad_hit[k]} |")
    out.append("")

    out += ["## A. 语料缺陷（GT 结构不合法）—— 应改语料", "",
            "| type/form | 值 | case |", "|---|---|---|"]
    if not bad:
        out.append("| （无） | | |")
    for k in sorted(bad):
        uniq = {}
        for cid, v in bad[k]:
            uniq.setdefault(v, cid)
        for v, cid in uniq.items():
            out.append(f"| {k} | `{v}` | {cid} |")
    out.append("")

    out += ["## B. 真缺口（GT 结构合法但漏检）—— 检测器形态覆盖", ""]
    if not gaps:
        out.append("（无）")
        out.append("")
    for k in sorted(gaps):
        uniq = {}
        for cid, v, t in gaps[k]:
            uniq.setdefault(v, (cid, t))
        out.append(f"### {k}（{len(gaps[k])} 条，去重 {len(uniq)} 个值）")
        out.append("")
        for v, (cid, t) in uniq.items():
            out.append(f"- `{v}` ← {cid}　文本={t!r}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["forms", "sanity"])
    ap.add_argument("--endpoint", default="http://127.0.0.1:8401/v1/privacy/redact")
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--out", default=None, help="同时写入该文件（便于贴进 README）")
    args = ap.parse_args()

    rows = collect(Path(args.cases), args.endpoint)
    text = cmd_forms(rows) if args.mode == "forms" else cmd_sanity(rows)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[analyze] -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
