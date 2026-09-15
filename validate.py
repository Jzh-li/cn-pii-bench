#!/usr/bin/env python3
"""validate.py —— 校验 fixtures/*.jsonl 的结构与一致性。

可作 E2E 自检 / CI 步骤的轻量 gate（不依赖 Go）。同时打印每个 case 的实体数与
语料规模口径。

## 为什么要查重复（2026-09-15 补）

同一句话抄两遍**不增加任何覆盖，只增加评分权重**。对抗语料曾出现 7 组完全重复
（14 行 = 7 个独立文本），导致标题 F1 被这些模板放大。重复是纯粹的手误，没有
正当理由，因此这里判为 **hard error**，让它在 CI 里立刻响。

## expect vs expect_miss

- `expect`：期望被检出的实体（计入 TP/FN）。
- `expect_miss`：**已知弱点台账** —— 真 PII，但当前检测器抓不到。它不是「期望为空」，
  必须写明 type/value/reason。用空 expect + note 的老写法会让弱点在评分里彻底隐形，
  见 bench_runner_adversarial.py 顶部说明。

用法：
    python3 validate.py                          # 校验合成语料
    python3 validate.py --cases fixtures/cases_adversarial.jsonl
    python3 validate.py --all                    # 校验全部语料
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"

KNOWN_TYPES = {
    "zh_person_name", "zh_phone", "zh_id_card", "zh_bank_card", "zh_address",
    "email", "ip_address", "date", "api_key", "token", "password", "plate",
    "url", "us_ssn", "credit_card",
}


def validate_file(path: Path) -> int:
    """校验单个 jsonl 语料，返回错误数。"""
    if not path.exists():
        print(f"FAIL: missing {path}")
        return 1

    n_total = 0
    n_pos = 0
    n_neg = 0
    n_miss = 0
    type_counter: dict[str, int] = {}
    errors: list[str] = []
    dup_groups: dict[str, list[str]] = defaultdict(list)
    texts: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {lineno}: invalid JSON: {e}")
                continue

            n_total += 1
            text: str = c.get("text", "")
            cid: str = c.get("id", f"line{lineno}")
            expect = c.get("expect", [])
            misses = c.get("expect_miss", [])
            texts.add(text)

            # 重复检测：完全相同的 (text, expect) 视为同一份样本被抄了多遍。
            dup_groups[json.dumps(
                {"text": text, "expect": expect}, ensure_ascii=False, sort_keys=True
            )].append(cid)

            if not expect and not misses:
                n_neg += 1
            if expect:
                n_pos += 1

            # detector（pkg/types.Entity）的 start/end 为 UTF-8 字节偏移，
            # 因此这里按字节切片比较，不能用 Python 字符串的字符索引。
            tb = text.encode("utf-8")
            for j, e in enumerate(expect):
                t = e.get("type", "")
                v = e.get("value", "")
                s = e.get("start", -1)
                ed = e.get("end", -1)
                type_counter[t] = type_counter.get(t, 0) + 1
                if t not in KNOWN_TYPES:
                    errors.append(f"{cid}[{j}]: 未知实体类型 {t!r}")
                if s < 0:
                    errors.append(f"{cid}[{j}]: start 必须 ≥ 0")
                if ed > len(tb):
                    errors.append(f"{cid}[{j}]: end {ed} > len(text bytes) {len(tb)}")
                if ed <= s:
                    errors.append(f"{cid}[{j}]: end 必须 > start")
                if 0 <= s < ed <= len(tb):
                    seg = tb[s:ed].decode("utf-8", "replace")
                    if seg != v:
                        errors.append(
                            f"{cid}[{j}]: value={v!r} != text[{s}:{ed}]={seg!r}"
                        )

            # expect_miss：必须写明 type/value/reason，且值确实出现在文本里。
            for j, m in enumerate(misses):
                n_miss += 1
                mt = m.get("type", "")
                mv = m.get("value", "")
                mr = m.get("reason", "")
                if mt not in KNOWN_TYPES:
                    errors.append(f"{cid}.expect_miss[{j}]: 未知实体类型 {mt!r}")
                if not mv:
                    errors.append(f"{cid}.expect_miss[{j}]: value 不能为空")
                elif mv not in text:
                    errors.append(
                        f"{cid}.expect_miss[{j}]: value={mv!r} 未出现在 text 中"
                    )
                if not mr:
                    errors.append(
                        f"{cid}.expect_miss[{j}]: 必须写明 reason（已知弱点要可追溯）"
                    )

            # 空洞的 expect_miss 会变成「永远不判错」的噪音，直接拦掉。
            if isinstance(misses, list) and misses and not any(
                m.get("value") for m in misses if isinstance(m, dict)
            ):
                errors.append(f"{cid}: expect_miss 不得全为空值")

    dups = {k: v for k, v in dup_groups.items() if len(v) > 1}
    for ids in dups.values():
        errors.append(f"重复样本（同 text+expect）: {', '.join(ids)}")

    print(f"=== validate.py: {path.name} ===")
    print(f"total cases: {n_total}  positive: {n_pos}  negative: {n_neg}  "
          f"expect_miss: {n_miss}")
    print(f"unique texts: {len(texts)}  duplicate groups: {len(dups)}")
    if type_counter:
        print("实体按类型统计:")
        for t, n in sorted(type_counter.items()):
            print(f"  {t}: {n}")

    if errors:
        print()
        print(f"FAIL ({len(errors)} 错误):")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... 还有 {len(errors) - 20} 项未显示")
        return len(errors)

    print("PASS: 全部样本结构合法")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(FIXTURES / "cases.jsonl"))
    parser.add_argument("--all", action="store_true", help="校验 fixtures/ 下全部 jsonl")
    args = parser.parse_args()

    if args.all:
        paths = sorted(FIXTURES.glob("cases*.jsonl"))
    else:
        paths = [Path(args.cases)]

    total_err = 0
    for i, p in enumerate(paths):
        if i:
            print()
        total_err += validate_file(p)
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())
