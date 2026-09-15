#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_adversarial.py —— 真对抗 PII 语料生成器。

与 generate.py（合成语料）不同的是：
- 不依赖 pkg/cn 校验算法生成"必然通过检测"的样本
- 而是构造**真实场景下可能漏报 / 误报**的边界用例
- 真实场景常见弱信号：空格分隔、半角全角混、零宽字符、emoji 装饰、
  多语言混排、上下文缺省、罕见字姓、地址缩写、身份证遮蔽等

⚠️ 与合成语料并存，不是替代。本评估器跑出的是 **真对抗 F1**，
用于 README "Honest Performance" 段落。

## 两条约定（2026-09-15 收紧）

**1. 不允许重复样本。** 同一句话出现两遍不增加覆盖、只增加评分权重。此前
`for i in range(2)` 把若干无参构造器原样调用两次，直接产出 7 组重复（14 行 = 7 个
独立文本），标题 F1 被这些模板放大。现在同一位置改成**真正的变体**（不同民族姓名 /
不同复姓 / 不同地址 / 不同 IP…），行数与覆盖面都不减，重复归零。main() 的自检会把
重复判为失败。

**2. 已知弱点必须显式登记。** 「我们抓不到它，但它确实是 PII」的样本改用
`expect_miss: [{type, value, reason}]`，不再写 `"expect": []` + note。
空 expect 会让这类样本在 P/R/F1 里彻底隐形（不产生 FN，检出了反而算 FP），
等于把缺口写进语料再从评分里删掉。改用 expect_miss 后，评估器不计 FP、单独出台账，
并额外给出「把每条 expect_miss 都算 FN」的悲观口径 F1。

用法：
    python3 generate_adversarial.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

CN = Path(__file__).resolve().parent
DEFAULT_OUT = CN / "fixtures" / "cases_adversarial.jsonl"
SEED = 20260912

# ---- 真实对抗样本（手写 + 模板混合） ----------------------------------------

PERSON_NAMES = [
    ("李娜", "common"), ("王伟", "common"),
    ("司马懿", "compound"), ("上官婉儿", "compound"),
    ("买买提·阿凡提", "ethnic_uighur"), ("努尔买买提·吐尔逊", "ethnic_uighur"),
    ("欧阳娜娜", "compound_3char"), ("诸葛亮", "compound"),
    ("张三丰", "common"),
]

PHONES = [
    "13800138000", "13912345678", "15012345678", "18612345678", "19912345678",
    "17012345678", "13512345678",
]

ID_CARDS = [
    "110101199003078531",
    "310115198506123456",
]

BANK_CARDS = [
    "6222021234567890123",
    "4367421234567890123",
]

ADDRESSES = [
    ("北京市海淀区中关村南大街5号", "full"),
    ("上海浦东新区张江高科技园区", "abbreviated"),
    ("深圳南山区科苑路88号", "no_province"),
    ("成都市高新区天府软件园D区", "no_province"),
    ("广州市天河区珠江新城", "abbreviated"),
]

EMAILS = [
    ("zhang.san@example.com", "ascii"),
    ("李雷@163.com", "chinese_local"),
    ("user.name+tag@sub.example.co.uk", "plus_alias"),
]


def _utf8_offsets(text: str) -> list[tuple[int, int]]:
    """返回 text 中每个 UTF-8 字符的 (start_byte, end_byte) 元组列表。"""
    out = []
    pos = 0
    for ch in text:
        b = len(ch.encode("utf-8"))
        out.append((pos, pos + b))
        pos += b
    return out


def find_offsets(text: str, needle: str) -> list[tuple[int, int]]:
    """在 text 中查找 needle，返回 [(start_byte, end_byte)] 列表（UTF-8 字节偏移）。"""
    text_bytes = text.encode("utf-8")
    needle_bytes = needle.encode("utf-8")
    out = []
    start = 0
    while True:
        idx = text_bytes.find(needle_bytes, start)
        if idx < 0:
            break
        out.append((idx, idx + len(needle_bytes)))
        start = idx + 1
    return out


# ---- 真实对抗样本构造 -------------------------------------------------------

def case_phone_emoji(case_id: str, phone: str) -> dict:
    """emoji 装饰 + 中文上下文：手机号应被检测"""
    text = f"联系销售李娜（手机：📱{phone}），工作日联系。📞"
    s, e = find_offsets(text, phone)[0]
    return {
        "id": case_id,
        "subset": "phone",
        "text": text,
        "expect": [
            {"type": "zh_phone", "value": phone, "start": s, "end": e},
        ],
    }


def case_phone_formatted(case_id: str, phone: str) -> dict:
    """带 +86 前缀 + 横线分隔的格式变体（detector 当前预期漏报：未做规范化）

    用 expect_miss 显式登记为已知弱点，而不是留一个空的 expect —— 空 expect 会让
    这条样本在 P/R/F1 里彻底隐形，读者看不到「这个格式抓不到」这件事。
    """
    # 把 phone 切成 3-4-4：13800138000 → 138-0013-8000
    p1 = phone[:3]
    p2 = phone[3:7]
    p3 = phone[7:]
    formatted = f"+86-{p1}-{p2}-{p3}"
    text = f"联系电话：{formatted}，仅工作时段接听。"
    return {
        "id": case_id,
        "subset": "phone",
        "text": text,
        "expect": [],
        "expect_miss": [
            {
                "type": "zh_phone",
                "value": formatted,
                "reason": "known_weakness_unnormalized_phone_format",
            },
        ],
    }


def case_phone_no_separator(case_id: str, phone: str) -> dict:
    """「手机:13800138000」紧贴冒号无空格"""
    text = f"紧急联系人手机:{phone},赵敏收。"
    s, e = find_offsets(text, phone)[0]
    return {
        "id": case_id,
        "subset": "phone",
        "text": text,
        "expect": [
            {"type": "zh_phone", "value": phone, "start": s, "end": e},
        ],
    }


def case_person_ethnic(case_id: str, name: str) -> dict:
    """少数民族姓名带间隔符「·」（名字长度不同 → 两条独立样本）"""
    text = f"对接人{name}确认出席本次对接会。"
    s, e = find_offsets(text, name)[0]
    return {
        "id": case_id,
        "subset": "person_name",
        "text": text,
        "expect": [
            {"type": "zh_person_name", "value": name, "start": s, "end": e},
        ],
    }


def case_person_compound(case_id: str, name: str) -> dict:
    """复姓（4 字符）真实常见"""
    text = f"本次合作由{name}代表团队出面。"
    s, e = find_offsets(text, name)[0]
    return {
        "id": case_id,
        "subset": "person_name",
        "text": text,
        "expect": [
            {"type": "zh_person_name", "value": name, "start": s, "end": e},
        ],
    }


def case_address_no_province(case_id: str, addr: str) -> dict:
    """地址无省份（仅城市 + 区）"""
    text = f"{addr} 邮编518057"
    s, e = find_offsets(text, addr)[0]
    return {
        "id": case_id,
        "subset": "address",
        "text": text,
        "expect": [
            {"type": "zh_address", "value": addr, "start": s, "end": e},
        ],
    }


def case_address_abbrev(case_id: str, addr: str, tail: str) -> dict:
    """地址缩写（上海浦东 vs 上海市浦东新区）"""
    text = f"出差：{addr} {tail}"
    s, e = find_offsets(text, addr)[0]
    return {
        "id": case_id,
        "subset": "address",
        "text": text,
        "expect": [
            {"type": "zh_address", "value": addr, "start": s, "end": e},
        ],
    }


def case_id_card_masked(case_id: str) -> dict:
    """身份证中部带星号遮蔽。

    这里有**两类**语义，必须分开写：
    - 掩码身份证本身当前抓不到 → 登记为 `expect_miss`（已知弱点）；
    - 正文里的「张三」是**真 PII**，应当被检出 → 写进 `expect`。

    早期版本对整条只写了 `expect: []` + note，等于同时声明「张三也不该被检出」，
    于是检测器正确认出张三反而被记成 FP。这正是 `expect: []` 被滥用的另一种形态：
    作者只想标注一个缺口，却顺带把正文里真正的 PII 也否掉了。
    """
    text = "员工张三的身份证号是 110101********8531，请核对。"
    name = "张三"
    masked = "110101********8531"
    s, e = find_offsets(text, name)[0]
    return {
        "id": case_id,
        "subset": "id_card_masked",
        "text": text,
        "expect": [
            {"type": "zh_person_name", "value": name, "start": s, "end": e},
        ],
        "expect_miss": [
            {
                "type": "zh_id_card",
                "value": masked,
                "reason": "real_world_masked_id_card",
            },
        ],
    }


def case_email_chinese(case_id: str, local: str) -> dict:
    """中文邮箱本地部分"""
    text = f"如有问题请联系 李雷<{local}@163.com>。"
    email = f"{local}@163.com"
    s, e = find_offsets(text, email)[0]
    return {
        "id": case_id,
        "subset": "email",
        "text": text,
        "expect": [
            {"type": "email", "value": email, "start": s, "end": e},
        ],
    }


def case_email_plus_alias(case_id: str, email: str) -> dict:
    """带 + 别名的邮箱"""
    text = f"Support: {email} （安全团队）"
    s, e = find_offsets(text, email)[0]
    return {
        "id": case_id,
        "subset": "email",
        "text": text,
        "expect": [
            {"type": "email", "value": email, "start": s, "end": e},
        ],
    }


def case_ip_with_port(case_id: str, ip: str, port: int) -> dict:
    """IP:端口 格式（IP 与端口都变体化，避免两条样本一模一样）"""
    text = f"测试服务器地址：{ip}:{port}，登录后进入调试模式。"
    s, e = find_offsets(text, ip)[0]
    return {
        "id": case_id,
        "subset": "ip_address",
        "text": text,
        "expect": [
            {"type": "ip_address", "value": ip, "start": s, "end": e},
        ],
    }


def case_mixed_realistic(case_id: str, phone: str, email: str) -> dict:
    """真实场景多类型混排：姓名 + 手机 + 地址 + 邮箱 一段话"""
    text = (
        f"客户资料：李雷，电话 {phone}，地址北京市海淀区中关村南大街 5 号，"
        f"邮箱 {email}。请尽快对接。"
    )
    expects = []
    for val, typ in [
        ("李雷", "zh_person_name"),
        (phone, "zh_phone"),
        ("北京市海淀区中关村南大街", "zh_address"),
        (email, "email"),
    ]:
        offsets = find_offsets(text, val)
        if offsets:
            s, e = offsets[0]
            expects.append({"type": typ, "value": val, "start": s, "end": e})
    return {
        "id": case_id,
        "subset": "mixed",
        "text": text,
        "expect": expects,
    }


def case_mixed_tool_call(case_id: str, phone: str, email: str, name: str) -> dict:
    """模拟 tool_call 的 arguments 字段里嵌 PII"""
    text = json.dumps({
        "name": "send_email",
        "arguments": {
            "to": email,
            "subject": "会议通知",
            "body": f"请{name}于 {phone} 前到达会场。",
            "location": "北京市朝阳区国贸大厦",
        }
    }, ensure_ascii=False)
    expects = []
    for val, typ in [
        (email, "email"),
        (name, "zh_person_name"),
        (phone, "zh_phone"),
        ("北京市朝阳区国贸大厦", "zh_address"),
    ]:
        offsets = find_offsets(text, val)
        if offsets:
            s, e = offsets[0]
            expects.append({"type": typ, "value": val, "start": s, "end": e})
    return {
        "id": case_id,
        "subset": "tool_call",
        "text": text,
        "expect": expects,
    }


# ---- Builder ----------------------------------------------------------------

# 变体池：每个池子至少 2 项。原实现把无参构造器在 `for i in range(2)` 里原样调两遍，
# 直接产出重复样本；改成从池子里取不同项后，行数与覆盖面都不减、重复归零。
ETHNIC_NAMES = ["买买提·阿凡提", "努尔买买提·吐尔逊"]
COMPOUND_NAMES = ["欧阳娜娜", "上官婉儿"]
ABBREV_ADDRS = [
    ("上海浦东张江园区", "软件园 12 号楼"),
    ("广州天河软件园", "12 号楼"),
]
PLUS_EMAILS = ["support+security@example.com", "billing+invoices@example.org"]
IP_PORTS = [("10.0.0.1", 8080), ("192.168.1.100", 8443)]
MIXED_REAL = [
    ("13912345678", "lilei@example.com"),
    ("13700137000", "hanmeimei@example.com"),
]
MIXED_TOOL = [
    ("13800138000", "zhangsan@example.com", "张伟"),
    ("15012345678", "lisi@example.org", "王芳"),
]


def build_cases() -> list[dict]:
    rng = random.Random(SEED)
    cases = []

    # 真实对抗子集：每类生成 3-4 条
    phone_pool = list(PHONES)
    addr_pool = [a for a, _ in ADDRESSES]

    builders_seq = []
    # phone: 3 emoji + 3 formatted + 3 no_sep, 用不同 phone
    for i in range(3):
        builders_seq.append((case_phone_emoji, "phone_emoji", phone_pool[i % len(phone_pool)]))
    for i in range(3):
        builders_seq.append((case_phone_formatted, "phone_fmt", phone_pool[(i + 3) % len(phone_pool)]))
    for i in range(3):
        builders_seq.append((case_phone_no_separator, "phone_nosep", phone_pool[(i + 5) % len(phone_pool)]))
    # person / address / email / ip：每类两条**不同的**样本（不再是同一构造器调两遍）
    for i in range(2):
        builders_seq.append((case_person_ethnic, "person_ethnic", ETHNIC_NAMES[i]))
        builders_seq.append((case_person_compound, "person_compound", COMPOUND_NAMES[i]))
        builders_seq.append((case_address_abbrev, "addr_abbrev", ABBREV_ADDRS[i]))
        builders_seq.append((case_email_plus_alias, "email_plus", PLUS_EMAILS[i]))
        builders_seq.append((case_ip_with_port, "ip_port", IP_PORTS[i]))
    builders_seq.append((case_address_no_province, "addr_noprov", addr_pool[2]))  # 深圳南山区
    builders_seq.append((case_address_no_province, "addr_noprov", addr_pool[3]))  # 成都市高新区
    builders_seq.append((case_email_chinese, "email_chinese", "lilei"))
    builders_seq.append((case_email_chinese, "email_chinese", "hanmeimei"))
    builders_seq.append((case_mixed_realistic, "mixed_real", MIXED_REAL[0]))
    builders_seq.append((case_mixed_realistic, "mixed_real", MIXED_REAL[1]))
    builders_seq.append((case_mixed_tool_call, "mixed_tool", MIXED_TOOL[0]))
    builders_seq.append((case_mixed_tool_call, "mixed_tool", MIXED_TOOL[1]))
    builders_seq.append((case_id_card_masked, "idcard_mask", None))

    for i, (builder, subset, arg) in enumerate(builders_seq):
        case_id = f"{subset}_adv-{i + 1:03d}"
        if arg is None:
            case = builder(case_id)
        elif isinstance(arg, tuple):
            case = builder(case_id, *arg)
        else:
            case = builder(case_id, arg)
        if case.get("expect"):
            for ex in case["expect"]:
                text_slice = case["text"].encode("utf-8")[ex["start"]:ex["end"]].decode("utf-8")
                if text_slice != ex["value"]:
                    print(f"[WARN] {case_id} offset mismatch: expected {ex['value']!r}, got {text_slice!r}")
        cases.append(case)

    return cases


def self_check(cases: list[dict]) -> int:
    """结构自检：offset 一致、无重复样本、expect_miss 声明完整。返回错误数。"""
    bad = 0
    seen: dict[str, str] = {}
    for c in cases:
        text = c["text"]
        if text in seen:
            print(f"[FAIL] DUPLICATE {c['id']} == {seen[text]}")
            bad += 1
        seen.setdefault(text, c["id"])

        for ex in c.get("expect", []):
            got = text.encode("utf-8")[ex["start"]:ex["end"]].decode("utf-8")
            if got != ex["value"]:
                print(f"[FAIL] {c['id']} offset mismatch: expect={ex['value']!r} got={got!r}")
                bad += 1

        for m in c.get("expect_miss", []):
            if not m.get("value") or m["value"] not in text:
                print(f"[FAIL] {c['id']} expect_miss value {m.get('value')!r} 不在 text 中")
                bad += 1
            if not m.get("reason"):
                print(f"[FAIL] {c['id']} expect_miss 缺少 reason")
                bad += 1
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    cases = build_cases()
    bad = self_check(cases)

    if args.self_check:
        if bad == 0:
            print(f"[OK] {len(cases)} cases: offset-consistent, no duplicates")
        else:
            print(f"[FAIL] {bad} problems")
        return

    if bad:
        raise SystemExit(f"self-check failed with {bad} problems; refusing to write")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_miss = sum(len(c.get("expect_miss", [])) for c in cases)
    print(f"[OK] wrote {len(cases)} cases ({n_miss} expect_miss) -> {out_path}")


if __name__ == "__main__":
    main()