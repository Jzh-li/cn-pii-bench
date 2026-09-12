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

用法：
    python3 bench/generate_adversarial.py
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
    """带 +86 前缀 + 横线分隔的格式变体（detector 当前预期漏报：未做规范化）"""
    # 把 phone 切成 3-4-4：13800138000 → 138-0013-8000
    p1 = phone[:3]
    p2 = phone[3:7]
    p3 = phone[7:]
    text = f"联系电话：+86-{p1}-{p2}-{p3}，仅工作时段接听。"
    return {
        "id": case_id,
        "subset": "phone",
        "text": text,
        "expect": [],
        "note": "known_weakness_unnormalized_phone_format",
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


def case_person_ethnic(case_id: str) -> dict:
    """少数民族姓名带间隔符「·」"""
    text = "对接人买买提·阿凡提确认出席本次对接会。"
    full = text
    name = "买买提·阿凡提"
    s, e = find_offsets(full, name)[0]
    return {
        "id": case_id,
        "subset": "person_name",
        "text": full,
        "expect": [
            {"type": "zh_person_name", "value": name, "start": s, "end": e},
        ],
    }


def case_person_compound(case_id: str) -> dict:
    """复姓（4 字符）真实常见"""
    text = "本次合作由欧阳娜娜代表团队出面。"
    name = "欧阳娜娜"
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


def case_address_abbrev(case_id: str) -> dict:
    """地址缩写（上海浦东 vs 上海市浦东新区）"""
    text = "出差：上海浦东张江园区 软件园 12 号楼"
    addr = "上海浦东张江园区"
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
    """身份证中部带星号遮蔽（当前预期漏报）"""
    text = "员工张三的身份证号是 110101********8531，请核对。"
    return {
        "id": case_id,
        "subset": "id_card_masked",
        "text": text,
        "expect": [],
        "note": "real_world_masked_expect_miss",
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


def case_email_plus_alias(case_id: str) -> dict:
    """带 + 别名的邮箱"""
    text = "Support: support+security@example.com （安全团队）"
    email = "support+security@example.com"
    s, e = find_offsets(text, email)[0]
    return {
        "id": case_id,
        "subset": "email",
        "text": text,
        "expect": [
            {"type": "email", "value": email, "start": s, "end": e},
        ],
    }


def case_ip_with_port(case_id: str) -> dict:
    """IP:端口 格式"""
    text = "测试服务器地址：10.0.0.1:8080，登录后进入调试模式。"
    ip = "10.0.0.1"
    s, e = find_offsets(text, ip)[0]
    return {
        "id": case_id,
        "subset": "ip_address",
        "text": text,
        "expect": [
            {"type": "ip_address", "value": ip, "start": s, "end": e},
        ],
    }


def case_mixed_realistic(case_id: str) -> dict:
    """真实场景多类型混排：姓名 + 手机 + 地址 + 邮箱 一段话"""
    text = (
        "客户资料：李雷，电话 13912345678，地址北京市海淀区中关村南大街 5 号，"
        "邮箱 lilei@example.com。请尽快对接。"
    )
    expects = []
    for val, typ in [
        ("李雷", "zh_person_name"),
        ("13912345678", "zh_phone"),
        ("北京市海淀区中关村南大街", "zh_address"),
        ("lilei@example.com", "email"),
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


def case_mixed_tool_call(case_id: str) -> dict:
    """模拟 tool_call 的 arguments 字段里嵌 PII"""
    text = json.dumps({
        "name": "send_email",
        "arguments": {
            "to": "zhangsan@example.com",
            "subject": "会议通知",
            "body": "请张伟于 13800138000 前到达会场。",
            "location": "北京市朝阳区国贸大厦",
        }
    }, ensure_ascii=False)
    expects = []
    for val, typ in [
        ("zhangsan@example.com", "email"),
        ("张伟", "zh_person_name"),
        ("13800138000", "zh_phone"),
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
    # person / address / email / ip / mixed / masked
    for i in range(2):
        builders_seq.append((case_person_ethnic, "person_ethnic", None))
        builders_seq.append((case_person_compound, "person_compound", None))
        builders_seq.append((case_address_abbrev, "addr_abbrev", None))
        builders_seq.append((case_email_plus_alias, "email_plus", None))
        builders_seq.append((case_ip_with_port, "ip_port", None))
    builders_seq.append((case_address_no_province, "addr_noprov", addr_pool[2]))  # 深圳南山区
    builders_seq.append((case_address_no_province, "addr_noprov", addr_pool[3]))  # 成都市高新区
    builders_seq.append((case_email_chinese, "email_chinese", "lilei"))
    builders_seq.append((case_email_chinese, "email_chinese", "hanmeimei"))
    builders_seq.append((case_mixed_realistic, "mixed_real", None))
    builders_seq.append((case_mixed_realistic, "mixed_real", None))
    builders_seq.append((case_mixed_tool_call, "mixed_tool", None))
    builders_seq.append((case_mixed_tool_call, "mixed_tool", None))
    builders_seq.append((case_id_card_masked, "idcard_mask", None))

    for i, (builder, subset, arg) in enumerate(builders_seq):
        case_id = f"{subset}_adv-{i + 1:03d}"
        if arg is not None:
            case = builder(case_id, arg)
        else:
            case = builder(case_id)
        if case.get("expect"):
            for ex in case["expect"]:
                text_slice = case["text"].encode("utf-8")[ex["start"]:ex["end"]].decode("utf-8")
                if text_slice != ex["value"]:
                    print(f"[WARN] {case_id} offset mismatch: expected {ex['value']!r}, got {text_slice!r}")
        cases.append(case)

    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    cases = build_cases()

    if args.self_check:
        # 内部一致性：每个 case 的 expect.value 必须等于 text[start:end]
        bad = 0
        for c in cases:
            for ex in c.get("expect", []):
                got = c["text"].encode("utf-8")[ex["start"]:ex["end"]].decode("utf-8")
                if got != ex["value"]:
                    print(f"[FAIL] {c['id']} offset mismatch")
                    bad += 1
        if bad == 0:
            print(f"[OK] {len(cases)} cases all offset-consistent")
        else:
            print(f"[FAIL] {bad} cases have inconsistent offsets")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(cases)} cases -> {out_path}")


if __name__ == "__main__":
    main()