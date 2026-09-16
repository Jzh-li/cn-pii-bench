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

## 三条约定（2026-09-16 收紧）

**3. 模板里禁止内联真 PII 字面量，必须走「PII 值登记表」。**
此前 `case_phone_emoji` 等构造器把装饰上下文用的真姓名直接写死在 f-string 里
（`f"联系销售李娜（手机：📱{phone}）…"`），而 `expect` 只登记了被测的手机号。
后果是**语料在惩罚正确答案**：检测器正确认出正文里的「李娜」，反而被记成 FP。
实测这批 case 贡献了融合评估全部 11 个「误报」中的 8 个。

现在改为：所有会出现在文本里的 PII 值统一登记进 `PII_REGISTRY`，
构造完 case 后由 `finalize_case()` **扫描全文、自动补标**任何漏标的已登记值。
于是「模板夹带真 PII 没标」这一类缺陷在结构上不再可能发生。

**4. 地址类实体的标注跨度 = 能独立定位到具体地点的最长连续串。**
此前同一文件里地址标注自相矛盾：`addr_noprov` / `mixed_tool` 标完整跨度，
`addr_abbrev` / `mixed_real` 只标了一半（丢掉「软件园 12 号楼」「5 号」）。
现统一为完整跨度，并在登记表里只登记最长形式。
`audit_corpus.py`（外部审计）会把「检出跨度是 GT 跨度的真超串」报为粒度差，作为兜底。

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

# ⚠️ 这些值是**假号码，但必须自洽**（2026-09-16 修订）。
#
# 检测器对手机号 / 身份证 / 银行卡都带校验（号段白名单 / GB11643 校验位 / Luhn）。
# 语料里放一个**不过校验**的号码，测出来的不是「检测器能不能找到 PII」，而是
# 「检测器会不会对非法号码误报」—— 与语料意图相反，且失败长得像一个真实的
# 形态缺口，会把后续优化方向带偏。这类缺陷已由 self_check() 的㋒守护。
ID_CARDS = [
    "110101199003078531",   # 校验位 1 ✓
    "310115198506123454",   # 原写作 …456，实际校验位应为 4（不自洽，已修正）
]

# 校验位为 X 的身份证：`lower_x` 形态的**专用源值**。
#
# 不能拿任意号码把末位替换成 x —— 那只对校验位本来就是 X 的号码无副作用。
# 对校验位是 1 的号码，`v[:17] + "x"` 产物是一个非法号，检测器拒掉它是**正确行为**，
# 但会被误读成「检测器不认小写 x」的形态缺口。
ID_CARD_X = "11010119900307002X"

BANK_CARDS = [
    "6222021234567890128",  # 原 …123，Luhn 不过（sum=75），校验位补成 8
    "4367421234567890129",  # 原 …123，Luhn 不过，校验位补成 9
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

# ---- 装饰性上下文里的真 PII（约定 3：不许内联字面量，统一从这里取）---------
# 这些值原先写死在各自的构造器 f-string 里，是「模板夹带真 PII 没标」的根因。
SALES_NAME = "李娜"            # case_phone_emoji    : "联系销售李娜（手机：…）"
CARRIER_RECEIVER = "赵敏"      # case_phone_no_separator: "…,赵敏收。"
EMAIL_OWNER = "李雷"           # case_email_chinese  : "如有问题请联系 李雷<…>"
TOOL_LOCATION = "北京市朝阳区国贸大厦"                 # case_mixed_tool_call
MIXED_ADDRESS = "北京市海淀区中关村南大街 5 号"          # case_mixed_realistic（含门牌）
MASKED_ID_CARD = "110101********8531"                 # case_id_card_masked（已知弱点）
ID_CARD_HOLDER = "张三"        # case_id_card_masked : "员工张三的身份证号是…"


# ---- PII 值登记表（约定 3）-------------------------------------------------
#
# 任何**会出现在语料文本里**的 PII 值，都必须登记进来。
# `finalize_case()` 会扫全文，把漏标的已登记值自动补进 expect / expect_miss，
# `self_check()` 再验一遍「每个已登记值的每处出现都被标注覆盖」。
#
# 关键约束：**登记地址时只登记最长形式**（约定 4）。登记短的，
# 自动补标出来的跨度就会是短的，粒度问题会原样复现。
PII_REGISTRY: list[tuple[str, str]] = []
# 应当落进 expect_miss 而非 expect 的值（真 PII 但当前抓不到，显式登记为已知弱点）
WEAK_VALUES: set[str] = set()


def register(typ: str, value: str) -> str:
    """登记一个可能出现在语料文本里的 PII 值，返回值本身便于内联使用。"""
    if value and (typ, value) not in PII_REGISTRY:
        PII_REGISTRY.append((typ, value))
    return value


def register_weak(typ: str, value: str, ) -> str:
    """登记为「已知弱点」：真 PII，但不要求当前检测器抓到。"""
    register(typ, value)
    WEAK_VALUES.add(value)
    return value


def _build_registry() -> None:
    """把全部池子灌进登记表。在 build_cases() 开头调用一次。

    注意：这里引用的一些池子（ABBREV_ADDRS / PLUS_EMAILS / …）在文件后面的
    Builder 段落才定义；Python 在**调用时**才解析全局名，所以没问题。
    """
    # 人名
    for name, _kind in PERSON_NAMES:
        register("zh_person_name", name)
    for name in (SALES_NAME, CARRIER_RECEIVER, EMAIL_OWNER, ID_CARD_HOLDER):
        register("zh_person_name", name)
    for name in TOOL_NAMES:
        register("zh_person_name", name)
    # 手机
    for p in PHONES:
        register("zh_phone", p)
    for p in FORMATTED_PHONES:
        register("zh_phone", p)
    for p, _email in MIXED_REAL:
        register("zh_phone", p)
    # 证件 / 银行卡
    for c in ID_CARDS:
        register("zh_id_card", c)
    for c in BANK_CARDS:
        register("zh_bank_card", c)
    # 地址（约定 4：只登记最长形式）
    for addr, _kind in ADDRESSES:
        register("zh_address", addr)
    for addr, tail in ABBREV_ADDRS:
        register("zh_address", f"{addr} {tail}")
    register("zh_address", MIXED_ADDRESS)
    register("zh_address", TOOL_LOCATION)
    # 邮箱
    for e, _kind in EMAILS:
        register("email", e)
    for e in PLUS_EMAILS:
        register("email", e)
    for e in CHINESE_EMAILS:
        register("email", e)
    for _p, e in MIXED_REAL:
        register("email", e)
    for p, e, _name in MIXED_TOOL:
        register("zh_phone", p)
        register("email", e)
    # IP（带端口格式）
    for ip, _port in IP_PORTS:
        register("ip_address", ip)
    # 已知弱点：真 PII 但当前抓不到
    register_weak("zh_id_card", MASKED_ID_CARD)
    # 矩阵形态值（全角 / 分隔符 / 分组 / 大小写）——见「表面形式 × 载体矩阵」
    _register_matrix_values()


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
    """emoji 装饰 + 中文上下文：手机号应被检测

    注意 f-string 里的 `{SALES_NAME}` 是真姓名（约定 3）。此处只登记被测的手机号，
    「李娜」由 `finalize_case()` 自动补标 —— 不要手写死，否则下次改池子又会漂。
    """
    text = f"联系销售{SALES_NAME}（手机：📱{phone}），工作日联系。📞"
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
    """带 +86 前缀 + 横线分隔的格式变体。

    【2026-09-16 修订】原实现把它登记为 expect_miss（「detector 当前预期漏报：未做规范化」）。
    引入 NER 侧车后，这类格式**三条全部稳定检出**，「抓不到」的前提已不成立，
    故提为正式 expect。若将来某条回归，评估器会直接反映为 FN，不再被豁免掩盖。
    """
    # 把 phone 切成 3-4-4：13800138000 → 138-0013-8000
    p1 = phone[:3]
    p2 = phone[3:7]
    p3 = phone[7:]
    formatted = f"+86-{p1}-{p2}-{p3}"
    text = f"联系电话：{formatted}，仅工作时段接听。"
    s, e = find_offsets(text, formatted)[0]
    return {
        "id": case_id,
        "subset": "phone",
        "text": text,
        "expect": [
            {"type": "zh_phone", "value": formatted, "start": s, "end": e},
        ],
    }


def case_phone_no_separator(case_id: str, phone: str) -> dict:
    """「手机:13800138000」紧贴冒号无空格

    `{CARRIER_RECEIVER}收` —— 「收」是签收的动词，实体是「赵敏」。
    登记表登记的也是「赵敏」，`finalize_case()` 只会标「赵敏」，不会把「收」吞进去。
    """
    text = f"紧急联系人手机:{phone},{CARRIER_RECEIVER}收。"
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
    """地址缩写（上海浦东 vs 上海市浦东新区）

    【2026-09-16 修订 · 约定 4】GT 由「缩写头」改为**完整跨度** `addr + " " + tail`。
    原实现只标 `上海浦东张江园区`，而模型检出 `上海浦东张江园区 软件园 12 号楼` ——
    同类型、互为真子串、值不等，在严格口径下**同时计 1 FP + 1 FN**。
    同文件的 `addr_noprov` / `mixed_tool` 都标完整跨度，此处属自相矛盾，统一之。
    """
    full = f"{addr} {tail}"
    text = f"出差：{full}"
    s, e = find_offsets(text, full)[0]
    return {
        "id": case_id,
        "subset": "address",
        "text": text,
        "expect": [
            {"type": "zh_address", "value": full, "start": s, "end": e},
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

    现在「张三」和掩码身份证都进了登记表，`expect` 留空也能被 `finalize_case()`
    自动补齐 —— 下面显式写明是为了让读者一眼看到该条考什么。
    """
    text = f"员工{ID_CARD_HOLDER}的身份证号是 {MASKED_ID_CARD}，请核对。"
    s, e = find_offsets(text, ID_CARD_HOLDER)[0]
    return {
        "id": case_id,
        "subset": "id_card_masked",
        "text": text,
        "expect": [
            {"type": "zh_person_name", "value": ID_CARD_HOLDER, "start": s, "end": e},
        ],
        "expect_miss": [
            {
                "type": "zh_id_card",
                "value": MASKED_ID_CARD,
                "reason": "real_world_masked_id_card",
            },
        ],
    }


def case_email_chinese(case_id: str, local: str) -> dict:
    """中文邮箱本地部分

    `{EMAIL_OWNER}` 是真姓名（约定 3），构造时只登记邮箱，
    正文里的「李雷」由 `finalize_case()` 自动补标。
    """
    text = f"如有问题请联系 {EMAIL_OWNER}<{local}@163.com>。"
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
    """真实场景多类型混排：姓名 + 手机 + 地址 + 邮箱 一段话

    【2026-09-16 修订 · 约定 4】地址 GT 由 `北京市海淀区中关村南大街` 改为
    含门牌的完整跨度 `MIXED_ADDRESS`。原实现把文本里的「5 号」留在了实体外，
    模型检出完整跨度反被记 FP。
    """
    text = (
        f"客户资料：{EMAIL_OWNER}，电话 {phone}，地址{MIXED_ADDRESS}，"
        f"邮箱 {email}。请尽快对接。"
    )
    expects = []
    for val, typ in [
        (EMAIL_OWNER, "zh_person_name"),
        (phone, "zh_phone"),
        (MIXED_ADDRESS, "zh_address"),
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
            "location": TOOL_LOCATION,
        }
    }, ensure_ascii=False)
    expects = []
    for val, typ in [
        (email, "email"),
        (name, "zh_person_name"),
        (phone, "zh_phone"),
        (TOOL_LOCATION, "zh_address"),
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
# 派生池（登记表要用）：与 build_cases() 里实际取用的下标严格对应
#   case_phone_formatted  : phone_pool[(i+3) % 7]，i=0,1,2 → 下标 3,4,5
#   case_email_chinese    : 固定 local ∈ {"lilei", "hanmeimei"}
#   case_mixed_tool_call  : 第 3 项即姓名
FORMATTED_PHONES = [
    f"+86-{PHONES[(i + 3) % len(PHONES)][:3]}"
    f"-{PHONES[(i + 3) % len(PHONES)][3:7]}"
    f"-{PHONES[(i + 3) % len(PHONES)][7:]}"
    for i in range(3)
]
CHINESE_EMAILS = ["lilei@163.com", "hanmeimei@163.com"]
TOOL_NAMES = [name for _p, _e, name in MIXED_TOOL]


def finalize_case(case: dict) -> list[str]:
    """扫描全文，把任何**漏标的已登记 PII 值**自动补进 expect / expect_miss。

    这是约定 3 的执行者。它把「模板夹带真 PII 没标」从一个「靠人记得」的纪律问题，
    变成结构上不可能发生的事：只要值进了 `PII_REGISTRY`，就一定会被标注。

    规则：
      - 值在 `WEAK_VALUES` 里 → 补进 `expect_miss`（真 PII 但当前抓不到）
      - 其余 → 补进 `expect`
      - 已被现有 expect / expect_miss 跨度**完全覆盖**的出现 → 跳过（不重复标）
      - 同一个值在一段文本里出现多次 → 每次出现都补一条

    返回本次自动补标的描述列表（供 self_check 打印）。
    """
    text = case["text"]
    expect = case.setdefault("expect", [])
    miss = case.setdefault("expect_miss", [])
    covered = [(e["start"], e["end"]) for e in expect] + \
              [(m["start"], m["end"]) for m in miss if "start" in m]
    # expect_miss 允许不写 offset（bench_runner_adversarial.py 只读 type/value/reason），
    # 所以另用 value 集合去重，避免同一个弱点值被补成两条。
    miss_values = {m.get("value") for m in miss}
    added: list[str] = []

    # 长值优先：地址登记的是最长形式，先匹配长的可避免被短值截断
    for typ, value in sorted(PII_REGISTRY, key=lambda kv: -len(kv[1])):
        for (s, e) in find_offsets(text, value):
            if any(cs <= s and e <= ce for cs, ce in covered):
                continue
            if value in WEAK_VALUES:
                if value in miss_values:
                    continue
                miss.append({
                    "type": typ, "value": value, "start": s, "end": e,
                    "reason": "registered_known_weakness",
                })
                miss_values.add(value)
            else:
                if (typ, value) in {(x["type"], x["value"]) for x in expect}:
                    continue
                expect.append({"type": typ, "value": value, "start": s, "end": e})
            covered.append((s, e))
            added.append(f"{case['id']}: +{typ}={value!r}@{s}:{e}")

    case["expect_miss"] = miss
    if not miss:
        case.pop("expect_miss", None)
    return added


# ---- 表面形式 × 载体矩阵（2026-09-16 扩充）-----------------------------------
#
# 为什么用矩阵而不是继续手写：手写的 28 条已经把「若干典型形态」覆盖完了，再手写
# 就是重复劳动。而漏检的来源是**形态 × 上下文**的组合，组合才是该被穷举的维度。
#
# 两个维度各自的职责：
#   - 表面形式（SURFACE_FORMS）：同一个值的不同书写形态。分隔符 / 国际区号 / 全角 /
#     分组 / 大小写 —— 真实文本里最常见的漏检来源。
#   - 载体（CARRIERS）：值出现的上下文骨架（聊天 / 日志 / 工单 / 签名 / 换行 …）。
#     载体本身**不含任何 PII**，否则会引入没有登记的假阳性源（precision 门禁容不下）。
#
# 两条纪律（沿用约定 3 / 4）：
#
#   1. 构造器只负责「把值放进文本」，**不手写 expect** —— 标注由 finalize_case()
#      扫全文完成。这样新增形态时漏标在结构上不可能发生。
#
#   2. **只登记「属于 PII 自身书写形态」的变体，不登记装饰性标点。**
#      `138-0013-8000` 的分隔符在值内部 → 登记整串（与既有 case_phone_formatted 一致）；
#      `“13800138000”` 的引号在值外部 → 不登记，让 finalize_case 标内部跨度。
#      搞反了就是在制造标注缺陷：登记带引号的版本，会把「正确只框住号码本身」的
#      检测器判成跨度错误 —— 语料开始惩罚正确答案，正是约定 3 当初要修的那个毛病。
FULLWIDTH_DIGITS = {str(d): chr(0xFF10 + d) for d in range(10)}


def _fullwidth(v: str) -> str:
    return "".join(FULLWIDTH_DIGITS.get(c, c) for c in v)


def _phone_groups(v: str, sep: str) -> str:
    return sep.join((v[:3], v[3:7], v[7:]))


def _lower_x(v: str) -> str:
    """把校验位 X 写成小写 x。

    ⚠️ 不能无条件 `v[:17] + "x"`（2026-09-16 修订）。那对**校验位不是 X** 的号码
    会造出一个非法身份证号 —— 检测器的 `ValidIDCard` 拒掉它是正确行为，但结果看起来
    像「检测器不认小写校验位」的形态缺口。此处对校验位非 X 的值返回 None，
    由生成器跳过；`lower_x` 形态改用专用源值 ID_CARD_X。
    """
    return v[:17] + "x" if v and v[-1] in "Xx" else None


# type -> [(形态名, 变换函数, 是否需登记变换结果, 该形态的源值)]
#
# 第三项为 True 表示变换改写了值的字面（全角 / 分隔符 / 分组 / 大小写），
# 新字面在文本里是**另一个字符串**，不登记就没有任何标注覆盖它；
# 为 False 表示只是「值周围多了装饰」（冒号、引号、端口号），值本身没变，
# 已经在登记表里了。
#
# 第四项为形态级源值覆盖，None = 用类型级 MATRIX_SOURCES。存在的理由：
# 有些形态只对**特定构造的值**成立（lower_x 只对校验位为 X 的号成立）。
# 若把专用值塞进类型级源值列表，bare/grouped 会拿它再跑一遍，白增 20 条重复权重；
# 若让变换函数硬造，就会产出非法值。两个都不好，所以把「源值」下沉到形态级。
SURFACE_FORMS: dict[str, list[tuple]] = {
    "zh_phone": [
        ("bare", lambda v: v, False, None),
        ("dashed", lambda v: _phone_groups(v, "-"), True, None),
        ("spaced", lambda v: _phone_groups(v, " "), True, None),
        ("dotted", lambda v: _phone_groups(v, "."), True, None),
        ("country", lambda v: "+86-" + _phone_groups(v, "-"), True, None),
        ("fullwidth", _fullwidth, True, None),
    ],
    "email": [
        ("plain", lambda v: v, False, None),
        ("upper", lambda v: v.upper(), True, None),
    ],
    "zh_id_card": [
        ("bare", lambda v: v, False, None),
        ("grouped", lambda v: f"{v[:6]} {v[6:10]} {v[10:14]} {v[14:]}", True, None),
        ("lower_x", _lower_x, True, [ID_CARD_X]),
    ],
    "zh_bank_card": [
        ("bare", lambda v: v, False, None),
        ("grouped", lambda v: " ".join(v[i:i + 4] for i in range(0, len(v), 4)), True, None),
    ],
    "ip_address": [
        ("bare", lambda v: v, False, None),
        ("with_port", lambda v: f"{v}:8443", False, None),
    ],
}

# 每个类型取哪几个基础值。刻意取得少：矩阵的价值在**形态 × 载体的组合**，
# 同一形态灌 7 个不同号码只会增加评分权重，不增加覆盖面
# （评估器的 unique_shapes 指标就是为盯这件事而存在的）。
MATRIX_SOURCES: dict[str, list] = {
    "zh_phone": [PHONES[0], PHONES[1], PHONES[6]],
    "email": [EMAILS[0][0]],
    "zh_id_card": [ID_CARDS[0]],
    "zh_bank_card": [BANK_CARDS[0]],
    "ip_address": [IP_PORTS[0][0], IP_PORTS[1][0]],
}

# 载体骨架。{V} 处会被填入表面形式。
# 不含任何 PII，也刻意避开 date / url / plate / token= 这类会命中检测器的形态
# （一个没登记的命中就是一次计分的假阳性，这条语料存在的意义不是给自己制造噪声）。
CARRIERS: list[tuple[str, str]] = [
    ("chat", "同事在群里说：{V}，记得回一下。"),
    ("colon_tight", "联系方式:{V}"),
    ("paren", "（{V}）"),
    ("quote", "他说“{V}”就挂了。"),
    ("log", "[INFO] upstream contact={V} status=queued"),
    ("ticket", "【工单】回访方式：{V}；优先级：普通"),
    ("signature", "—— 如有疑问请联系 {V}"),
    ("trailing", "需要确认的信息是 {V}"),
    ("fw_colon", "联系方式：{V}，谢谢"),
    ("newline", "第一行说明\n{V}\n第三行备注"),
]


def _form_sources(typ: str, form: tuple) -> list:
    """某形态实际使用的源值：形态级覆盖优先，否则用类型级 MATRIX_SOURCES。"""
    override = form[3] if len(form) > 3 else None
    return override if override is not None else MATRIX_SOURCES[typ]


def _register_matrix_values() -> None:
    """把矩阵里所有「改写了字面」的形态登记进 PII_REGISTRY。

    必须在 build_matrix_cases() 之前调用（_build_registry 里调），
    否则 finalize_case() 找不到这些值，生成的 case 会带着空 expect 悄悄进入语料 ——
    一条谁也测不到的样本，比一条错标的样本更难发现。
    """
    for typ, _values in MATRIX_SOURCES.items():
        for form in SURFACE_FORMS[typ]:
            _name, fn, need_register = form[0], form[1], form[2]
            if not need_register:
                continue
            for base in _form_sources(typ, form):
                rendered = fn(base)
                if rendered is None:
                    continue
                register(typ, rendered)


def build_matrix_cases(tag: str = "sf") -> list[dict]:
    """按「类型 × 基础值 × 表面形式 × 载体」生成矩阵样本，标注全部走 finalize_case。"""
    cases: list[dict] = []
    n = 0
    for typ, _values in MATRIX_SOURCES.items():
        for form in SURFACE_FORMS[typ]:
            form_name, fn = form[0], form[1]
            for base in _form_sources(typ, form):
                rendered = fn(base)
                if rendered is None:
                    # 形态对该源值不适用（如 lower_x 遇到校验位非 X 的号）。
                    # 静默跳过是安全的：它不产生 case，也就不会产生未标注样本。
                    continue
                for carrier_name, tpl in CARRIERS:
                    text = tpl.replace("{V}", rendered)
                    n += 1
                    case = {
                        "id": f"{typ}-{form_name}-{carrier_name}_{tag}-{n:03d}",
                        "subset": "surface_matrix",
                        "text": text,
                        "expect": [],
                    }
                    case["_autofill"] = finalize_case(case)
                    if not case["expect"]:
                        # 兜底：形态改了字面却没登记，这一条会变成「零标注样本」。
                        # 零标注样本在 P/R/F1 里彻底隐形（不产生 FN，检出反而算 FP），
                        # 是最危险的一类缺陷，所以这里直接失败而不是打印告警。
                        raise SystemExit(
                            f"matrix case {case['id']} 生成后没有任何标注："
                            f"形态 {form_name!r} 的变换结果 {rendered!r} 未登记？"
                        )
                    cases.append(case)
    return cases


def build_cases(with_matrix: bool = False) -> list[dict]:
    _build_registry()
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
        # 约定 3：扫全文、自动补标任何漏标的已登记 PII
        case["_autofill"] = finalize_case(case)
        cases.append(case)

    # 矩阵扩充（--matrix）：默认关闭，保证既有 28 条的 corpus 逐字节不变。
    # 打开后产出的是一份更大的语料，用于评估形态 × 载体的覆盖面。
    if with_matrix:
        cases.extend(build_matrix_cases())

    # 补标之后再验一次 offset（自动补的偏移量也必须自洽）
    for case in cases:
        for ex in case.get("expect", []):
            text_slice = case["text"].encode("utf-8")[ex["start"]:ex["end"]].decode("utf-8")
            if text_slice != ex["value"]:
                print(f"[WARN] {case['id']} offset mismatch: "
                      f"expected {ex['value']!r}, got {text_slice!r}")

    return cases


# ---- 结构自洽性（约定 5，2026-09-16）----------------------------------------
#
# 语料里的假 PII 必须**结构自洽**：手机号落在真实号段、身份证过 GB11643 校验位、
# 银行卡过 Luhn。理由：检测器对这些类型都带校验，用一个不过校验的号码，
# 测到的就不是「能不能找到 PII」，而是「会不会对非法号码误报」——与语料意图相反。
#
# 这类缺陷的形态特别隐蔽：**它长得像一个真实的形态缺口**。实测例子：
# 语料写 `6222021234567890123`（Luhn sum=75，不过），报告显示
# 「zh_bank_card 召回 0%」，读起来像「检测器根本没有银行卡规则」；
# 实际是检测器**正确地**拒掉了一个非法号 —— 而那个号是语料自己造错了。
# 类似的还有 `v[:17] + "x"` 造出的非法身份证号（校验位本来是 1）。
#
# 结论：必须在生成器里变成结构上不可能，而不是靠人肉核对（与约定 3 同构）。
_HALF_DIGITS = {v: k for k, v in FULLWIDTH_DIGITS.items()}


def _sane_digits(v: str, keep_x: bool = False) -> str:
    """全角转半角 + 剥掉分隔符。

    形态变体在**语义**上必须仍是同一个合法值：`138-0013-8000` 与 `13800138000`
    是同一个号码的两种写法，校验的是剥掉分隔符之后的数字串。

    `keep_x=True` 供身份证使用：校验位 X/x 是号码的一部分，剥掉会让
    `11010119900307002x` 退化成 17 位，被误判为不自洽。
    """
    out = []
    for c in v:
        c = _HALF_DIGITS.get(c, c)
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


_ID_W = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_ID_MAP = "10X98765432"


def _id_card_ok(v: str) -> bool:
    d = _sane_digits(v, keep_x=True)
    if len(d) == 15:
        return True
    if len(d) != 18 or not d[:17].isdigit():
        return False
    return d[17].upper() == _ID_MAP[sum(int(d[i]) * _ID_W[i] for i in range(17)) % 11]


def _bank_card_ok(v: str) -> bool:
    d = _sane_digits(v)
    if not (16 <= len(d) <= 19):
        return False
    # 检测器侧的 ValidBankCard 允许 18 位，但会先排除合法身份证号；
    # 这里不复刻那条排除（语料作者不该把身份证号登记成银行卡）。
    return _luhn_ok(d)


def _phone_ok(v: str) -> bool:
    d = _sane_digits(v)
    if len(d) == 13 and d.startswith("86"):
        d = d[2:]
    prefixes = {p for p in cn_phone_prefixes()}
    return len(d) == 11 and d[0] == "1" and d[:3] in prefixes


def cn_phone_prefixes() -> list:
    """与检测器 pkg/cn.PhonePrefixes() 对齐的号段白名单（2026 年现行）。"""
    return [
        "130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
        "145", "146", "147", "148", "149",
        "150", "151", "152", "153", "155", "156", "157", "158", "159",
        "166", "167",
        "170", "171", "172", "173", "174", "175", "176", "177", "178", "179",
        "180", "181", "182", "183", "184", "185", "186", "187", "188", "189",
        "190", "191", "192", "193", "195", "196", "197", "198", "199",
    ]


def _ip_ok(v: str) -> bool:
    head = v.split(":")[0]
    parts = head.split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 and str(int(p)) == p for p in parts)


# 只对「带结构校验的类型」做检查。地名 / 人名 / 邮箱没有可机器判定的校验位，
# 不做结构校验（邮箱的中文 local part 也会被 ASCII 正则误杀）。
SANE_CHECKS = {
    "zh_phone": _phone_ok,
    "zh_id_card": _id_card_ok,
    "zh_bank_card": _bank_card_ok,
    "ip_address": _ip_ok,
}


def check_value_sanity() -> int:
    """㋒ 登记表里每个值必须结构自洽。返回失败数。

    含 `*` 的掩码值（如 `110101********8531`）跳过：掩码形式本来就不是完整号码。
    """
    bad = 0
    for typ, value in PII_REGISTRY:
        check = SANE_CHECKS.get(typ)
        if check is None or "*" in value or not value:
            continue
        if not check(value):
            print(f"[FAIL] 语料值不自洽：{typ}={value!r} 未通过结构校验"
                  f"（手机号段 / 身份证校验位 / Luhn）—— 检测器拒掉它是正确行为，"
                  f"这会被误读成形态缺口")
            bad += 1
    return bad


def self_check(cases: list[dict]) -> int:
    """结构自检：offset 一致、无重复样本、expect_miss 声明完整、登记表覆盖完整。

    新增两条（2026-09-16）：
      ㋐ **登记表覆盖**：文本中每一处已登记 PII 的出现，都必须被某条
         expect / expect_miss 的跨度**精确覆盖**。这条把「模板夹带真 PII 没标」
         变成不可能——只要值进了 PII_REGISTRY 就一定会被抓到。
      ㋑ **同类型跨度不得重叠**：约定 4 的机器化。若地址同时标了
         `上海浦东张江园区` 和 `上海浦东张江园区 软件园 12 号楼`，两者重叠 → 失败。
         它保证「地址只登记最长形式」是被强制执行而不是靠人记得。

    新增一条（2026-09-16，约定 5）：
      ㋒ **值结构自洽**：见 check_value_sanity()。防的是「假号码不过校验位，
      于是检测器正确地拒掉它，报告里却显示成一个形态缺口」。
    """
    bad = check_value_sanity()
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

        # ㋐ 登记表覆盖（正向）：登记了的必须被标注
        # expect 必带 offset；expect_miss 允许不带（评估器只读 type/value/reason），
        # 不带时就地按 value 反查偏移，保证覆盖判定不因「没写 offset」而误报。
        covered = [(e["start"], e["end"]) for e in c.get("expect", [])]
        for m in c.get("expect_miss", []):
            if "start" in m and "end" in m:
                covered.append((m["start"], m["end"]))
            else:
                covered.extend(find_offsets(text, m["value"]))
        for typ, value in PII_REGISTRY:
            for (s, e) in find_offsets(text, value):
                if not any(cs <= s and e <= ce for cs, ce in covered):
                    print(f"[FAIL] {c['id']} 已登记 PII {typ}={value!r} 在 {s}:{e} "
                          f"未被任何 expect/expect_miss 覆盖 —— 检查是否漏标")
                    bad += 1

        # ㋐ 登记表覆盖（反向）：标注了的值必须已登记
        # 保证登记表是**完备**的。少了这条，有人内联一个新字面量并手工标注后，
        # 登记表就漏了它，正向检查再也守不住。
        # 残留缺口：内联新字面量且**完全不标**时，两条离线检查都拦不住 ——
        # 这类只能靠外部审计（audit_corpus.py 用两引擎实测反查）兜底。
        registered = {(t, v) for t, v in PII_REGISTRY}
        for item in c.get("expect", []) + c.get("expect_miss", []):
            if (item["type"], item["value"]) not in registered:
                print(f"[FAIL] {c['id']} 标注了未登记的 PII "
                      f"{item['type']}={item['value']!r} —— 请先 register() 再标注")
                bad += 1

        # ㋑ 同类型跨度不得重叠（约定 4 的机器化）
        spans = [(e["type"], e["value"], e["start"], e["end"])
                 for e in c.get("expect", [])]
        for i in range(len(spans)):
            t1, v1, s1, e1 = spans[i]
            for t2, v2, s2, e2 in spans[i + 1:]:
                if t1 != t2:
                    continue
                if s1 < e2 and s2 < e1:
                    print(f"[FAIL] {c['id']} 同类型跨度重叠："
                          f"{t1} {v1!r}[{s1}:{e1}] ∩ {v2!r}[{s2}:{e2}] "
                          f"—— 地址等可变长实体只应登记最长形式")
                    bad += 1
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--matrix", action="store_true",
        help="额外生成「表面形式 × 载体」矩阵样本（默认关闭，保证既有语料逐字节不变）",
    )
    args = parser.parse_args()

    cases = build_cases(with_matrix=args.matrix)
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
    n_auto = sum(len(c.get("_autofill") or []) for c in cases)
    for c in cases:
        for line in (c.get("_autofill") or []):
            print(f"  [auto] {line}")
    with out_path.open("w", encoding="utf-8") as f:
        for c in cases:
            c.pop("_autofill", None)          # 内部调试字段，不写进语料
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_miss = sum(len(c.get("expect_miss", [])) for c in cases)
    n_gt = sum(len(c.get("expect", [])) for c in cases)
    print(f"[OK] wrote {len(cases)} cases / GT {n_gt} / expect_miss {n_miss} -> {out_path}")
    print(f"     其中 finalize_case() 自动补标 {n_auto} 处（约定 3）")
    print(f"     PII 登记表共 {len(PII_REGISTRY)} 个值（含 {len(WEAK_VALUES)} 个已知弱点）")


if __name__ == "__main__":
    main()