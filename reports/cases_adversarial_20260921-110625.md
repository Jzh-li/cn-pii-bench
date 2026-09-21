# 真对抗语料评估报告

- **Endpoint**: `http://127.0.0.1:8401/v1/privacy/redact`
- **Cases**: `/home/jzhli/cn-pii-bench/fixtures/cases_adversarial.jsonl`
- **Total cases**: 28  |  **Errors**: 0
- **Corpus**: 28 unique texts / 0 duplicate groups / 21 unique shapes（数字归一后的句法形态数）
- **Entities（span 口径 = 主口径，expect 定分）**: TP=26  FP=0  FN=22
- **Precision**: 1.0000  **Recall**: 0.5417  **F1**: 0.7027

### 双口径对照（只报其中一个都是误导）

| 口径 | 含义 | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| `span` ⬅️ 门禁 | 同类型且互为子串 —— **主口径**，衡量「有没有找到」 | 26 | 0 | 22 | 1.0000 | 0.5417 | **0.7027** |
| `strict` | 值逐字相等 —— **最保守下界**，衡量「跨度和语料是否一致」 | 24 | 2 | 24 | 0.9231 | 0.5000 | **0.6486** |

- 两口径 F1 差 **+0.0541**。差距来自「找到了但跨度与语料不完全一致」（典型是地址少标/多标一个门牌）。**这不是能力差异，是口径差异** ——标称性能必须同时给出这两个数。
- **悲观口径（把每条 expect_miss 计入 FN）**: FN=23  Recall=0.5306  **F1=0.6933**
- **Latency p50/p95/p99 (ms)**: 0/0/7

## 已知弱点台账（expect_miss）

- 共 1 条；**已恢复 0**、仍漏报 1

| Case | 类型 | 值 | 原因 | 状态 |
|---|---|---|---|---|
| idcard_mask_adv-028 | zh_id_card | `110101********8531` | real_world_masked_id_card | ❌ 仍漏报 |

## by-subset（span 口径）

| Subset | n_cases | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| address | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| email | 4 | 4 | 0 | 2 | 1.0000 | 0.6667 | 0.8000 |
| id_card_masked | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| ip_address | 2 | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| mixed | 2 | 6 | 0 | 2 | 1.0000 | 0.7500 | 0.8571 |
| person_name | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| phone | 9 | 9 | 0 | 6 | 1.0000 | 0.6000 | 0.7500 |
| tool_call | 2 | 4 | 0 | 4 | 1.0000 | 0.5000 | 0.6667 |

## 漏报样本（span 口径，前 10）

- **phone_emoji_adv-001** (phone)  漏 1 个；期望 ['13800138000', '李娜']；检出 [('zh_phone', '13800138000')]
- **phone_emoji_adv-002** (phone)  漏 1 个；期望 ['13912345678', '李娜']；检出 [('zh_phone', '13912345678')]
- **phone_emoji_adv-003** (phone)  漏 1 个；期望 ['15012345678', '李娜']；检出 [('zh_phone', '15012345678')]
- **phone_nosep_adv-007** (phone)  漏 1 个；期望 ['17012345678', '赵敏']；检出 [('zh_phone', '17012345678')]
- **phone_nosep_adv-008** (phone)  漏 1 个；期望 ['13512345678', '赵敏']；检出 [('zh_phone', '13512345678')]
- **phone_nosep_adv-009** (phone)  漏 1 个；期望 ['13800138000', '赵敏']；检出 [('zh_phone', '13800138000')]
- **person_ethnic_adv-010** (person_name)  漏 1 个；期望 ['买买提·阿凡提']；检出 []
- **person_compound_adv-011** (person_name)  漏 1 个；期望 ['欧阳娜娜']；检出 []
- **addr_abbrev_adv-012** (address)  漏 1 个；期望 ['上海浦东张江园区 软件园 12 号楼']；检出 []
- **person_ethnic_adv-015** (person_name)  漏 1 个；期望 ['努尔买买提·吐尔逊']；检出 []
