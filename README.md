# cn-pii-bench（LLMate Gate 自评估语料 · 契约 §9 · 探路者探路）

LLMate Gate 的中文 PII 评测语料与评估器。**双层口径**：L1 检测器级（span 四元组
匹配）+ L2 泄漏级（结构化载体对等性 + 往返）。

设计借鉴 [privaite-bench](https://github.com/mandiant/privaite-bench)
（`bench_precision_recall.py` 实体级 P/R/F1 + FP/FN 逐条 dump；
`bench_structured.py` 4 种结构化载体对等性 + 往返），但**实现重写**：
- 评分对象是**检测器 span**（`/_api/detect`）和**端到端 payload**（`/v1/privacy/redact`），
  不是字符串包含；
- 只覆盖中文 PII，没有 multi-language / spaCy / ONNX。

## 目录

```
bench/
├─ README.md                    本文件
├─ runner.py                    L1 检测器级评估（强四元组匹配 + by-subset + FP/FN dump + JSON 报告 + selftest）
├─ bench_runner_adversarial.py  真对抗评估（expect_miss 豁免 + 弱点台账 + 严格/悲观双口径）
├─ carriers.py                  L2 泄漏级评估（4 载体对等性 + 往返 + 守门模式 + selftest）
├─ generate.py                  240 条合成中文语料生成器（身份证校验位 / 手机号段 / Luhn / 复姓）
├─ generate_en.py               180 条英文语料生成器
├─ generate_adversarial.py      28 条真对抗语料生成器（含 expect_miss 台账）
├─ validate.py                  三语料结构自检（重复即硬错）
├─ fixtures/
│   ├─ cases.jsonl              240 条合成中文样本（8 子集 × 30）
│   ├─ cases_en.jsonl           180 条英文样本
│   ├─ cases_adversarial.jsonl  28 条真对抗样本
│   └─ cases.schema.json
└─ reports/                     评估产物（md + json 时间戳报告）
```

## 语料

### 合成中文 `cases.jsonl`（240 条）

8 子集 × 30，全部 `provenance=synthetic`（由 `generate.py` 合成）：

| 子集 | 内容 | 关键风险点 |
|---|---|---|
| person_name | 单姓 + 复姓 + 少数民族姓名 | 复姓长度对齐、上下文无关消歧 |
| phone | 三大运营商号段 | 段号合法 + 11 位 |
| id_card | 18 位 + ISO 7064 校验位 | 校验位合法 |
| bank_card | BIN + Luhn | Luhn 通过 |
| address | 直辖市 / 省市区逆向 | 行政区划对齐 |
| email | 域名白名单 | 邮箱格式 |
| tool_call | JSON 字符串里嵌 PII | 字符串内的 PII 检测 |
| mixed | 中英混合长文本 | 多类同现 |
| adversarial | 错位 / 间隔符 / 英文夹杂 | 干扰符 |

**注意**：在自作者合成语料上自评，F1 高只说明「检测器与生成器对同一套仿真规则
达成一致」，**不构成真实场景结论**（详见 `Specs/03-测试规约.md` §4.3.4）。

### 英文基线 `cases_en.jsonl`（180 条）

`credit_card` / `email` / `ip_address` / `plate` / `url` / `us_ssn` 各 30。

### 真对抗 `cases_adversarial.jsonl`（28 条）

手工构造，覆盖合成语料无法生成的**真实形态**：
复姓人名（欧阳娜娜 / 上官婉儿）、少数民族姓名（买买提·阿凡提）、
`+` 标签邮箱、带端口 IP、园区式地址、缺省级前缀地址、掩码身份证、
多类同现长文本、嵌套 tool_call。共 21 个独立句法形态（数字归一后）。

#### `expect` / `expect_miss` / `expect: []` 的区分

| 字段 | 语义 | 计分影响 |
|---|---|---|
| `expect: [...]` | 这句话里**应该**被检出这些 PII | 漏了算 FN，多检出算 FP |
| `expect: []` | 这句话里**确实没有** PII | 任何检出都是 FP |
| `expect_miss: [...]` | **已知弱点**：值真实存在、当前引擎检不出，我们知情并接受 | 该值被检出**不算 FP**（豁免）；同时计入「弱点台账」与**悲观口径** FN |

三者必须严格区分。早期版本把「已知弱点」写成 `expect: []`，等于**把缺陷写成了期望**
——既放弃度量，又把真实 PII 判成误报（`idcard_mask_adv-028` 就曾把同句的真名 `张三`
误判为 FP）。

#### 三语料均要求「正文唯一」

`generate*.py` 逐条重试直到正文唯一；`validate.py` 把重复组视为**硬错误**。
原因：同一句话换几个数字就产出 N 条，指标会被系统性做虚。三个语料当前
**0 重复组**。

## 评估

### 0. 语料自检（不启网关，必跑）

```bash
python3 validate.py --all          # 三语料全查
python3 validate.py --cases fixtures/cases_adversarial.jsonl
```

校验：type 合法、offset 与 value 自洽、`expect_miss` 的三元组完整
（type 已知 / value 非空且确实出现在正文 / reason 非空）、**无重复组**。

### 1. L1 检测器级（runner.py）

```bash
python3 runner.py --endpoint http://127.0.0.1:8413/_api/detect --cases fixtures/cases.jsonl
python3 runner.py --endpoint http://127.0.0.1:8413/_api/detect --cases fixtures/cases_en.jsonl
python3 runner.py --selftest      # 不启网关
```

> `--endpoint` **必须给完整 URL（带 `/_api/detect`）**。只给 `http://127.0.0.1:8413`
> 会 POST 到根路径拿到 404，全部计为 FN、指标全 0 —— 不报错，静默归零。

产物：`reports/phase0_<engine>_<corpus>_<ts>.md` + `.json`
（per_type / per_subset / fp_items / fn_items）。

> **文件名带语料名**：时间戳只有秒级精度，不带语料名时连续跑两个语料会**静默互相覆盖**
> （`cases.jsonl` 的报告曾被 `cases_en.jsonl` 的报告盖掉）。

### 2. 真对抗（bench_runner_adversarial.py）

```bash
# 摘要（stdout JSON，并落盘 reports/adversarial_<ts>.json）
python3 bench_runner_adversarial.py --endpoint http://127.0.0.1:8413/v1/privacy/redact

# 完整 markdown 报告（并落盘 .md + .json）
python3 bench_runner_adversarial.py --endpoint http://127.0.0.1:8413/v1/privacy/redact --report
```

产出的报告**同时给严格口径与悲观口径两个 F1**：

- **严格**：`expect_miss` 豁免（只对「非已知弱点」计分）
- **悲观**：把每条 `expect_miss` 都计入 FN

只报严格口径会显得比实际强，两个都报才是诚实口径。

### 3. L2 泄漏级（carriers.py）

```bash
python3 carriers.py --base-url http://127.0.0.1:8413
python3 carriers.py --selftest --limit 40
```

4 载体：`flat` / `multimodal` / `tool_call` / `tool_call_nested`。
**regressions vs flat = 0、round-trip 100%** 是 v1 守门。

口径是**泄漏级**（脱敏后载荷里 PII 原串是否消失），不是检测器 span 级。
这个口径比 L1 更敏感：它能抓到「同一请求里同一个值出现在两个字段、第二个字段没脱敏」
这类 span 级看指标的盲区（2026-09-15 的 P0-12 泄漏就是这样被定位的）。

#### 「被脱敏」≠「可逆」：按设计不可逆的豁免

`zh_bank_card` 在 `strategy=placeholder` 下的命运是 **mask**（保留后 4 位、其余填 `*`），
**按设计不可逆**——掩码值本来就不该被还原。评估器因此对每个实体类型
**实测一次**可逆性（脱敏 → 还原），不可逆类型豁免往返断言并单列入台账：

```
[carriers] 可逆性探测：{'zh_person_name': None, 'zh_phone': True, 'zh_id_card': True,
                        'zh_bank_card': False, 'zh_address': True, 'email': True}
[carriers] 按设计不可逆（豁免往返断言）：['zh_bank_card']
```

- 不硬编码类型清单（那会把网关策略实现细节抄进基准，必然漂移）；
- 探测失败的走**保守**路径（不豁免，照旧断言往返）；
- 豁免的值进报告台账，**不静默丢弃**。

> 早期版本把「值从载荷里消失」与「值可被还原」当成同一件事，于是任何含银行卡的
> 样本都被判为往返失败：全量 240 条里 **144 条假警报**（36 × 4 载体）。
> 因为只跑 `--limit 60` 子集而长期未暴露 —— **守门必须跑全量**。

## 当前基线（2026-09-15，`engine=regex`）

| 语料 | 条数 | P | R | F1 | 备注 |
|---|---|---|---|---|---|
| `cases.jsonl` | 240 | 1.0000 | 1.0000 | 1.0000 | 自作者语料，非对外结论 |
| `cases_en.jsonl` | 180 | 1.0000 | 1.0000 | 1.0000 | |
| `cases_adversarial.jsonl`（严格） | 28 | 1.0000 | 0.6216 | **0.7667** | 诚实口径 |
| `cases_adversarial.jsonl`（悲观） | 28 | 1.0000 | 0.5610 | **0.7188** | |
| carriers（4 载体，**全量 240**） | 240×4 | — | — | — | **regressions=0, round-trip 240/240** ✅ |

已知弱点台账 4 条、已恢复 0（3 条 `+86-…` 手机号形态 + 1 条掩码身份证）。

> **示例命令一律不带 `--limit`。** 子集采样会整类地掩盖缺陷：
> `--limit 60` 恰好只取到 `person_name` + `phone`，银行卡一条都不进样本。

## 任务清单

- [x] 0.1 骨架（loader + fixtures 占位）
- [x] 0.2 注入 240 合成样本（8 子集 × 30）
- [x] 0.3 跑分脚本（runner.py / carriers.py / selftest）
- [x] 0.4 第三方标注硬约束（SPEC §4.3.4，待 v1 截止前补 human-annotated + third-party 集）
- [x] 0.6 真对抗语料 + 评估器（`expect_miss` 台账 + 双口径）
- [x] 0.7 重复语料硬门禁（三语料 0 重复组）
- [x] 0.8 L2 载体对等性达标（全量 240 条 regressions=0、往返 240/240）
- [x] 0.9 按设计不可逆显式入账（可逆性实测探测 + 台账）
- [x] 1.0 报告产物可区分（文件名带语料名；三个评估器统一 `.md` + `.json`）
- [ ] 0.5 接入 PII Engineer 模型做基准（暂缓，当前真对抗召回率 0.5610~0.6216，远未到 85%）
