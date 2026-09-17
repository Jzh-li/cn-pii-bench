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
├─ bench_runner_adversarial.py  真对抗评估（expect_miss 豁免 + 弱点台账 + span/strict 双口径 + 悲观口径）
├─ analyze_corpus.py            语料质量诊断（`forms` 形态缺口聚合 / `sanity` GT 结构自洽核验）
├─ carriers.py                  L2 泄漏级评估（4 载体对等性 + 往返 + 守门模式 + selftest）
├─ generate.py                  240 条合成中文语料生成器（身份证校验位 / 手机号段 / Luhn / 复姓）
├─ generate_en.py               180 条英文语料生成器
├─ generate_adversarial.py      真对抗语料生成器（PII 登记表 + 自动补标 + 自检 + `--matrix` 形态矩阵）
├─ validate.py                  三语料结构自检（重复即硬错）
├─ fixtures/
│   ├─ cases.jsonl              240 条合成中文样本（8 子集 × 30）
│   ├─ cases_en.jsonl           180 条英文样本
│   ├─ cases_adversarial.jsonl  28 条真对抗样本 / 48 个 GT 实体
│   ├─ cases_adversarial_ext.jsonl  318 条（28 手写 + 290 形态矩阵）/ 338 个 GT 实体
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

### 真对抗 `cases_adversarial.jsonl`（28 条 / 48 个 GT 实体）

手工构造，覆盖合成语料无法生成的**真实形态**：
复姓人名（欧阳娜娜 / 上官婉儿）、少数民族姓名（买买提·阿凡提）、
`+` 标签邮箱、带端口 IP、园区式地址、缺省级前缀地址、掩码身份证、
多类同现长文本、嵌套 tool_call。共 21 个独立句法形态（数字归一后）。

#### 语料修订（2026-09-16）：从「惩罚正确答案」到自守

上一版语料有一条**结构性缺陷**：生成器的构造器把装饰上下文用的真 PII
**硬编码进 f-string**，而 `expect` 只登记「被测的那一个实体」。
典型如：

```python
text = f"联系销售李娜（手机：📱{phone}），工作日联系。📞"
#            ^^^^ 是真姓名，但 expect 里只有 zh_phone
```

后果是检测器**正确认出正文里的真 PII 反而被记成 FP**。实测这 3 个构造器污染了
8 个 case，贡献了融合评估全部 11 个「误报」中的 8 个 —— 换句话说，
**那批「误报」没有一个是真误报**。类似地，地址标注粒度在同一文件里自相矛盾
（`addr_noprov` / `mixed_tool` 标完整跨度，`addr_abbrev` / `mixed_real` 只标一半）。

现用两条约定把它变成结构上不可能：

**约定 3 — 模板禁止内联真 PII 字面量。** 所有会出现在文本里的 PII 统一登记进
`PII_REGISTRY`；构造完 case 后由 `finalize_case()` 扫描全文、**自动补标**任何
漏标的已登记值。只要值进了登记表，就不可能不被标注。
`self_check()` 双向校验：登记了必须被标注，标注了必须已登记。

**约定 4 — 地址标注 = 能独立定位到具体地点的最长连续串。**
登记表里只登记最长形式；`self_check()` 断言同类型跨度**不得重叠**，
从机制上禁止「短跨度 + 长跨度同时存在」。

**约定 5 — 假号码必须自洽。** 语料里的手机号 / 身份证 / 银行卡是假的，
但必须通过各自的真实校验（号段白名单 / GB11643 校验位 / Luhn）。
`self_check()` 的 ㋒ 断言登记表里每个值都自洽。

理由是检测器对这些类型**都带校验**：用一个不过校验的号码，测到的不是
「能不能找到 PII」，而是「会不会对非法号码误报」——与语料意图正好相反。

这类缺陷的隐蔽之处在于**它长得像一个真实的形态缺口**。实测例子：
语料写 `6222021234567890123`（Luhn sum=75，不过），报告显示
「`zh_bank_card` 召回 0%」，读起来像「检测器根本没有银行卡规则」；
实际是检测器**正确地**拒掉了一个非法号 —— 而那个号是语料自己造错的
（`zh_bank_card` 的规则一直在：13-19 位 + Luhn 分派）。
同批还有 `v[:17] + "x"` 造出的非法身份证号（该号校验位本来是 `1`）。
两者合计 30 个 GT、占矩阵语料 10.3%，**修正后矩阵召回从 0.3639 升到 0.4231**。

> 残留缺口（已知且接受）：若有人内联一个**新**字面量且**完全不标**，两条离线检查
> 都拦不住（值没进登记表）。这类只能靠外部审计兜底 —— `audit_corpus.py`
> 用两引擎实测反查漏标。

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
原因：同一句话换几个数字就产出 N 条，指标会被系统性做虚。各语料当前
**0 重复组**（含 318 条矩阵语料）。

### 真对抗矩阵 `cases_adversarial_ext.jsonl`（318 条 / 338 个 GT 实体）

28 条手写 + 290 条矩阵（`--matrix` 生成）。矩阵是「**表面形式 × 载体**」的组合：

- **表面形式（`SURFACE_FORMS`）**：同一个值的不同书写形态 —— 裸写 / 分隔符 /
  国际区号 / 全角 / 分组 / 大小写。真实文本里漏检的主要来源。
- **载体（`CARRIERS`）**：值出现的上下文骨架（聊天 / 日志 / 工单 / 括号 / 引号 /
  签名 / 换行 …），本身**不含任何 PII** —— 否则会引入没登记的假阳性源，
  而 precision 门禁容不下自造的噪声。

不做成继续手写的理由：手写把「典型形态」覆盖完后就是重复劳动，而漏检来自
**形态 × 上下文的组合**，组合才是该被穷举的维度。

三条与既有约定同构的纪律：

1. 构造器只负责「把值放进文本」，**不手写 expect** —— 标注由 `finalize_case()`
   扫全文完成，新增形态时漏标在结构上不可能（约定 3）。
2. **只登记「属于 PII 自身书写形态」的变体，不登记装饰性标点。**
   `138-0013-8000` 的分隔符在值**内部** → 登记整串；`“13800138000”` 的引号在值
   **外部** → 不登记，让自动补标只框住号码本身。搞反了就是在**惩罚正确答案**：
   登记带引号的版本，会把「正确只框住号码」的检测器判成跨度错误。
3. **形态级源值**（`SURFACE_FORMS` 的第四项）：有些形态只对**特定构造的值**成立。
   `lower_x`（小写校验位）只对校验位本来就是 `X` 的号成立 —— 拿任意号码把末位
   替换成 `x`，产物是一个**非法**身份证号（见约定 5）。
   专用值下沉到形态级，避免 bare/grouped 拿它再跑一遍白增重复权重。

> `--matrix` **默认关闭**：不带它时输出与 28 条时代**逐字节一致**
> （md5 `409c6ca33536f5c12f1e1fe87228b1a9`），保证既有基线不被扰动 ——
> 扩充语料不该顺带改写历史。

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
# 摘要（stdout JSON，并落盘 reports/<语料名>_<ts>.json）
python3 bench_runner_adversarial.py --endpoint http://127.0.0.1:8413/v1/privacy/redact

# 指定语料（矩阵语料：318 条）
python3 bench_runner_adversarial.py --endpoint http://127.0.0.1:8413/v1/privacy/redact \
    --cases fixtures/cases_adversarial_ext.jsonl --report

# 完整 markdown 报告（并落盘 .md + .json）
python3 bench_runner_adversarial.py --endpoint http://127.0.0.1:8413/v1/privacy/redact --report

# 只跑一个口径（默认 both）
python3 bench_runner_adversarial.py --endpoint ... --match span
python3 bench_runner_adversarial.py --endpoint ... --match strict --gate-on strict
```

> **报告文件名带语料名**（2026-09-16 修）：`reports/cases_adversarial_ext_<ts>.md`。
> 原先只写 `adversarial_<ts>.md`，而「先跑 28 条基线、再跑 318 条矩阵」是最常见的用法 ——
> 两个语料的报告会在同一目录里**静默互相覆盖**（`runner.py` 早期正是这样丢过报告）。

产出的报告**同时给三个口径的 F1**（`--match both` 为默认）：

| 口径 | 匹配规则 | 衡量什么 | 用途 |
|---|---|---|---|
| **`span`**（主口径） | 同类型且值**互为子串** | 有没有找到 | 默认门禁（`--gate-on span`） |
| `strict`（最保守下界） | 值**逐字相等** | 跨度和语料是否逐字一致 | 对标历史基线、跨版本可比 |
| 悲观 | 在门禁口径上把每条 `expect_miss` 计入 FN | 已知缺口全算漏 | 让缺口在标题数字外可见 |

**为什么必须双口径**：`strict` 对 `zh_address` 这类**可变长**实体过苛 ——
文本 `…地址北京市海淀区中关村南大街 5 号…`、GT 为含门牌全串、检出为
`北京市海淀区中关村南大街`（少了门牌）时，被记 **1 FP + 1 FN**：一次「找到了但略短」
却被罚两次。反向（检测器标得更全）同样被罚。于是 `strict` 实际在测
「跨度是否与语料作者的习惯一致」，而不是「有没有找到 PII」。

实测两口径 F1 相差约 **0.06 ~ 0.09**，差距 100% 来自地址跨度 —— **这是口径差异，
不是能力差异**，所以两个数必须并排出现，只报其中一个都是误导。

#### 实测基线（2026-09-16，28 条 / 48 个 GT 实体）

| 检测配置 | `span` F1（主） | `strict` F1（下界） |
|---|---|---|
| 网关 regex 引擎（现役） | 0.6479 | 0.5915 |
| sidecar NER（pii-engineer） | 0.9091 | 0.8409 |
| **regex ∪ NER（`union`）** | **0.9787** | **0.8842** |

> ⚠️ 上一版 README 里「regex 严格口径 F1 = 0.7667」是在**旧语料**（37 个 GT、
> 大量真 PII 漏标）上算出的，与本表不可直接比较。见下节「语料修订」。

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

## 检测质量线：形态覆盖缺口（2026-09-16 实测）

矩阵语料跑完后，用 `analyze_corpus.py` 把分数**拆到形态维度**上看：

```bash
python3 analyze_corpus.py forms  --endpoint http://127.0.0.1:8413/v1/privacy/redact \
    --cases fixtures/cases_adversarial_ext.jsonl
python3 analyze_corpus.py sanity --endpoint http://127.0.0.1:8413/v1/privacy/redact \
    --cases fixtures/cases_adversarial_ext.jsonl
```

### 结论一：载体完全无影响，形态是唯一变量

10 个载体的漏检率**一模一样**（0.586），一个不差；而形态之间是 `0.000` 与 `1.000`
的两极。含义：检测器对 PII **周围的装饰**（冒号 / 引号 / 括号 / 换行 / 日志骨架）
毫不敏感，对 PII **自身的书写形态**极度敏感。

因此载体维度给的是**覆盖证据**（证明装饰不影响判定），不是区分度；
后续若要扩语料，扩形态比扩载体有效。

### 结论二：值内部出现分隔符或非 ASCII 数字 → 100% 漏检

| 类型 / 形态 | 示例 | 漏检率 |
|---|---|---|
| `zh_phone/bare` | `13800138000` | 0.000 |
| `zh_phone/dashed` | `138-0013-8000` | **1.000** |
| `zh_phone/spaced` | `138 0013 8000` | **1.000** |
| `zh_phone/dotted` | `138.0013.8000` | **1.000** |
| `zh_phone/country` | `+86-138-0013-8000` | **1.000** |
| `zh_phone/fullwidth` | `１３８００１３８０００` | **1.000** |
| `zh_id_card/bare` | `110101199003078531` | 0.000 |
| `zh_id_card/grouped` | `110101 1990 0307 8531` | **1.000** |
| `zh_id_card/lower_x` | `11010119900307002x` | 0.000 |
| `zh_bank_card/bare` | `6222021234567890128` | 0.000 |
| `zh_bank_card/grouped` | `6222 0212 3456 7890 128` | **1.000** |
| `email`（plain / upper） | `ZHANG.SAN@EXAMPLE.COM` | 0.000 |
| `ip_address`（bare / with_port） | `192.168.1.100:8443` | 0.000 |

规律一句话：**值内部有分隔符 → 漏；值外部有装饰 → 命中。**

根因：所有数字类规则都是「连续数字串」正则 —— `1[3-9][0-9]{9}`、
`[1-9][0-9]{5}(?:19|20)…[0-9Xx]`、`[0-9]{13,19}`（手机 / 身份证 / 银行卡），
没有任何分隔符容忍；全角数字更是在字节层面就不匹配。

### 结论三：哪些**不是**缺口（避免按错误的缺口去优化）

- `lower_x`：检测器已把末位小写 `x` 归一后校验，**0 漏检**，无需处理；
- `zh_bank_card`：规则一直在（13-19 位 + Luhn 分派），此前「召回 0%」是
  **语料造假号**造成的（见约定 5），不是缺功能；
- 全角之外，`fullwidth` 形态在检测前做一次 NFKC 归一即可覆盖，
  与分隔符是同一类问题（归一化预处理）。

> 复现口径：以上为 `engine=regex` + `fail_closed=true` 的现役配置，
> 语料为 `cases_adversarial_ext.jsonl`（矩阵部分 290 个 GT）。

## 当前基线（2026-09-16，语料修订后）

`engine=regex`（网关现役配置）：

| 语料 | 条数 | 口径 | P | R | F1 | 备注 |
|---|---|---|---|---|---|---|
| `cases.jsonl` | 240 | strict | 1.0000 | 1.0000 | 1.0000 | 自作者语料，非对外结论 |
| `cases_en.jsonl` | 180 | strict | 1.0000 | 1.0000 | 1.0000 | |
| `cases_adversarial.jsonl` | 28 | **span（主）** | 1.0000 | 0.4792 | **0.6479** | 诚实口径 |
| `cases_adversarial.jsonl` | 28 | strict（下界） | 0.9130 | 0.4375 | **0.5915** | 对标历史基线用 |
| `cases_adversarial.jsonl` | 28 | 悲观 | 1.0000 | 0.4694 | **0.6389** | |
| `cases_adversarial_ext.jsonl` | 318 | **span（主）** | 1.0000 | 0.4231 | **0.5946** | 含 290 条形态矩阵 |
| `cases_adversarial_ext.jsonl` | 318 | strict（下界） | 0.9860 | 0.4172 | **0.5863** | |
| `cases_adversarial_ext.jsonl` | 318 | 悲观 | 1.0000 | 0.4218 | **0.5934** | |
| carriers（4 载体，**全量 240**） | 240×4 | — | — | — | — | **regressions=0, round-trip 240/240** ✅ |

> 矩阵语料的 F1 **低于** 28 条语料（0.5946 vs 0.6479），这是**预期的**：
> 它专门采样了既有检测器最弱的形态（分隔符），把原先被「裸写样本」平均掉的
> 缺口暴露出来。两份语料**并存、不替代** —— 28 条用于跨版本可比，
> 318 条用于定位缺口。**不能用矩阵语料的数字去宣布性能下降。**

> ⚠️ 矩阵语料**不进门禁，但已进 CI**（2026-09-17）：它的召回（span 0.4231）必然低于阈值，
> 直接参与判定会让 CI 常红。所以它在 LLMate-Gate 的 `bench-gate` job 里以
> **「形态诊断（非门禁）」** 这一步运行 —— 带 `--report` 但**不带任何阈值**，
> 报告随 artifact 上传。目的是让「形态覆盖缺口」每轮 CI 都可见
> （此前该数字只存在于本地报告，CI 上完全看不到），
> 等形态容忍（分隔符 / 全角归一化）落地后再决定是否升为门禁。

> ⚠️ **门禁阈值已随之调整**：`--min-recall` 由 `0.60` 降为 **`0.46`**，且 CI 里显式写了
> `--gate-on span`（此前靠脚本默认值）。原因是门禁锚定的 28 条基线语料在 2026-09-16
> 修订过（GT 37 → 48），基线本身从 span R=0.6216 变为 0.4792 —— 阈值只是**跟随基线**，
> 语义仍是「防回归」。详见 LLMate-Gate 的 `.github/workflows/ci.yml` 该步骤注释。

> ⚠️ **与 2026-09-15 的 0.7667 不可直接比较。** 旧语料只有 37 个 GT 实体、且存在
> 「真 PII 漏标 + 地址粒度不一致」（见「语料修订」节），那个数字同时高估了精确率
> （把检出真 PII 记成 FP 后又漏标）与召回率（漏标即不产生 FN）。修订后 GT 增至
> 48 个，基线随之下降 —— **这是度量变准了，不是引擎变差了**。

已知弱点台账 **1 条**、已恢复 0（掩码身份证 `110101********8531`）。
原先登记为弱点的 3 条 `+86-…` 手机号已被 NER 侧车稳定检出，故**提为正式 expect**。

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
- [x] 1.1 语料标注自守（PII 登记表 + 自动补标 + 双口径 `--match`，见「语料修订」节）
- [ ] 0.5 接入 PII Engineer 模型做基准 → **已实测**（sidecar span F1 0.9091 / strict 0.8409；
      与 regex 融合后 span 0.9787 / strict 0.8842）。剩余阻塞项是**延迟**（p50 3.4s vs
      网关 500ms 硬超时），不是质量
- [x] 1.2 扩充真对抗语料 28 → 300+（当前每条实体权重 1/48，单条波动即可影响 2 个点）
      → **已完成 28 → 318**（`--matrix` 表面形式 × 载体；既有 28 条逐字节不变）
- [x] 1.4 语料值结构自洽硬门（约定 5：号段 / GB11643 校验位 / Luhn，`self_check()` ㋒）
- [x] 1.5 形态缺口定位工具（`analyze_corpus.py forms|sanity`）
- [ ] 1.6 **检测器形态容忍**（分隔符 / 全角 → 归一化预处理）
      缺口已定位：170 个 GT 全漏在 `dashed` / `spaced` / `dotted` / `country` /
      `fullwidth` / `grouped` 六种形态上，`bare` 类 100% 命中（见「形态覆盖缺口」节）。
      **待拍板后再动检测器代码。**
- [ ] 1.3 三方独立标注集（去自证偏差）
- [x] 1.7 门禁口径对齐（2026-09-17）—— LLMate-Gate 把 `bench` gitlink 前移到 dev HEAD
      （`c8db66d → b9533c1`，启用修订后语料 GT 48 + 本矩阵语料），并把
      `--min-recall` 0.60 → **0.46**、显式加 `--gate-on span`；本矩阵语料以
      **「形态诊断（非门禁）」** 步骤进 CI。前移前提已实测：`err_count=0`、
      评估器脚本在 `c8db66d..b9533c1` 之间**零改动**（只换语料刻度）、
      L2 守门 `regressions=0`。
