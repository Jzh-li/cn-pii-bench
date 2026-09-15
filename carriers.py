#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench/carriers.py —— 结构化载体对等性基准（L2 泄漏口径）。

迁移来源：privaite-bench 的 `bench_structured.py`（MIT 思路借鉴，实现重写）。

为什么需要这份基准
------------------
`runner.py` 是 **L1 检测器口径**：把正文丢给 /_api/detect，比对检测器吐出的 span
四元组。它能拿 F1=1.0，但**完全不能证明 PII 没漏出去**——如果替换层、递归扫描层
或还原层有洞，检测器依然可以全对。

`carriers.py` 是 **L2 泄漏口径**：把同一份语料重新包裹成 4 种 OpenAI 消息载体，
走网关 `/v1/privacy/redact`（键保留、任意嵌套 JSON 递归脱敏），然后直接问一句——
**原文里的 PII 字符串，还出现在脱敏后的载荷里吗？**

两个指标（对齐 privaite-bench）：

  parity       flat 基线能脱掉的每条 PII，每种结构化载体都必须同样脱掉。
               「flat 抓到、载体漏了」= 一条 regression。目标：**0**。
  round-trip   用 /v1/privacy/restore 还原后，**可逆命运**的每条 PII 必须无损回来。

               注意「被脱敏」≠「可逆」：`zh_bank_card` 在 placeholder 策略下是
               mask（保留后 4 位），按设计不可逆。评估器对每个实体类型实测一次
               可逆性（probe_reversible），不可逆类型豁免往返断言并单独入台账——
               不硬编码类型清单，也不静默丢弃。

载体（CARRIERS）：
  flat               {"role":"user","content": text}
  multimodal         {"role":"user","content":[{"type":"text","text": text}]}
  tool_call          tool_calls[0].function.arguments = {"note": text}
  tool_call_nested   tool_calls[0].function.arguments = {"user":{"profile":{"bio": text}}, "tags":[text]}

自检（--selftest）
------------------
不启网关即可验证「评估器本身是好的」：起一个进程内假脱敏服务，跑两遍——
  1) 正确实现     → 期望 0 regression、round-trip 全通过
  2) 故意漏实现   → 只脱敏 content、不脱敏 tool_calls → 期望 **必须报出 regression**
第 2 遍是关键：如果评估器在坏实现上也报 0，说明这个基准本身是瞎的。

用法：
    python3 bench/carriers.py --base-url http://127.0.0.1:8401
    python3 bench/carriers.py --selftest          # 无需网关
    python3 bench/carriers.py --limit 30          # 只跑前 N 条（快速冒烟）
"""
from __future__ import annotations

import argparse
import http.server
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runner import DEFAULT_CASES, load_cases  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "reports"

CARRIERS = ["flat", "multimodal", "tool_call", "tool_call_nested"]

# tool_call 载体里用的函数名（仅形状占位，不参与评分）
_FN_FLAT = "save_note"
_FN_NESTED = "save_profile"

# 绕开环境里的 http_proxy：评测目标是 127.0.0.1，走代理会被劫持/挂起
# （urlopen 默认读 http_proxy；本地回环也不例外）。runner.py 与
# bench_runner_adversarial.py 同样处理，三处口径保持一致。
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# --------------------------------------------------------------------------
# 载体构造
# --------------------------------------------------------------------------

def wrap(case_id: str, text: str, carrier: str) -> list[dict]:
    """把扁平正文包裹成指定载体的 messages 数组。"""
    if carrier == "flat":
        return [{"role": "user", "content": text}]
    if carrier == "multimodal":
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]
    if carrier == "tool_call":
        args = json.dumps({"action": "note", "note": text}, ensure_ascii=False)
        return [{"role": "assistant", "tool_calls": [
            {"id": f"call_{case_id}", "type": "function",
             "function": {"name": _FN_FLAT, "arguments": args}}]}]
    if carrier == "tool_call_nested":
        args = json.dumps(
            {"user": {"profile": {"bio": text}}, "tags": [text]}, ensure_ascii=False)
        return [{"role": "assistant", "tool_calls": [
            {"id": f"call_{case_id}", "type": "function",
             "function": {"name": _FN_NESTED, "arguments": args}}]}]
    raise ValueError(f"unknown carrier: {carrier}")


def flatten(obj: Any) -> str:
    """把任意嵌套 JSON 拍平成单个字符串，用于泄漏判定。"""
    return json.dumps(obj, ensure_ascii=False)


def score_leaks(values: list[str], redacted_flat: str) -> tuple[list[str], list[str]]:
    """返回 (anonymized, leaked)。"""
    anonymized = [v for v in values if v not in redacted_flat]
    leaked = [v for v in values if v in redacted_flat]
    return anonymized, leaked


# --------------------------------------------------------------------------
# 网关调用
# --------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: float, token: str = "") -> tuple[dict, int, str]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body, int((time.perf_counter() - t0) * 1000), ""
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        return {}, int((time.perf_counter() - t0) * 1000), f"{type(e).__name__}: {e}"


def gateway_redact(base: str, messages: list[dict], timeout: float, token: str,
                   strategy: str) -> tuple[Any, str, int, str]:
    """POST /v1/privacy/redact → (redacted_json, request_id, latency_ms, error)。"""
    body, ms, err = _post(base.rstrip("/") + "/v1/privacy/redact",
                          {"json": messages, "strategy": strategy}, timeout, token)
    if err:
        return None, "", ms, err
    return body.get("json"), str(body.get("request_id", "")), ms, ""


def gateway_restore(base: str, redacted: Any, request_id: str, timeout: float,
                    token: str) -> tuple[Any, int, str]:
    """POST /v1/privacy/restore → (restored_json, latency_ms, error)。"""
    body, ms, err = _post(base.rstrip("/") + "/v1/privacy/restore",
                          {"json": redacted, "request_id": request_id}, timeout, token)
    if err:
        return None, ms, err
    return body.get("json"), ms, ""


# --------------------------------------------------------------------------
# 评估主循环
# --------------------------------------------------------------------------

def probe_reversible(typ: str, value: str,
                     redact: Callable[[list[dict]], tuple[Any, str, int, str]],
                     restore: Callable[[Any, str], tuple[Any, int, str]],
                     cache: dict[str, bool | None]) -> bool | None:
    """向网关**实测**「该实体类型在当前策略下是否可逆」，结果按类型缓存。

    为什么必须实测：`zh_bank_card` 在 `strategy=placeholder` 下走的是 **mask**
    （`replacer.go` 的 `fateFor`：保留后 4 位、其余填 `*`），这是**按设计不可逆**——
    掩码值本来就不该、也不可能被还原。

    早期评估器把「值从载荷里消失」（`anonymized`）与「值可被还原」当成同一件事
    （`rt = all(v in rflat for v in anon)`），于是任何含银行卡的样本都被判为往返失败：
    全量 240 条里 144 条误报（30 条 bank_card + 6 条含银行卡的 adversarial，×4 载体），
    而 `--limit 60` 的子集恰好只含 person_name/phone，所以一直没暴露。

    这里不硬编码「哪些类型不可逆」（那会把网关策略实现细节抄进基准、必然漂移），
    改为用**最小样本**探测一次：

    - 值没被脱掉 → 返回 None（不是可逆性问题，照旧断言往返）
    - 值被脱掉且能还原 → True（必须断言往返）
    - 值被脱掉但还原不回来 → False（mask/redact，豁免往返断言）
    - 探测本身失败 → None（**宁严勿松**：不豁免）

    探测结果按类型缓存，一次运行只需 ~8 次额外请求。
    """
    if typ in cache:
        return cache[typ]
    sample = f"这是我的{value}，请记录。"
    redacted, rid, _, err = redact([{"role": "user", "content": sample}])
    if err or not rid:
        cache[typ] = None
        return None
    if value in flatten(redacted):
        cache[typ] = None          # 压根没脱掉，不属于可逆性范畴
        return None
    restored, _, rerr = restore(redacted, rid)
    if rerr:
        cache[typ] = None
        return None
    cache[typ] = value in flatten(restored)
    return cache[typ]


def eval_case(case_id: str, pairs: list[tuple[str, str]], text: str,
              redact: Callable[[list[dict]], tuple[Any, str, int, str]],
              restore: Callable[[Any, str], tuple[Any, int, str]],
              reversibility: dict[str, bool | None] | None = None) -> dict:
    reversibility = {} if reversibility is None else reversibility
    values = [v for _, v in pairs]

    carriers: dict[str, dict] = {}
    for carrier in CARRIERS:
        messages = wrap(case_id, text, carrier)
        redacted, rid, ms, err = redact(messages)
        if err:
            carriers[carrier] = {"error": err, "anonymized": [], "leaked": values,
                                 "roundtrip_ok": None, "latency_ms": ms,
                                 "by_design_irreversible": []}
            continue
        anon, leaked = score_leaks(values, flatten(redacted))

        # 只有「可逆类型」才断言往返；mask/redact 类型按设计不可逆，
        # 单独记进 by_design_irreversible 台账（可见、不静默丢弃）。
        for typ, val in pairs:
            probe_reversible(typ, val, redact, restore, reversibility)
        anon_set = set(anon)
        by_design = [v for t, v in pairs
                     if reversibility.get(t) is False and v in anon_set]
        rt_values = [v for t, v in pairs
                     if reversibility.get(t) is not False and v in anon_set]

        rt: bool | None = None
        if rid:
            restored, _, rerr = restore(redacted, rid)
            if rerr:
                rt = False
            else:
                rflat = flatten(restored)
                rt = all(v in rflat for v in rt_values)
        carriers[carrier] = {"anonymized": anon, "leaked": leaked,
                             "roundtrip_ok": rt, "latency_ms": ms,
                             "request_id": rid,
                             "by_design_irreversible": by_design,
                             "roundtrip_checked_values": rt_values}
    return {"id": case_id, "carriers": carriers}


def summarize(results: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    flat_total = sum(len(r["carriers"]["flat"]["anonymized"]) for r in results)
    for carrier in CARRIERS:
        anonymized = sum(len(r["carriers"][carrier]["anonymized"]) for r in results)
        leaked = sum(len(r["carriers"][carrier]["leaked"]) for r in results)
        regressions: list[list[str]] = []
        for r in results:
            c = r["carriers"][carrier]
            flat_ok = set(r["carriers"]["flat"]["anonymized"])
            for pii in flat_ok & set(c["leaked"]):
                regressions.append([r["id"], pii])
        rt = [r["carriers"][carrier]["roundtrip_ok"] for r in results]
        rt_checked = [x for x in rt if x is not None]
        by_design: list[list[str]] = []
        for r in results:
            for v in r["carriers"][carrier].get("by_design_irreversible", []):
                by_design.append([r["id"], v])
        # 请求错误必须单独计数。否则「网关不可达」会让所有载体都 0 regression、
        # 0 往返失败 —— 门禁打印 PARITY OK 并退出 0，什么都没测到却说通过。
        # 这是典型的「不会红的门禁」，比没有门禁更糟：它给出虚假保证。
        errors = [[r["id"], r["carriers"][carrier]["error"]]
                  for r in results if "error" in r["carriers"][carrier]]
        summary[carrier] = {
            "anonymized": anonymized,
            "leaked": leaked,
            "regressions": regressions,
            "roundtrip_checked": len(rt_checked),
            "roundtrip_failures": sum(1 for x in rt_checked if x is False),
            "by_design_irreversible": by_design,
            "errors": errors,
        }
    summary["_flat_baseline"] = {"anonymized": flat_total}
    return summary


def render_report(summary: dict, results: list[dict], endpoint: str, cases_path: Path,
                  engine: str) -> str:
    flat_total = summary["_flat_baseline"]["anonymized"]
    n = len(results)
    lines = [
        f"# cn-pii-bench · 结构化载体对等性报告 · 引擎 `{engine}`",
        "",
        f"- 端点：`{endpoint}`",
        f"- 语料：`{cases_path}`（{n} 条）",
        f"- 评估时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- 口径：**L2 泄漏级**（脱敏后载荷中 PII 原串是否消失），非检测器 span 级",
        f"- flat 基线脱敏条数：**{flat_total}**",
        "",
        "| 载体 | 已脱敏 | 泄漏 | 相对 flat 的 regression | 往返通过 | 请求错误 |",
        "|---|---|---|---|---|---|",
    ]
    total_reg = 0
    total_err = 0
    for carrier in CARRIERS:
        s = summary[carrier]
        total_reg += len(s["regressions"])
        total_err += len(s["errors"])
        flag = "OK" if not s["regressions"] else f"**{len(s['regressions'])} REGRESSIONS**"
        rt_txt = (f"{s['roundtrip_checked'] - s['roundtrip_failures']}/{s['roundtrip_checked']}"
                  if s["roundtrip_checked"] else "—")
        err_txt = "0" if not s["errors"] else f"**{len(s['errors'])} ERROR**"
        lines.append(
            f"| `{carrier}` | {s['anonymized']} | {s['leaked']} | {flag} | {rt_txt} | {err_txt} |")

    total_rt_fail = sum(summary[c]["roundtrip_failures"] for c in CARRIERS)
    lines += ["", "## 结论", ""]
    if total_err:
        # 先把「没测到」和「测了不达标」分开报。少了这一支，网关不可达会渲染成
        # PARITY OK —— 一个永远不会红的门禁比没有门禁更糟，因为它给出虚假保证。
        lines.append(f"❌ **无法判定（{total_err} 条请求错误）**："
                     f"端点不可达或返回异常，本次**没有测到**任何结论。")
        lines.append("")
        lines.append("| 载体 | Case | 错误 |")
        lines.append("|---|---|---|")
        for carrier in CARRIERS:
            for cid, err in summary[carrier]["errors"][:5]:
                lines.append(f"| `{carrier}` | `{cid}` | {err} |")
    elif total_reg == 0 and total_rt_fail == 0:
        lines.append(
            f"✅ **PARITY OK**：{len(CARRIERS) - 1} 种结构化载体相对 flat 基线 "
            f"0 regression，往返 0 失败。")
    else:
        lines.append(f"❌ **PARITY FAILED**：{total_reg} regression / "
                     f"{total_rt_fail} 往返失败。")
        for carrier in CARRIERS:
            for cid, pii in summary[carrier]["regressions"]:
                lines.append(f"- regression `[{carrier}]` `{cid}`：{pii!r}")
        for carrier in CARRIERS:
            if summary[carrier]["roundtrip_failures"]:
                lines.append(f"- 往返失败 `{carrier}`："
                             f"{summary[carrier]['roundtrip_failures']} 条")

    # 按设计不可逆的实体：掩码/抹除类型本就还原不回来，**不计入往返失败**，
    # 但也不静默丢弃 —— 在这里与「评测器不确定的类型」分开列账。
    n_by_design = sum(len(summary[c]["by_design_irreversible"]) for c in CARRIERS)
    if n_by_design:
        flat_by_design = summary["flat"]["by_design_irreversible"]
        lines += [
            "",
            f"## 按设计不可逆（豁免往返断言）：{len(flat_by_design)} 条 · flat 载体",
            "",
            "mask / redact 命运的值**本就不该被还原**（掩码保后 4 位即其目的）。",
            "此处单列台账，不静默丢弃：",
            "",
            "| Case | 值 |",
            "|---|---|",
        ]
        seen: set[tuple[str, str]] = set()
        for cid, val in flat_by_design:
            if (cid, val) in seen:
                continue
            seen.add((cid, val))
            lines.append(f"| `{cid}` | `{val}` |")
        lines.append("")
        lines.append(f"> 判定方式：对每个实体类型发起一次最小样本探测（"
                     f"脱敏 → 还原），还原不回来的类型即按设计不可逆。"
                     f"探测结果按类型缓存，不硬编码类型清单。")

    lines += ["", "---", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 自检：进程内假脱敏服务
# --------------------------------------------------------------------------

def _mk_fake_server(terms: dict[str, str], break_tool_calls: bool):
    """起一个线程内假脱敏服务。

    break_tool_calls=True 时故意只脱敏 content、跳过 tool_calls——
    用来验证评估器能真的抓到这个漏洞。
    """
    vault: dict[str, dict[str, str]] = {}
    seq = [0]

    def scrub(obj: Any, path: tuple, ph: dict[str, str]) -> Any:
        if isinstance(obj, dict):
            return {k: scrub(v, path + (k,), ph) for k, v in obj.items()}
        if isinstance(obj, list):
            return [scrub(v, path, ph) for v in obj]
        if isinstance(obj, str):
            if break_tool_calls and "tool_calls" in path:
                return obj
            for term in sorted(terms, key=len, reverse=True):
                if term in obj:
                    ph[ph.get(term) or f"<{terms[term]}_{len(ph)}>"] = term
                    ph.setdefault(term, f"<{terms[term]}_{len(ph)}>")
                    obj = obj.replace(term, ph[term])
            return obj
        return obj

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # 静音
            pass

        def _send(self, obj: dict, code: int = 200):
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path.endswith("/redact"):
                seq[0] += 1
                rid = f"selftest-{seq[0]}"
                ph: dict[str, str] = {}
                out = scrub(req.get("json"), (), ph)
                vault[rid] = {v: k for k, v in ph.items() if not k.startswith("<")}
                # 上面 scrub 的 ph 双向混在一起，重建成 占位符->原文
                vault[rid] = _rebuild(ph)
                self._send({"json": out, "request_id": rid, "changed": bool(ph)})
            elif self.path.endswith("/restore"):
                rid = req.get("request_id", "")
                m = vault.get(rid, {})
                out = _unmap(req.get("json"), m)
                self._send({"json": out})
            else:
                self._send({"error": "not found"}, 404)

    def _rebuild(ph: dict[str, str]) -> dict[str, str]:
        # ph 里既有 term->placeholder 也有 placeholder->term，取前者反向
        out = {}
        for k, v in ph.items():
            if k in terms and v.startswith("<"):
                out[v] = k
        return out

    def _unmap(obj: Any, m: dict[str, str]) -> Any:
        if isinstance(obj, dict):
            return {k: _unmap(v, m) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_unmap(v, m) for v in obj]
        if isinstance(obj, str):
            for ph, orig in m.items():
                obj = obj.replace(ph, orig)
            return obj
        return obj

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def selftest(limit: int) -> int:
    cases = []
    for cid, _subset, expect, text, _misses in load_cases(DEFAULT_CASES):
        cases.append((cid, [(e.type, e.value) for e in expect], text))
        if len(cases) >= limit:
            break
    if not cases:
        print("FAIL: 语料为空", file=sys.stderr)
        return 1
    terms = {v: "PII" for _c, pairs, _t in cases for _typ, v in pairs}

    failures = 0

    # 1) 正确实现 —— 期望 0 regression、往返全通过
    srv = _mk_fake_server(terms, break_tool_calls=False)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    rev: dict[str, bool | None] = {}
    results = [eval_case(cid, pairs, text,
                         lambda m: gateway_redact(base, m, 10.0, "", "placeholder"),
                         lambda r, rid: gateway_restore(base, r, rid, 10.0, ""),
                         rev)
               for cid, pairs, text in cases]
    s = summarize(results)
    reg = sum(len(s[c]["regressions"]) for c in CARRIERS)
    rtf = sum(s[c]["roundtrip_failures"] for c in CARRIERS)
    print(f"[selftest] 正确实现：regression={reg} 往返失败={rtf}")
    if reg != 0 or rtf != 0:
        print("  FAIL 正确实现本应 0 regression / 0 往返失败 —— 评估器有 bug")
        failures += 1
    srv.shutdown()

    # 2) 故意漏实现（不脱敏 tool_calls）—— 期望必须报出 regression
    srv = _mk_fake_server(terms, break_tool_calls=True)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    rev2: dict[str, bool | None] = {}
    results = [eval_case(cid, pairs, text,
                         lambda m: gateway_redact(base, m, 10.0, "", "placeholder"),
                         lambda r, rid: gateway_restore(base, r, rid, 10.0, ""),
                         rev2)
               for cid, pairs, text in cases]
    s = summarize(results)
    bad = sum(len(s[c]["regressions"]) for c in ("tool_call", "tool_call_nested"))
    print(f"[selftest] 故意漏实现（跳过 tool_calls）：regression={bad}")
    if bad == 0:
        print("  FAIL 坏实现没被抓出来 —— 这个基准是瞎的，不能用于守门")
        failures += 1
    srv.shutdown()

    # 3) 端点不可达 —— 必须被判为「无法判定」而不是「0 regression 所以 OK」
    dead = "http://127.0.0.1:1"   # 1 号端口不会有服务
    rev3: dict[str, bool | None] = {}
    results = [eval_case(cid, pairs, text,
                         lambda m: gateway_redact(dead, m, 2.0, "", "placeholder"),
                         lambda r, rid: gateway_restore(dead, r, rid, 2.0, ""),
                         rev3)
               for cid, pairs, text in cases[:3]]
    s = summarize(results)
    errs = sum(len(s[c]["errors"]) for c in CARRIERS)
    regs = sum(len(s[c]["regressions"]) for c in CARRIERS)
    print(f"[selftest] 端点不可达：请求错误={errs} regression={regs}")
    if errs == 0:
        print("  FAIL 端点不可达却没记到错误 —— 门禁在网关挂掉时会误判为通过")
        failures += 1
    if regs != 0:
        print(f"  FAIL 端点不可达时不应该产出 regression（实得 {regs}）—— "
              f"错误应走独立的错误通道")
        failures += 1

    print("SELFTEST PASSED" if failures == 0 else "SELFTEST FAILED")
    return failures


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="结构化载体对等性基准（L2 泄漏口径）")
    p.add_argument("--base-url", default="http://127.0.0.1:8401", help="网关根地址")
    p.add_argument("--token", default="", help="网关 auth token（未开启鉴权可留空）")
    p.add_argument("--engine", default="regex", help="引擎名（写入报告）")
    p.add_argument("--strategy", default="placeholder", choices=["placeholder", "simulate"])
    p.add_argument("--cases", default=str(DEFAULT_CASES))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--gate", action="store_true",
                   help="守门模式：有 regression / 往返失败 / 请求错误则退出码 2（供 CI 用）")
    p.add_argument("--selftest", action="store_true", help="不启网关自检评估器")
    args = p.parse_args(argv)

    if args.selftest:
        return 1 if selftest(args.limit or 40) else 0

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"FAIL: cases not found: {cases_path}", file=sys.stderr)
        return 1

    todo: list[tuple[str, list[tuple[str, str]], str]] = []
    for cid, _subset, expect, text, _misses in load_cases(cases_path):
        todo.append((cid, [(e.type, e.value) for e in expect], text))
        if args.limit and len(todo) >= args.limit:
            break

    def redact(m):
        return gateway_redact(args.base_url, m, args.timeout, args.token, args.strategy)

    def restore(r, rid):
        return gateway_restore(args.base_url, r, rid, args.timeout, args.token)

    reversibility: dict[str, bool | None] = {}
    results = [eval_case(cid, pairs, text, redact, restore, reversibility)
               for cid, pairs, text in todo]
    irreversible = sorted(t for t, ok in reversibility.items() if ok is False)
    print(f"[carriers] 可逆性探测：{reversibility}", file=sys.stderr)
    if irreversible:
        print(f"[carriers] 按设计不可逆（豁免往返断言）：{irreversible}", file=sys.stderr)
    summary = summarize(results)
    report = render_report(summary, results,
                           f"{args.base_url}/v1/privacy/*", cases_path, args.engine)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"carriers_{args.engine}_{ts}.md"
    js_path = out_dir / f"carriers_{args.engine}_{ts}.json"
    md_path.write_text(report, encoding="utf-8")
    js_path.write_text(json.dumps(
        {"engine": args.engine, "endpoint": args.base_url,
         "cases": str(cases_path), "n": len(results),
         "summary": summary, "by_case": results},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(report)
    total_reg = sum(len(summary[c]["regressions"]) for c in CARRIERS)
    total_rt = sum(summary[c]["roundtrip_failures"] for c in CARRIERS)
    total_err = sum(len(summary[c]["errors"]) for c in CARRIERS)
    print(f"[carriers] regressions={total_reg} roundtrip_failures={total_rt} "
          f"request_errors={total_err}")
    print(f"[carriers] md  -> {md_path}")
    print(f"[carriers] json-> {js_path}")

    if args.gate:
        if total_err:
            print(f"[carriers] 门禁未通过：{total_err} 条请求错误 —— "
                  f"端点不可达或返回异常，本次没有任何有效结论", file=sys.stderr)
            return 2
        if total_reg or total_rt:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
