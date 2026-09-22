# -*- coding: utf-8 -*-
"""
代码生成评测（写代码工具的静态事实核对）：把「写代码写对没」变成数字。

镜像 answer_eval 的「facts 静态核对」判法，但判的对象换成「agent 生成的代码」：
  - 每题 = 一条「写代码」指令 + 若干「代码事实」facts
  - facts = 代码结构断言（import / 调用 / 关键字参数 / 函数/类定义 / 字段 / 赋值），
    对 AST 做结构核对（见 code_facts.py）——注释/字符串进不了语法树，假货和真货不再混淆
  - 三层判分：① ast.parse 语法门（能不能加载）→ ② facts 结构核对（API 用对没）
    → ③ runnable=True 的纯 Python 题真执行 run_check 断言（比静态更强的信号）
  - 两个汇总数：pass@1（首轮 run 就过）/ fix-rate（首轮 run 没过 → 喂回失败，让 agent 用
    run_file 自测自纠后再判），对应 review 里的「写文件≠能运行」——run_file 补上执行闭环。

和 answer_eval 的分工：那里 facts 判「说没说对」，这里 facts 判「代码里有没有用对」。
和 eval_code（search_code 检索）的分工：那里评「源码检索到没」，这里评「写出来的代码对不对」。

facts 是 ground truth，写前 grep 文档核对的 API 面：
  - kv_cache_dtype="fp8" / enable_prefix_caching=True / max_num_batched_tokens=16384
  - from vllm.lora.request import LoRARequest + enable_lora=True + LoRARequest(...) / lora_request=
  - StructuredOutputsParams(choice=...) + SamplingParams(structured_outputs=...)（guided_* 已废弃）
  - speculative_config={"method":"ngram", prompt_lookup_min/max, num_speculative_tokens}

评测把 agent 当「黑盒」跑：agent.run() 走生产 loop（router + 工具），只额外开 write 工具
（VLLM_COPILOT_ALLOW_WRITE=1），让 agent 用 write_file 把代码落盘到 workspace/，再读回来判。

用法：
    $env:DEEPSEEK_API_KEY = "sk-你的key"
    py -3.12 eval_codegen.py
"""

import ast
import os
import re

# 必须放在 import agent 之前：agent 在模块加载时读这两个开关。
os.environ["VLLM_COPILOT_ALLOW_WRITE"] = "1"  # 暴露 read_file/write_file/edit_file（写代码被测的工具）
os.environ["VLLM_COPILOT_ALLOW_RUN_PYTHON"] = "1"  # 暴露 run_python/run_file（fix-rate 轮要 run_file 自测自纠）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

import agent
from agent import get_client, _WRITE_ROOT
from eval_data import CODE_GEN_QUESTIONS
from code_facts import check_facts  # AST 结构核对（实验 16：替代 substring 判分）


def extract_code(answer, path):
    """优先读 workspace 里落盘的文件；没落盘则回退到答案里的 ```python 代码块；再不行整段答案。"""
    full = os.path.join(_WRITE_ROOT, path)
    if os.path.exists(full):
        with open(full, encoding="utf-8") as f:
            return f.read()
    m = re.search(r"```python\s*\n?(.*?)```", answer, re.DOTALL)
    if m:
        return m.group(1).strip()
    return answer


def syntax_ok(code):
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"{e.__class__.__name__}: {e}"


def run_python_check(code, check, path):
    """纯 Python 题真执行：把代码 exec 进干净命名空间，再跑断言片段。

    只对 runnable=True 的纯函数题用（不 import vllm、无副作用），和 agent.run_python 的
    目的同源（验证写出的代码真能跑通）。失败返回 (False, 错误信息)。
    """
    try:
        ns = {}
        exec(compile(code, path, "exec"), ns)
        exec(check, ns)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def eval_one(item, client, fix_prompt=None):
    """跑一轮：seed 预铺 → prompt → agent.run → 读回 workspace 代码 → 三层判分。

    fix_prompt 非空表示「修复轮」：不重新 seed（在上一轮产物上继续改），用带错误信息的 prompt。
    返回结果 dict（供汇总 + 修复轮衔接）。
    """
    path = item["path"]
    if fix_prompt is None:  # 只有第一轮才 seed 预置文件（修复轮在已有产物上改）
        for seed_path, seed_content in item.get("seed", {}).items():
            full = os.path.join(_WRITE_ROOT, seed_path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(seed_content)
    prompt = fix_prompt or (
        item["task"] + "\n\n"
        "Use the read_file/write_file/edit_file tools as needed to produce the code in "
        "workspace/" + path + ". Reply with just the file path and a one-sentence summary."
    )
    answer, _ = agent.run(prompt, client, temperature=0)
    code = extract_code(answer, path)
    syn, syn_err = syntax_ok(code)
    facts = check_facts(code, item["facts"])
    n_hit = sum(1 for _, ok in facts if ok)
    n_fact = len(facts)
    run, run_err = (None, None)
    if item.get("runnable"):
        run, run_err = run_python_check(code, item["run_check"], path)
    return {"id": item["id"], "ast": syn, "n_hit": n_hit, "n_fact": n_fact,
            "run": run, "facts": facts, "run_err": run_err, "syn_err": syn_err,
            "code": code}


def _print_round(r):
    status = "✓" if r["ast"] else "✗"
    print(f"  ast语法 {status}  {'' if r['ast'] else r['syn_err']}")
    print(f"  facts {r['n_hit']}/{r['n_fact']}")
    for desc, ok in r["facts"]:
        print(f"    {'✓' if ok else '✗'} {desc}")
    if r["run"] is not None:
        print(f"  run   {'✓ 通过' if r['run'] else '✗ ' + (r['run_err'] or '')}")
    first = next((ln for ln in r["code"].splitlines() if ln.strip()), "<空>")
    print(f"  产物首行: {first[:90]}")


def main():
    client = get_client()
    agent.load_index(client)  # 同生产：doc 索引 + code 索引（让 search_code 工具可用）

    # 清空 workspace，避免上一轮落盘文件被当成本轮产物读出来。
    os.makedirs(_WRITE_ROOT, exist_ok=True)
    for f in os.listdir(_WRITE_ROOT):
        os.remove(os.path.join(_WRITE_ROOT, f))

    rows = []
    for item in CODE_GEN_QUESTIONS:
        print(f"\n===== {item['id']} =====")
        r = eval_one(item, client)
        rows.append(r)
        _print_round(r)

    # ---- fix-rate 轮：runnable 题第一轮 run 没过 → 喂回失败，让 agent 用 run_file 自测并修复 ----
    # 只对「真执行」题做：静态 facts 没过没有可喂回的运行时错误，run_file 也帮不上。
    fixable = [r for r in rows if r["run"] is False]
    for r in fixable:
        item = next(it for it in CODE_GEN_QUESTIONS if it["id"] == r["id"])
        print(f"\n===== {r['id']} 修复轮 =====")
        fix_prompt = (
            item["task"] + "\n\n"
            "Your code in workspace/" + item["path"] + " failed this test:\n"
            + item["run_check"] + "\n"
            "Actual error:\n" + (r["run_err"] or "（未知）") + "\n\n"
            "Read the file with read_file, fix it with edit_file/write_file, then verify with the "
            "run_file tool by passing the test above as its `check` argument. "
            "Reply with just the file path and a one-sentence summary."
        )
        r2 = eval_one(item, client, fix_prompt=fix_prompt)
        r["fix_run"] = r2["run"]
        r["fix_err"] = r2["run_err"]
        print(f"  修复后 run: {'✓ 通过' if r2['run'] else '✗ ' + (r2['run_err'] or '')}")

    # 汇总
    n = len(rows)
    n_ast = sum(1 for r in rows if r["ast"])
    n_fact_hit = sum(r["n_hit"] for r in rows)
    n_fact_total = sum(r["n_fact"] for r in rows)
    runnable = [r for r in rows if r["run"] is not None]
    n_pass1 = sum(1 for r in runnable if r["run"] is True)
    n_fixed = sum(1 for r in fixable if r.get("fix_run") is True)

    print("\n===== 代码生成正确性汇总 =====")
    print(f"  语法可加载(ast.parse) : {n_ast}/{n}")
    print(f"  fact 命中            : {n_fact_hit}/{n_fact_total}  ({n_fact_hit/n_fact_total:.2f})")
    if runnable:
        print(f"  pass@1（首轮 run 过）: {n_pass1}/{len(runnable)}  ({n_pass1/len(runnable):.2f})")
    if fixable:
        print(f"  fix-rate（失败→修复轮过）: {n_fixed}/{len(fixable)}  ({n_fixed/len(fixable):.2f})")

    print("\n  每题矩阵（ast / fact / run）：")
    print("  id               ast  fact      run")
    for r in rows:
        run_col = "  -" if r["run"] is None else ("✓" if r["run"] else "✗")
        print(f"  {r['id']:<16}  {'✓' if r['ast'] else '✗'}    {r['n_hit']}/{r['n_fact']:<6}  {run_col}")

    # 失败题聚焦：哪些 fact / run 没过
    fails = [r for r in rows if (not r["ast"] or r["n_hit"] < r["n_fact"] or r["run"] is False)]
    if fails:
        print("\n===== 失败题诊断 =====")
        for r in fails:
            print(f"\n❌ {r['id']}")
            if not r["ast"]:
                print(f"   语法错误: {r['syn_err']}")
            for desc, ok in r["facts"]:
                if not ok:
                    print(f"   fact 未过: {desc}")
            if r["run"] is False:
                print(f"   run 未过: {r['run_err']}")
    else:
        print("\n  全部通过。")


if __name__ == "__main__":
    main()
