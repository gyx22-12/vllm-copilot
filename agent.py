# -*- coding: utf-8 -*-
"""
Agent（里程碑 3）：把单步 RAG 升级成"多步决策"的 ReAct 循环。

单步 RAG（rag_baseline.py）：query → 检索 top-3 → 生成，LLM 只当最后一步的生成器。
Agent：LLM 自己决定"要不要查、查什么、查几次"，循环直到给出答案或超步数。

循环结构：
    for step in range(max_steps):
        resp = LLM(messages, tools=TOOLS)        # 让 LLM 决定下一步
        if resp 有 tool_calls:                    # LLM 想查工具
            把 tool_calls 记进 messages
            逐个 execute 工具，把结果回填 messages
        else:                                     # LLM 直接给答案
            return resp 文本

依赖 + 跑法：
    $env:DEEPSEEK_API_KEY = "sk-你的key"
    py -3.12 agent.py
"""

import os
import json
import re
import subprocess
import sys
import time
import tempfile

# 国内直连 huggingface 会超时，走镜像；必须在 import sentence_transformers 之前设置。
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 模型已缓存时用离线模式加载：跳过「查 adapter_config.json 是否存在」这类多余网络请求，
# 否则到 hf-mirror 的 SSL 握手一旦挂起，SentenceTransformer(...) 会卡死。
# 需要下载新模型时，设 $env:HF_HUB_OFFLINE = "0" 即可切回联网。
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from rag_baseline import load_documents, chunk_markdown_with_parent, embed
from bm25 import BM25, rrf_fusion
from rerank import Reranker
from config import CHUNK_SIZE, RECALL_N, TOP_N, MAX_CONTEXT_CHARS, MAX_CODE_CONTEXT_CHARS  # 切块/检索共用常量单一来源（见 config.py）
from context_budget import pack_budget  # 字符预算装箱单一来源（build_search/build_search_code/eval_qa 三处共用）

# ---------- 常量 ----------
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# CHUNK_SIZE / RECALL_N / TOP_N 见 config.py（切块+检索共用的单一来源，不再这里重复定义）
RERANK_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "deepseek-chat"
RUN_PYTHON_TIMEOUT = 10       # run_python 子进程超时（秒）
RUN_PYTHON_MAX_OUTPUT = 4000  # run_python 输出截断（字符）
RUN_FILE_TIMEOUT = 30         # run_file 子进程超时（秒）：跑的是完整脚本，比 run_python 一段代码宽一些
# run_python / run_file 是「LLM 生成的代码 → 本机子进程执行」的任意代码执行面，默认关闭。
# 显式设 VLLM_COPILOT_ALLOW_RUN_PYTHON=1 才暴露这两个工具、并允许 route() 分派到 run_python。
_ALLOW_RUN_PYTHON = os.environ.get("VLLM_COPILOT_ALLOW_RUN_PYTHON") == "1"
# 写文件工具（read_file / write_file / edit_file）：任意写盘会持久化污染仓库，比执行代码更危险，
# 默认关闭。显式设 VLLM_COPILOT_ALLOW_WRITE=1 才暴露；且只能写 WRITE_ROOT（workspace/）白名单，
# 不碰 corpus/ 与 vLLM 源码——这是「让 agent 真能写代码」的工具，安全边界比 run_python 更严。
_ALLOW_WRITE = os.environ.get("VLLM_COPILOT_ALLOW_WRITE") == "1"
_WRITE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
MAX_FILE_CHARS = 200000   # 单文件读/写/替换的字符上限（防一次写超大文件）

# 代码索引目录（里程碑 5）：vLLM 源码里投机解码的实现
_VLLM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "corpus", "src", "vllm-0.29.0")
CODE_DIRS = [
    os.path.join(_VLLM_ROOT, "vllm", "v1", "spec_decode"),
    os.path.join(_VLLM_ROOT, "vllm", "v1", "worker", "gpu", "spec_decode"),
]

# ---------- 索引（启动时加载一次，供 build_search 复用）----------
CHUNKS = []               # [(来源路径, 文本), ...]
CHUNK_PARENTS = []        # [(来源路径, 父段落文本), ...] 与 CHUNKS 一一对应（小-to-大 检索）
CHUNK_VECS = None         # (n_chunks, dim) 归一化向量
BM25_IDX = None           # BM25 打分器
_EMBED = None             # SentenceTransformer 模型
_RERANK = None            # cross-encoder 重排器
_CODE = None              # 代码索引（CodeIndex，load_index(client) 时加载）


def load_index(client=None):
    """加载 vLLM 文档语料 + 建索引；传入 client 时额外加载代码索引（docstring 合成）。"""
    global CHUNKS, CHUNK_PARENTS, CHUNK_VECS, BM25_IDX, _EMBED, _RERANK, _CODE
    here = os.path.dirname(os.path.abspath(__file__))
    docs = load_documents([os.path.join(here, "corpus", "docs")], extensions=(".md",))
    CHUNKS = []
    CHUNK_PARENTS = []
    for src, text in docs:
        for parent, c in chunk_markdown_with_parent(text, max_chunk=CHUNK_SIZE, parent_level=2):
            CHUNKS.append((src, c))
            CHUNK_PARENTS.append((src, parent))
    _EMBED = SentenceTransformer(EMBED_MODEL)
    CHUNK_VECS = embed([c for _, c in CHUNKS], _EMBED)
    BM25_IDX = BM25([c for _, c in CHUNKS])
    _RERANK = Reranker(RERANK_MODEL)
    print(f"索引就绪：{len(docs)} 文档 → {len(CHUNKS)} chunk（embedding + BM25 + reranker）")

    _CODE = None
    if client is not None:
        from code_index import CodeIndex
        _CODE = CodeIndex()
        n_f, n_c = _CODE.load(CODE_DIRS, synth=True, client=client,
                              embed_model=_EMBED, reranker=_RERANK, rel_root=_VLLM_ROOT)
        print(f"代码索引就绪：{n_f} 个 .py → {n_c} 个符号 chunk（docstring 合成，复用同一套 embedding/rerank）")


def get_client():
    """DeepSeek 客户端。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "未找到 DEEPSEEK_API_KEY。先设置：\n  $env:DEEPSEEK_API_KEY = \"sk-你的key\""
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# ---------- 工具实现 ----------

def build_search(query, top_k=TOP_N, max_chars=MAX_CONTEXT_CHARS):
    """两段式检索：混合召回 top-RECALL_N → cross-encoder 重排 → 返回 top_k（带出处）。

    小-to-大：命中的小块展开成父段落（##节），去重后作为上下文还给 LLM；再用 max_chars
    兜底——父段落展开会把上下文撑大（eval_50 实测 ##节 top-10 均值 3.5 万字符、整篇 14 万），
    按排名截断装箱（pack_budget）：整段装得下就装，装不下截到剩余预算再装、停（截断保留段首答案，
    优于跳过整段——实测 0.86/0.67 vs 0.82/0.60，见 eval_qa 预算消融列）。
    """
    q_vec = embed([QUERY_INSTRUCTION + query], _EMBED)[0]
    vec_scores = CHUNK_VECS @ q_vec
    bm25_scores = BM25_IDX.scores(query)
    fused = rrf_fusion(vec_scores, bm25_scores)
    recall_idx = np.argsort(fused)[::-1][:RECALL_N]                    # 第 1 段：粗召回 RECALL_N 个候选
    rerank_idx, _ = _RERANK.rerank(query, recall_idx, CHUNKS, top_k=top_k)  # 第 2 段：精排 top_k

    parts, seen = [], set()
    for i in rerank_idx:
        parent = CHUNK_PARENTS[i][1]
        if parent in seen:
            continue
        seen.add(parent)
        parts.append((os.path.basename(CHUNK_PARENTS[i][0]), parent))

    # 字符预算：截断装箱（单一来源 pack_budget）——整段装得下就装，装不下截到剩余预算再装、停。
    # 截断保留超长 top-1 的段首（答案句所在），优于跳过整段；装不下时 stderr 会打「返回 N/M 段」观测。
    items = [f"[来源 {src}]\n{p}" for src, p in parts]
    return "\n\n".join(pack_budget(items, max_chars, label="build_search"))


def build_search_code(query, top_k=TOP_N, max_chars=MAX_CODE_CONTEXT_CHARS):
    """在 vLLM 源码索引里检索实现细节，返回带 [source 路径:行号] 出处的源码片段。

    和 build_search 一样加字符预算：代码 chunk（符号粒度 + docstring 合成）也可能很大
    （实测 top-5 曾到 1.8 万字符），按排名截断装箱（pack_budget）：装得下整段装，装不下截断。
    """
    if _CODE is None:
        return "（代码索引未加载）"
    items = [text for _, text in _CODE.search(query, top_k=top_k)]
    return "\n\n".join(pack_budget(items, max_chars, label="build_search_code"))


def build_run_python(code, timeout=RUN_PYTHON_TIMEOUT):
    """在本地子进程里跑一段 Python，返回 stdout/stderr（截断）。

    给 agent 第三种能力：涉及「算一下 / 验证一下」的问题（如按公式算 KV cache 大小、
    核验某段短 vLLM 调用能不能跑通）不再靠猜数值，而是真执行。

    安全边界：只在本机子进程执行、限时、输出截断——仍是任意代码执行，仅用于本地 copilot 场景。
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        out = proc.stdout or ""
        if proc.stderr:
            out += "\n[stderr]\n" + proc.stderr
        if proc.returncode != 0:
            out = f"[退出码 {proc.returncode}]\n" + out
        return (out.strip() or "（无输出）")[:RUN_PYTHON_MAX_OUTPUT]
    except subprocess.TimeoutExpired:
        return f"（执行超时，超过 {timeout}s 已终止）"


def _resolve_write_path(path):
    """把相对路径解析到 WRITE_ROOT 下；越界（绝对路径 / .. / 软链逃逸）抛 ValueError。"""
    root = os.path.realpath(_WRITE_ROOT)
    target = os.path.realpath(os.path.join(root, path))
    if os.path.commonpath([root, target]) != root:
        raise ValueError(f"路径越界：{path} 不在 workspace/ 白名单内")
    return target


def build_read_file(path):
    """读 workspace/ 下的一个文件，返回内容（截断到 MAX_FILE_CHARS）。"""
    try:
        target = _resolve_write_path(path)
        with open(target, encoding="utf-8") as f:
            return f.read(MAX_FILE_CHARS)
    except FileNotFoundError:
        return f"（文件不存在：{path}）"
    except (ValueError, OSError) as e:
        return f"（读文件失败：{e}）"


def build_write_file(path, content):
    """把 content 覆盖写到 workspace/ 下的 path（自动建父目录），返回确认。"""
    try:
        target = _resolve_write_path(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {path}（{len(content)} 字符）"
    except (ValueError, OSError) as e:
        return f"（写文件失败：{e}）"


def build_edit_file(path, old_string, new_string):
    """把 workspace/ 下文件的 old_string 精确替换成 new_string（只替换第一处）。"""
    if not old_string:
        return "（old_string 不能为空串：空串会匹配任意位置，无法安全替换。请给出要替换的原文片段。）"
    try:
        target = _resolve_write_path(path)
        with open(target, encoding="utf-8") as f:
            text = f.read()
        if old_string not in text:
            return f"（未找到要替换的片段：{old_string[:80]}）"
        n = text.count(old_string)
        text = text.replace(old_string, new_string, 1)
        with open(target, "w", encoding="utf-8") as f:
            f.write(text)
        note = f"（另有 {n - 1} 处相同片段未动）" if n > 1 else ""
        return f"已替换 1 处于 {path}{note}"
    except FileNotFoundError:
        return f"（文件不存在：{path}）"
    except (ValueError, OSError) as e:
        return f"（改文件失败：{e}）"


# run_file 的子进程驱动：把「跑被测文件 + 跑 check 断言」整体搬进子进程，结果写临时结果文件。
# 为什么必须子进程（P0-2，见落地顺序文档决策二）：被测代码与运行器共享同一进程时，被测代码
# 既能用 sys.exit() 抛 SystemExit（BaseException，进程内 except Exception 接不住）杀掉运行器，
# 也能用死循环 / os._exit() 挂住或绕过一切异常处理——能拦住这三样的只有进程边界。驱动内的
# except BaseException 只是把 SystemExit 翻译成可读回报，真正承重的是 subprocess.run 的隔离。
# 结果走临时文件、不走 stdout 哨兵：否则被测文件自己打印一行 __RESULT__ 就能污染解析。
# 已知限制：subprocess.run 的超时只杀直接子进程，孙进程（multiprocessing）可能残留；runnable
# 题是纯 Python，影响小（要严格就加 Windows 的 CREATE_NEW_PROCESS_GROUP）。
_RUN_FILE_DRIVER = r'''
import contextlib
import io
import json
import runpy
import sys

target, check, result_file = sys.argv[1], sys.argv[2], sys.argv[3]
out_buf, err_buf = io.StringIO(), io.StringIO()
ns, run_ok, check_ok, err = {}, False, False, ""
try:
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        ns = runpy.run_path(target, run_name="__main__")  # 保持 __main__ 语义（与原 in-process 一致）
    run_ok = True
except BaseException as e:  # SystemExit 也在这里；进程边界内，杀掉的是子进程不是 agent
    err = f"{type(e).__name__}: {e}"
if run_ok and check:
    try:
        exec(compile(check, "<check>", "exec"), ns)
        check_ok = True
    except BaseException as e:
        err = f"{type(e).__name__}: {e}"
with open(result_file, "w", encoding="utf-8") as f:
    json.dump({"run_ok": run_ok, "check_ok": check_ok, "err": err,
               "stdout": out_buf.getvalue(), "stderr": err_buf.getvalue()},
              f, ensure_ascii=False)
'''


def _run_subprocess(target, check, timeout):
    """在子进程里跑一个 .py 文件（runpy）+ 可选 check 断言，返回 (res, timed_out)。

    res = {"run_ok", "check_ok", "err", "stdout", "stderr"}，由 _RUN_FILE_DRIVER 写进临时结果文件；
    驱动没写出结果文件（os._exit / 段错误 / 驱动自身崩）时返回合成的 res（run_ok=False）。
    timed_out 表示超时。build_run_file 与 eval_codegen.run_python_check 共用这条路径（消掉两套执行实现）。
    """
    fd, result_file = tempfile.mkstemp(suffix=".json", prefix="runfile_")
    os.close(fd)
    try:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _RUN_FILE_DRIVER, target, check or "", result_file],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, True
        try:
            with open(result_file, encoding="utf-8") as f:
                return json.load(f), False
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "run_ok": False, "check_ok": False,
                "err": f"进程被硬退出（码 {proc.returncode}）",
                "stdout": proc.stdout or "", "stderr": proc.stderr or "",
            }, False
    finally:
        try:
            os.remove(result_file)
        except OSError:
            pass


def build_run_file(path, check=None):
    """运行 workspace/ 下已写好的 .py 文件（子进程 runpy 执行），返回 stdout/stderr 或 check 结果。

    「写代码 → 真执行」闭环：agent 用 write_file 落盘后，用它把文件真跑一遍——看到报错就修，
    这是 fix-rate（多轮自纠）的来源；run_python 只能跑 LLM 直接给的字符串，跑不了已落盘的文件。

    check 是可选断言片段（如 `assert next_power_of_2(5) == 8`），跑完后 exec 进文件的命名空间，
    用来验证「跑出来的值对不对」。整个「跑 + check」搬进子进程（P0-2）：被测代码 sys.exit()、
    死循环、os._exit() 都被子进程边界隔住，杀不掉 agent 进程（见 _run_subprocess）。
    """
    try:
        target = _resolve_write_path(path)
    except (ValueError, OSError) as e:
        return f"（路径非法：{e}）"
    if not os.path.exists(target):
        return f"（文件不存在：{path}）"

    res, timed_out = _run_subprocess(target, check, RUN_FILE_TIMEOUT)
    if timed_out:
        return f"（执行超时，超过 {RUN_FILE_TIMEOUT}s 已终止）"

    out = res["stdout"]
    if res["stderr"]:
        out += "\n[stderr]\n" + res["stderr"]
    if not res["run_ok"]:
        return (f"[运行失败 {res['err']}]\n" + out.strip())[:RUN_PYTHON_MAX_OUTPUT]
    if check:
        if res["check_ok"]:
            return ("check 通过\n" + (out.strip() or "（无输出）"))[:RUN_PYTHON_MAX_OUTPUT]
        return (f"[check 失败 {res['err']}]\n" + (out.strip() or "（无输出）"))[:RUN_PYTHON_MAX_OUTPUT]
    return (out.strip() or "（无输出）")[:RUN_PYTHON_MAX_OUTPUT]


def route(query):
    """显式 router（保守版）：判三类意图，其余一律预检索文档。

    - write_code：要「生成/写」代码文件 → 走 write_file（gated on _ALLOW_WRITE）。
    - run_python：要「执行/算」一段代码 → 走 run_python（gated on _ALLOW_RUN_PYTHON）。
    - search：其余，预检索文档。

    教训 1（用户 review + test_route.py 实测）：用 implement/kernel/compute 分「文档 vs 源码」有
    11% 误判——65 题里 7 道被拍错。文档题满篇都是这些词，关键词分不开。search_code 是「文档查
    不到再用的兜底」，交给 LLM 在循环里自己决定，不由关键词抢占。
    教训 2（实验 14）：原版把 write 和 run 混在 run_python 的触发词里，「Write a Python script」
    被误路由到 run_python（hint 让 agent 跑代码而非写文件）——router 的意图表里根本没有「生成
    代码」。拆开：write/generate/create/implement/produce → write_code（额外认 function/program/
    class/module 这些名词）；run/execute/compute/calculate → run_python。
    """
    q = query.lower()
    if _ALLOW_WRITE and re.search(
            r"\b(write|generate|create|implement|produce)\b.{0,30}"
            r"\b(python|code|script|snippet|function|program|class|module)\b", q):
        return "write_code"
    if _ALLOW_RUN_PYTHON and re.search(
            r"\b(run|execute|compute|calculate)\b.{0,30}\b(python|code|script|snippet)\b", q):
        return "run_python"
    return "search"


# 工具 JSON Schema：告诉 LLM「有哪些工具、怎么用」。
# 文档检索（概念题）和源码检索（实现题）分开，agent 自己按题型选（原始 router）。
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "在 vLLM 文档库中检索概念/用法/配置：先混合召回（向量+BM25），"
            "再用 cross-encoder 重排，返回最相关的 top-k 个 chunk。"
            "问题问「是什么/怎么配/默认值」时用它。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用于检索的关键词或问题"}
            },
            "required": ["query"]
        }
    }
}

SEARCH_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_code",
        "description": (
            "在 vLLM 源码索引中检索实现细节：函数/类/方法的真实代码，带 文件:行号 出处。"
            "问题问「怎么实现/底层机制/代码逻辑」，或文档查不到、只给概念时用它。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用于检索源码的关键词（函数名/类名/机制名）"}
            },
            "required": ["query"]
        }
    }
}

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "在本机子进程跑一段 Python 代码（限时 10s、输出截断），返回 stdout/stderr。"
            "用于「算一下/验证一下」：按公式算数值、核验某段短 vLLM 调用能否跑通等。"
            "不要用它跑长任务或依赖未安装包的代码。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的完整 Python 源码（字符串）"}
            },
            "required": ["code"]
        }
    }
}

RUN_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "run_file",
        "description": (
            "运行 workspace/ 目录下已写好的 .py 文件（子进程 runpy 执行、限时 30s），返回 stdout/stderr。"
            "写完代码用 write_file 落盘后，再 run_file 真跑一遍看报错、据此修复（可带 check 断言验证结果）。"
            "path 是相对 workspace/ 的相对路径；check 是可选断言片段，跑完 exec 进文件命名空间。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对 workspace/ 的要运行的 .py 文件路径"},
                "check": {"type": "string", "description": "可选：跑完后 exec 进文件命名空间的断言代码（如 assert 某变量值）"}
            },
            "required": ["path"]
        }
    }
}


READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "读 workspace/ 目录下的一个文本文件，返回内容。写代码前先读回已写文件核对。"
            "path 是相对 workspace/ 的相对路径（如 src/foo.py），不能 .. 越界。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对 workspace/ 的文件路径"}
            },
            "required": ["path"]
        }
    }
}

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "把完整内容写入 workspace/ 目录下的一个文件（覆盖写，自动建父目录），用于生成代码/脚本/笔记。"
            "path 是相对 workspace/ 的相对路径（如 src/foo.py），不能 .. 越界；content 是完整文件内容。"
            "写前想清楚整体结构，一次写完整，别写一半。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对 workspace/ 的文件路径"},
                "content": {"type": "string", "description": "完整文件内容"}
            },
            "required": ["path", "content"]
        }
    }
}

EDIT_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "把 workspace/ 目录下文件里的某段精确文本（old_string）替换成 new_string（只替换第一处）。"
            "用于小步改已有文件：old_string 必须与文件原文逐字一致（含缩进），否则替换失败。"
            "path 是相对 workspace/ 的相对路径，不能 .. 越界。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对 workspace/ 的文件路径"},
                "old_string": {"type": "string", "description": "要替换的原文片段（逐字一致）"},
                "new_string": {"type": "string", "description": "替换后的新片段"}
            },
            "required": ["path", "old_string", "new_string"]
        }
    }
}


def _get_tools():
    """工具列表：search 恒有；search_code 代码索引加载后暴露；run_python / 写文件 默认关（环境变量开）。"""
    tools = [SEARCH_TOOL]
    if _CODE is not None:
        tools.append(SEARCH_CODE_TOOL)
    if _ALLOW_RUN_PYTHON:
        tools.extend([RUN_PYTHON_TOOL, RUN_FILE_TOOL])
    if _ALLOW_WRITE:
        tools.extend([READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL])
    return tools


def execute(name, arguments):
    """工具分派：把 JSON 字符串解析成 dict，按工具名调用，返回字符串结果。

    LLM 有时返回非法 JSON（截断/未转义），json.loads 会抛 JSONDecodeError——若不接住，
    会一路炸穿 run() 让整个任务崩掉。这里接住，把错误信息原样喂回给 LLM 让它重试。
    """
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError) as e:
        return f"（工具参数不是合法 JSON：{e}。请重新输出正确格式的工具调用参数。）"
    if name == "search":
        return build_search(args.get("query"))
    if name == "search_code":
        return build_search_code(args.get("query"))
    if name == "run_python":
        return build_run_python(args.get("code"))
    if name == "run_file":
        return build_run_file(args.get("path"), args.get("check"))
    if name == "read_file":
        return build_read_file(args.get("path"))
    if name == "write_file":
        return build_write_file(args.get("path"), args.get("content"))
    if name == "edit_file":
        return build_edit_file(args.get("path"), args.get("old_string"), args.get("new_string"))
    raise ValueError(f"未知工具: {name}")


def _usage(resp):
    """安全取出 token 用量（SDK 版本差异：cached_tokens 可能没有）。"""
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt": 0, "completion": 0, "cached": 0}
    details = getattr(u, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    return {
        "prompt": getattr(u, "prompt_tokens", 0) or 0,
        "completion": getattr(u, "completion_tokens", 0) or 0,
        "cached": cached,
    }


def run(query, client, max_steps=6, temperature=None, trace=False):
    """ReAct 主循环 + 显式 router 预取：先零成本定路由、预检索，再让 LLM 补查或作答。

    返回 (answer, contexts)：contexts 含预检索结果，供 judge.py 评估"答案是否忠实于原文"。
    trace=True 时改返回 (answer, contexts, stats)——stats 是 per-step 延迟/token/工具耗时的
    打点（结构见 profiler.summarize），供 profiler.py 汇总成本。打点收在这里而非另抄一份循环，
    是因为之前 profiler.py 镜像了循环、router/run_python/prefetch 都没跟上，测的是「旧 agent」。
    """
    contexts = []
    n_tool_calls = 0
    if trace:
        t_start = time.perf_counter()
        steps = []
        prefetch_lat = None

    # 1) 显式 router（保守版）：判 run_python / write_code，其余预检索文档（0 次 LLM 调用）。
    routed = route(query)
    prefetch = None
    if routed == "run_python":
        hint = "本题需要跑代码验证——用 run_python 工具执行，别凭空猜数值。"
    elif routed == "write_code":
        hint = ("本题要写/改代码——用 write_file 或 edit_file 把完整代码落到 workspace/ 下"
                "（改已有文件先 read_file 读回），必要时先 search/search_code 查 API/实现，"
                "写完可用 run_file 真跑一遍看报错、据此修复，别只在对话里贴代码或凭空编。")
    else:
        if trace:
            t0 = time.perf_counter()
        result = build_search(query)
        if trace:
            prefetch_lat = time.perf_counter() - t0
        contexts.append(result)
        n_tool_calls += 1
        # 把预检索结果伪装成"已完成的工具调用"喂进历史，LLM 直接基于它作答或补一次精确检索。
        prefetch = (
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "prefetch_0", "type": "function",
                             "function": {"name": "search",
                                          "arguments": json.dumps({"query": query})}}]},
            {"role": "tool", "tool_call_id": "prefetch_0", "content": result},
        )
        hint = ("本题已预检索文档——先据此作答；若不够最多再查 1~2 次（文档不足时可用 search_code 查源码）"
                "就给出答案，别重复查同一关键词。")

    tool_desc = "search（查文档概念）"
    if _CODE is not None:
        tool_desc += "、search_code（查源码实现）"
    if _ALLOW_RUN_PYTHON:
        tool_desc += "、run_python（跑一段 Python 验证/计算）、run_file（运行已写文件）"
    if _ALLOW_WRITE:
        tool_desc += "、read_file/write_file/edit_file（读写 workspace/ 下的文件，写代码用）"
    system = (
        f"你是 vLLM 工程助手。可调用 {tool_desc}。只依据检索结果回答，关键结论标注出处（文件/行号）；"
        "查不到就直说资料不足，不要编造。" + hint
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    if prefetch:
        messages.extend(prefetch)

    for step in range(max_steps):
        if trace:
            t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=_get_tools(), temperature=temperature
        )
        msg = resp.choices[0].message
        if trace:
            rec = {"llm_lat": time.perf_counter() - t0, **_usage(resp), "tools": []}
        if msg.tool_calls:
            # 把"我要调工具"记进历史（显式拼 dict，兼容各版本 SDK）
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                if trace:
                    t1 = time.perf_counter()
                result = execute(tc.function.name, tc.function.arguments)
                if trace:
                    rec["tools"].append((tc.function.name, time.perf_counter() - t1, len(result)))
                print(f"[step {step + 1}] 调工具 {tc.function.name}({tc.function.arguments})")
                print(f"          返回 {len(result)} 字符")
                contexts.append(result)
                n_tool_calls += 1
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            if trace:
                steps.append(rec)
        else:
            print(f"[agent] 共 {n_tool_calls} 次工具调用（含 router 预检索）")
            if trace:
                steps.append(rec)
                return msg.content, contexts, {
                    "total_lat": time.perf_counter() - t_start,
                    "prefetch": ({"lat": prefetch_lat, "chars": len(contexts[0])}
                                 if prefetch_lat is not None else None),
                    "steps": steps,
                }
            return msg.content, contexts
    print(f"[agent] 达最大步数仍未作答；共 {n_tool_calls} 次工具调用")
    if trace:
        return "（达到最大步数，仍未作答）", contexts, {
            "total_lat": time.perf_counter() - t_start,
            "prefetch": ({"lat": prefetch_lat, "chars": len(contexts[0])}
                         if prefetch_lat is not None else None),
            "steps": steps,
        }
    return "（达到最大步数，仍未作答）", contexts


if __name__ == "__main__":
    client = get_client()
    load_index(client)
    print("\n索引就绪。请用英文提问（语料和 embedding 都是英文）。空行或 exit 退出。\n")
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not query or query.lower() in ("exit", "quit", "q"):
            break
        answer, _ = run(query, client)
        print("\n===== 最终回答 =====\n" + answer + "\n")
