# -*- coding: utf-8 -*-
"""「被抹掉的领域词里，有多少是真实用户会说的」——纯静态，不加载模型、不调 API。

问题背景：实验 18 把「同源词」全部从题面抹掉，同时抹掉了两类东西——
  (a) 符号名/内部名（LoRAResolver、prompt_lookup_min）→ 抹掉是对的
  (b) 领域普通词（hidden states / n-gram / draft model / speculation）→ 抹掉是过头的，
      因为这些正是真实用户提问时会用的词。

本脚本量化 (b) 的规模：对每题，取 gold 符号自身的源码文本（就是索引里的答案 chunk 内容），
统计其中「稀有权重高」的词（df<=RARE_DF），并按形态分成两类：
  - natural：全小写、纯字母、长度>=3 的词形（用户会说的那种：speculation / cache / draft）
  - identifier：含大写或下划线、或纯数字的词形（只有读过代码才知道：SpecDecodeMetadata / num_spec_tokens）
"natural 稀有词 >= 1" 的题 = 自然措辞提问时「词面抓手可恢复」的题。
"""
import ast
import json
import os
import re
from collections import Counter

BASE = r"C:\Users\GYX\rag-project"
SRC = os.path.join(BASE, "corpus", "src", "vllm-0.29.0")
CODE_DIRS = [
    "vllm/v1/spec_decode",
    "vllm/v1/worker/gpu/spec_decode",
]
RARE_DF = 3          # 48 个文件里出现在 <=3 个文件 = 真有区分度（df<=10 太松，几乎没筛）
STOP = set("""the a an of to in on for and or not is are was were be been being it its this that these those
as at by with from into over under out up down all any both each few more most other some such no nor only own
same so than too very can will just should now if then else when while where which who whom what how why
self none true false return def class import from pass raise yield lambda args kwargs value values name names
type types list dict set str int float bool len range print""".split())
OUT = r"C:\Users\GYX\WorkBuddy\2026-09-17-10-52-55\rag-review-scripts\_recoverable_bridge.json"

TOK = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOK.findall(text.lower())


# ---- 1. 收集语料 + df ----
files = []
for d in CODE_DIRS:
    root = os.path.join(SRC, d.replace("/", os.sep))
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.endswith(".py"):
                files.append(os.path.join(dirpath, n))

df = Counter()
raw = {}
for p in files:
    try:
        with open(p, encoding="utf-8") as f:
            t = f.read()
    except (UnicodeDecodeError, OSError):
        continue
    raw[p] = t
    for term in set(tokenize(t)):
        df[term] += 1

print("files", len(raw), flush=True)

# ---- 2. 逐题：找 gold 符号的源码段 ----
import sys
sys.path.insert(0, BASE)
from eval_data import CODE_QA_SET  # noqa: E402


def find_symbol(tree, name):
    """返回 (节点, 是否在嵌套块内)。含类/函数/赋值目标。"""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            out.append(node)
        if isinstance(node, (ast.AnnAssign, ast.Assign)):
            tgts = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            for t in tgts:
                if isinstance(t, ast.Name) and t.id == name:
                    out.append(node)
    return out


# 复用它来判「稀有词里哪些是自然语言词」
def is_natural_word(term, seg_lower, seg_raw):
    """term 是 lower 后的 token。若它在原文里以「全小写纯字母、长度>=3」出现过 → natural。"""
    if not term.isalpha() or len(term) < 3:
        return False
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", seg_raw) is not None


rows = []
for qa in CODE_QA_SET:
    gold = qa["gold"][0]
    seg = None
    for p, t in raw.items():
        try:
            tree = ast.parse(t)
        except SyntaxError:
            continue
        nodes = find_symbol(tree, gold)
        if nodes:
            seg = ast.get_source_segment(t, nodes[0]) or ""
            break
    if not seg:
        rows.append({"id": qa["id"], "gold": gold, "found": False})
        continue

    toks = sorted(set(tokenize(seg)))
    rare = [w for w in toks if df.get(w, 0) <= RARE_DF and len(w) >= 3 and w not in STOP]
    natural = [w for w in rare if is_natural_word(w, seg.lower(), seg)]
    ident = [w for w in rare if w not in natural]
    # 按 df 升序（越稀有权重越高）排序，便于看「最抓得住的词」
    natural_sorted = sorted(natural, key=lambda w: df.get(w, 0))
    rows.append({
        "id": qa["id"], "gold": gold, "found": True,
        "n_tokens": len(toks), "n_rare": len(rare),
        "n_rare_natural": len(natural),
        "rare_natural": natural_sorted,
        "rare_natural_df": {w: df.get(w, 0) for w in natural_sorted[:10]},
        "n_rare_ident": len(ident),
        "rare_ident_sample": ident[:12],
    })

with_natural = [r for r in rows if r.get("n_rare_natural", 0) >= 1]
summary = {
    "n_questions": len(rows),
    "n_questions_with_recoverable_natural_term": len(with_natural),
    "n_questions_zero_natural": sum(1 for r in rows if r.get("found") and r["n_rare_natural"] == 0),
    "mean_rare_natural": round(sum(r.get("n_rare_natural", 0) for r in rows) / len(rows), 2),
    "ids_with_natural": [r["id"] for r in with_natural],
    "ids_without_natural": [r["id"] for r in rows if r.get("found") and r["n_rare_natural"] == 0],
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))

# 上一轮 lexicial_bridge 认定的「题面与答案零稀有词重合」10 题 —— 看它们答案里的自然语言稀有词
ZERO_SHARED = ["ngram-gpu-update", "sd-metadata", "logsumexp", "medusa", "mtp", "multi-mtp",
               "dflash", "speculator-factory", "eagle-spec", "gemma4"]
print("\n===== 上一轮「题面零稀有词重合」的题，其答案 chunk 里的自然语言稀有词 =====")
for r in rows:
    if r.get("id") in ZERO_SHARED:
        print(f"\n[{r['id']}]  gold={r['gold']}")
        print(f"  稀有自然语言词({r['n_rare_natural']}): {r['rare_natural'][:14]}")
        print(f"  df: {r['rare_natural_df']}")
        print(f"  稀有内部名样例: {r['rare_ident_sample']}")
print("done")
