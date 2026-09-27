# -*- coding: utf-8 -*-
"""纯静态诊断（不加载模型）：题面还剩多少「可检索抓手」。

思路：BM25/向量的召回都要靠 query 与答案 chunk 之间的某种重合。给每题算
  overlap_idf = sum(idf(t) for t in query ∩ gold_chunk) / sum(idf(t) for t in query)
idf 从真实代码索引的 df 统计里来（纯 tokenize，不需要任何模型）。
然后和 _stage_diag.json 里的实测「是否进召回」对齐，看这个数是不是预测器。
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, r"C:\Users\GYX\rag-project")
os.chdir(r"C:\Users\GYX\rag-project")

from bm25 import BM25, tokenize
from code_chunker import chunk_python
from code_index import _split_long_chunk
from transformers import AutoTokenizer
from eval_data import CODE_QA_SET
from eval_code import _has_symbol
from agent import CODE_DIRS

VLLM_ROOT = r"C:\Users\GYX\rag-project\corpus\src\vllm-0.29.0"
OUT = r"C:\Users\GYX\WorkBuddy\2026-09-17-10-52-55\rag-review-scripts\_lexical_bridge.json"
DIAG = r"C:\Users\GYX\WorkBuddy\2026-09-17-10-52-55\rag-review-scripts\_stage_diag.json"

files = []
for d in CODE_DIRS:
    files += glob.glob(os.path.join(d, "**", "*.py"), recursive=True)
files = sorted({f for f in files if "__pycache__" not in f})
tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
chunks = []
for p in files:
    try:
        with open(p, encoding="utf-8") as f:
            src = f.read()
    except (UnicodeDecodeError, OSError):
        continue
    rel = os.path.relpath(p, VLLM_ROOT)
    for text, _ in chunk_python(rel, src):
        for _r, t2 in _split_long_chunk(rel, text, tok):
            chunks.append(t2)

bm = BM25(chunks)  # 只为拿 df / idf，不调任何模型

with open(DIAG, encoding="utf-8") as f:
    diag = {r["id"]: r for r in json.load(f)["rows"]}

rows = []
for qa in CODE_QA_SET:
    q = qa["q"]
    gold = qa["gold"][0]
    qtok = [t for t in tokenize(q) if bm.df.get(t, 0) > 0]
    gold_tok = set()
    for t in chunks:
        if _has_symbol(t, gold):
            gold_tok |= set(tokenize(t))
    shared = [t for t in set(qtok) if t in gold_tok]
    tot_idf = sum(bm._idf(t) for t in set(qtok))
    sh_idf = sum(bm._idf(t) for t in shared)
    # 只算「稀有词」抓手：df 越小越稀有；取 df<=10 的
    rare = [t for t in shared if bm.df.get(t, 0) <= 10]
    rows.append({
        "id": qa["id"], "gold": gold,
        "n_query_terms": len(set(qtok)),
        "n_shared_terms": len(shared),
        "shared_terms": sorted(shared),
        "rare_shared_terms": sorted(rare),
        "n_rare_shared": len(rare),
        "idf_weighted_overlap": round(sh_idf / tot_idf, 3) if tot_idf else 0.0,
        "recall_hit": diag[qa["id"]]["recall_has_gold"],
        "final_rank": diag[qa["id"]]["rank_final_dedup_ON"],
    })

hit = [r for r in rows if r["recall_hit"]]
miss = [r for r in rows if not r["recall_hit"]]


def mean(xs, key):
    return round(sum(x[key] for x in xs) / len(xs), 3) if xs else None


out = {
    "n": len(rows),
    "mean_idf_overlap_recall_hit": mean(hit, "idf_weighted_overlap"),
    "mean_idf_overlap_recall_miss": mean(miss, "idf_weighted_overlap"),
    "mean_rare_shared_hit": mean(hit, "n_rare_shared"),
    "mean_rare_shared_miss": mean(miss, "n_rare_shared"),
    "zero_rare_shared": [r["id"] for r in rows if r["n_rare_shared"] == 0],
    "rows": rows,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("done")
