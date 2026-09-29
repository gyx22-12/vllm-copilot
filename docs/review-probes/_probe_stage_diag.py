# -*- coding: utf-8 -*-
"""逐环节诊断 18 题的检索失败：gold 在「向量 / BM25 / RRF融合 / 重排」各环节排第几。

同时做 A/B：同符号多段合并（dedup）开 / 关，隔离它到底贡献了多少。
需要加载模型（约 2-4 分钟）。
"""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\GYX\rag-project")
os.chdir(r"C:\Users\GYX\rag-project")

import numpy as np
from code_index import CodeIndex, _symbol_key, OVERVIEW_WEIGHT
from eval_data import CODE_QA_SET
from eval_code import _has_symbol
from agent import get_client, _VLLM_ROOT, CODE_DIRS
from config import RECALL_N
from rag_baseline import QUERY_INSTRUCTION
from bm25 import rrf_fusion

OUT = r"C:\Users\GYX\WorkBuddy\2026-09-17-10-52-55\rag-review-scripts\_stage_diag.json"

client = get_client()
idx = CodeIndex()
n_files, n_chunks = idx.load(CODE_DIRS, synth=True, client=client, rel_root=_VLLM_ROOT)
chunks = idx.chunks
texts = [t for _, t in chunks]
overview = idx._overview_mask
w = np.where(overview, OVERVIEW_WEIGHT, 1.0)


def rank_of(arr, gold, dedup):
    """gold 在降序数组里的名次（1-based）。dedup=True 时按符号去重后再数。"""
    order = np.argsort(arr)[::-1]
    seen = set()
    rank = 0
    for i in order:
        if dedup:
            key = _symbol_key(chunks[i][1])
            if key in seen:
                continue
            seen.add(key)
        rank += 1
        if _has_symbol(chunks[i][1], gold):
            return rank
        if not dedup and rank > 3000:
            break
    return None


def topk_symbols(arr, k, dedup=True):
    order = np.argsort(arr)[::-1]
    seen, out = set(), []
    for i in order:
        key = _symbol_key(chunks[i][1])
        if dedup:
            if key in seen:
                continue
            seen.add(key)
        out.append(i)
        if len(out) >= k:
            break
    return out


def hit_at(indices, gold, k):
    return any(_has_symbol(chunks[i][1], gold) for i in indices[:k])


rows = []
for qa in CODE_QA_SET:
    gold = qa["gold"]
    q = qa["q"]
    qv = idx._embed.encode([QUERY_INSTRUCTION + q], normalize_embeddings=True)[0]
    vec = (idx.vecs @ qv) * w
    bm = idx.bm25.scores(q) * w
    fused = rrf_fusion(vec, bm)

    r_vec = rank_of(vec, gold[0], True)
    r_bm = rank_of(bm, gold[0], True)
    r_fused = rank_of(fused, gold[0], True)

    # dedup 开：召回 + 重排
    recall_idx = topk_symbols(fused, RECALL_N, dedup=True)
    idxs, scores = idx._rerank.rerank(q, np.asarray(recall_idx), chunks, top_k=None,
                                      max_chars=768)
    s = scores * np.where(overview[idxs], OVERVIEW_WEIGHT, 1.0)
    reordered = idxs[np.argsort(-s)]
    r_final_dedup = rank_of_ordered = None
    for r, i in enumerate(reordered, 1):
        if _has_symbol(chunks[i][1], gold[0]):
            r_final_dedup = r
            break

    # dedup 关：召回里塞满同符号各段，再重排
    recall_raw = topk_symbols(fused, RECALL_N, dedup=False)
    idxs2, scores2 = idx._rerank.rerank(q, np.asarray(recall_raw), chunks, top_k=None,
                                        max_chars=768)
    s2 = scores2 * np.where(overview[idxs2], OVERVIEW_WEIGHT, 1.0)
    reordered2 = idxs2[np.argsort(-s2)]
    r_final_raw = None
    for r, i in enumerate(reordered2, 1):
        if _has_symbol(chunks[i][1], gold[0]):
            r_final_raw = r
            break

    # 词面桥：query 的实词有多少出现在 gold 符号自己的 chunk 里
    import re as _re
    qtok = set(_re.findall(r"[A-Za-z0-9]+", q.lower()))
    gold_chunks = [t for t in texts if _has_symbol(t, gold[0])]
    gtok = set()
    for t in gold_chunks:
        gtok |= set(_re.findall(r"[A-Za-z0-9]+", t.lower()))
    common = sorted(qtok & gtok)

    rows.append({
        "id": qa["id"], "gold": gold[0], "query": q,
        "n_gold_chunks": len(gold_chunks),
        "query_tokens": len(qtok),
        "shared_tokens_with_gold_chunks": common,
        "n_shared": len(common),
        "n_shared_content": len([c for c in common if len(c) > 2]),
        "rank_vector_dedup": r_vec,
        "rank_bm25_dedup": r_bm,
        "rank_fused_dedup": r_fused,
        "rank_final_dedup_ON": r_final_dedup,
        "rank_final_dedup_OFF": r_final_raw,
        "recall_has_gold": any(_has_symbol(chunks[i][1], gold[0]) for i in recall_idx),
        "recall_raw_has_gold": any(_has_symbol(chunks[i][1], gold[0]) for i in recall_raw),
    })

n = len(rows)
summary = {
    "n_questions": n,
    "R@10_on": round(sum(1 for r in rows if r["rank_final_dedup_ON"] and r["rank_final_dedup_ON"] <= 10) / n, 3),
    "R@1_on": round(sum(1 for r in rows if r["rank_final_dedup_ON"] == 1) / n, 3),
    "R@10_off": round(sum(1 for r in rows if r["rank_final_dedup_OFF"] and r["rank_final_dedup_OFF"] <= 10) / n, 3),
    "R@1_off": round(sum(1 for r in rows if r["rank_final_dedup_OFF"] == 1) / n, 3),
    "n_gold_in_recall_on": sum(1 for r in rows if r["recall_has_gold"]),
    "n_gold_in_recall_off": sum(1 for r in rows if r["recall_raw_has_gold"]),
    "n_zero_lexical_bridge": sum(1 for r in rows if r["n_shared_content"] == 0),
    "avg_shared_content": round(sum(r["n_shared_content"] for r in rows) / n, 2),
}
out = {"summary": summary, "rows": rows}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("done")
