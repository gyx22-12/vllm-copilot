# -*- coding: utf-8 -*-
"""召回策略 A/B：RRF 融合到底是在帮忙还是拖后腿？

逐题诊断发现 8/18 题 gold 根本没进召回（top-20），而其中 4 题「纯向量排名」明明在 20 以内：
  ngram-gpu-update  vec=7  → fused=21（掉出）
  logsumexp         vec=20 → fused=55
  multi-mtp         vec=12 → fused=23
  speculator-factory vec=3 → fused=32
猜测：RRF 给 BM25 一半权重，而 BM25 在这些「去同源」的自然语言 query 上接近噪声
（gold 的 BM25 排名 303/209/141/134…），于是把已经排好的向量信号稀释掉。

本脚本固定其它一切（同一索引、同一重排器、同一 top-10），只换召回策略：
  S1 现行   RRF(1,1) k=60, RECALL_N=20
  S2        纯向量 top-20
  S3        加权 RRF 0.85/0.15, top-20
  S4        现行 RRF 但 RECALL_N=40
  S5        并列名次融合（取两者较好名次）top-20
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
from rag_baseline import QUERY_INSTRUCTION

OUT = r"C:\Users\GYX\WorkBuddy\2026-09-17-10-52-55\rag-review-scripts\_recall_ab.json"
K = 60

client = get_client()
idx = CodeIndex()
idx.load(CODE_DIRS, synth=True, client=client, rel_root=_VLLM_ROOT)
chunks = idx.chunks
overview = idx._overview_mask
w = np.where(overview, OVERVIEW_WEIGHT, 1.0)
texts = [t for _, t in chunks]


def rank_vec(x):
    return np.argsort(np.argsort(-x))


def recall_by(score, n, dedup=True):
    order = np.argsort(score)[::-1]
    seen, out = set(), []
    for i in order:
        key = _symbol_key(chunks[i][1])
        if dedup:
            if key in seen:
                continue
            seen.add(key)
        out.append(i)
        if len(out) >= n:
            break
    return np.asarray(out)


def run_strategy(name, rec_all, queries, qvecs):
    ranked = []
    for q, qv, rec_indices in zip(queries, qvecs, rec_all):
        idxs, scores = idx._rerank.rerank(q, rec_indices, chunks, top_k=None, max_chars=768)
        s = scores * np.where(overview[idxs], OVERVIEW_WEIGHT, 1.0)
        ranked.append(idxs[np.argsort(-s)][:10])
    n = len(queries)
    res = {"strategy": name}
    for k in (1, 3, 5, 10):
        res[f"R@{k}"] = round(sum(
            1 for qa, r in zip(CODE_QA_SET, ranked)
            if any(_has_symbol(texts[i], qa["gold"][0]) for i in r[:k])) / n, 3)
    rrs = []
    for qa, r in zip(CODE_QA_SET, ranked):
        rr = 0.0
        for pos, i in enumerate(r, 1):
            if _has_symbol(texts[i], qa["gold"][0]):
                rr = 1.0 / pos
                break
        rrs.append(rr)
    res["MRR"] = round(sum(rrs) / n, 3)
    return res


queries = [qa["q"] for qa in CODE_QA_SET]
qvecs = [idx._embed.encode([QUERY_INSTRUCTION + q], normalize_embeddings=True)[0] for q in queries]

vec_all = [ (idx.vecs @ qv) * w for qv in qvecs ]
bm_all = [ idx.bm25.scores(q) * w for q in queries ]

results = []
# S1 现行 RRF(1,1)
res_seen = {}
for name, fn in [
    ("S1_current_RRF_1_1_n20", lambda v, b: (1/(K+rank_vec(v)+1) + 1/(K+rank_vec(b)+1), 20)),
    ("S2_vector_only_n20", lambda v, b: (v, 20)),
    ("S3_weighted_RRF_0.85_0.15_n20", lambda v, b: (0.85/(K+rank_vec(v)+1) + 0.15/(K+rank_vec(b)+1), 20)),
    ("S4_current_RRF_n40", lambda v, b: (1/(K+rank_vec(v)+1) + 1/(K+rank_vec(b)+1), 40)),
    ("S5_min_rank_fusion_n20", lambda v, b: (-np.minimum(rank_vec(v), rank_vec(b)).astype(float), 20)),
    ("S6_vector_n40", lambda v, b: (v, 40)),
]:
    per_q_totals = []
    all_rec = []
    for v, b in zip(vec_all, bm_all):
        score, n = fn(v, b)
        all_rec.append(recall_by(score, n))
    r = run_strategy(name, all_rec, queries, qvecs)
    # 记录召回覆盖率
    covered = sum(1 for rec, qa in zip(all_rec, CODE_QA_SET)
                  if any(_has_symbol(texts[i], qa["gold"][0]) for i in rec))
    r["recall_covered"] = f"{covered}/{len(CODE_QA_SET)}"
    results.append(r)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("done")
