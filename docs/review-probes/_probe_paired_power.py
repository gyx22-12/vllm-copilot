# -*- coding: utf-8 -*-
"""配对显著性 + 统计功效探针（纯只读，不改 rag-project 任何文件）。

要回答的问题：上一轮 6 组召回策略 A/B 里「R@10 0.444→0.556」到底是真信号还是 2 题噪声？
现有 _recall_ab.json 只落了聚合值，无法做配对检验——本脚本重跑 6 个策略并记录逐题名次，
然后给出：
  1) 逐题首发命中名次矩阵（S1..S6）
  2) 配对 McNemar 精确检验（R@1 / R@10，S1 vs 每个替代策略）
  3) 配对 bootstrap 95% CI（ΔR@10 / ΔMRR）
  4) 统计功效：n=18 下的最小可检测差、以及要检测给定效应需要多少题

判据说明：配对设计的意义在于消掉「题目难度」这个方差——同一题换策略，差异只来自策略。
"""
import json
import math
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

OUT = r"C:\Users\GYX\WorkBuddy\2026-09-17-10-52-55\rag-review-scripts\_paired_power.json"
K = 60
N_BOOT = 20000
RNG = np.random.default_rng(20260922)

client = get_client()
idx = CodeIndex()
idx.load(CODE_DIRS, synth=True, client=client, rel_root=_VLLM_ROOT)
chunks = idx.chunks
overview = idx._overview_mask
w = np.where(overview, OVERVIEW_WEIGHT, 1.0)
texts = [t for _, t in chunks]
print("index ready", len(chunks), flush=True)


def rank_vec(x):
    return np.argsort(np.argsort(-x))


def recall_by(score, n, dedup=True):
    """按 fused 分取前 n 个候选（召回阶段），同符号多段去重（对齐生产 search()）。"""
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


STRATEGIES = [
    ("S1_current_RRF_1_1_n20", lambda v, b: (1 / (K + rank_vec(v) + 1) + 1 / (K + rank_vec(b) + 1), 20)),
    ("S2_vector_only_n20", lambda v, b: (v, 20)),
    ("S3_weighted_RRF_0.85_0.15_n20", lambda v, b: (0.85 / (K + rank_vec(v) + 1) + 0.15 / (K + rank_vec(b) + 1), 20)),
    ("S4_current_RRF_n40", lambda v, b: (1 / (K + rank_vec(v) + 1) + 1 / (K + rank_vec(b) + 1), 40)),
    ("S5_min_rank_fusion_n20", lambda v, b: (-np.minimum(rank_vec(v), rank_vec(b)).astype(float), 20)),
    ("S6_vector_n40", lambda v, b: (v, 40)),
]

queries = [qa["q"] for qa in CODE_QA_SET]
gold = [qa["gold"][0] for qa in CODE_QA_SET]
qvecs = [idx._embed.encode([QUERY_INSTRUCTION + q], normalize_embeddings=True)[0] for q in queries]

vec_all = [(idx.vecs @ qv) * w for qv in qvecs]
bm_all = [idx.bm25.scores(q) * w for q in queries]

# per_rank[strategy][qi] = 首发命中名次（1-based），None = top-10 内未命中
per_rank = {}
for name, fn in STRATEGIES:
    ranks = []
    for qi, (v, b) in enumerate(zip(vec_all, bm_all)):
        score, n = fn(v, b)
        rec = recall_by(score, n)
        # 对齐生产检索段：cross-encoder 重排 + module-overview 降权
        idxs, scores = idx._rerank.rerank(queries[qi], rec, chunks, top_k=None, max_chars=768)
        s = scores * np.where(overview[idxs], OVERVIEW_WEIGHT, 1.0)
        ranked = idxs[np.argsort(-s)][:10]
        pos = None
        for p, i in enumerate(ranked, 1):
            if _has_symbol(texts[i], gold[qi]):
                pos = p
                break
        ranks.append(pos)
    per_rank[name] = ranks
    print("done", name, flush=True)


def hit_at(ranks, k):
    return [1 if r is not None and r <= k else 0 for r in ranks]


def mcnemar_exact(b, c):
    """配对二值：b = 前者错后者对，c = 前者对后者错。双侧精确 p（二项分布）。"""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, i) for i in range(m + 1)) * (0.5 ** n)
    return float(min(1.0, 2 * tail))


def boot_ci(deltas, level=0.95):
    """配对 bootstrap：重采样「题」，得到均值差的置信区间。"""
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    idxs = RNG.integers(0, n, size=(N_BOOT, n))
    means = d[idxs].mean(axis=1)
    lo = float(np.percentile(means, (1 - level) / 2 * 100))
    hi = float(np.percentile(means, (1 + level) / 2 * 100))
    p_le0 = float((means <= 0).mean())
    return {"delta": float(d.mean()), "ci95": [round(lo, 3), round(hi, 3)],
            "p_delta_le_0": round(p_le0, 4), "n": n}


base = "S1_current_RRF_1_1_n20"
paired = {}
for name, _ in STRATEGIES:
    if name == base:
        continue
    entry = {}
    for k in (1, 10):
        a, bb = hit_at(per_rank[base], k), hit_at(per_rank[name], k)
        b = sum(1 for x, y in zip(a, bb) if x == 0 and y == 1)  # S1 错、替代对
        c = sum(1 for x, y in zip(a, bb) if x == 1 and y == 0)
        entry[f"R@{k}"] = {"S1": round(sum(a) / len(a), 3), "alt": round(sum(bb) / len(bb), 3),
                           "b_S1miss_altHit": b, "c_S1hit_altMiss": c,
                           "mcnemar_p": round(mcnemar_exact(b, c), 4)}
    rr = lambda ranks: [1.0 / r if r else 0.0 for r in ranks]
    d = [y - x for x, y in zip(rr(per_rank[base]), rr(per_rank[name]))]
    entry["MRR"] = boot_ci(d)
    paired[f"{name}_vs_S1"] = entry

# ---- 统计功效：n=18 下，配对二值差需要多少题才显著 ----
n = len(CODE_QA_SET)
power = {
    "n_questions": n,
    "one_question_worth_of_R@k": round(1.0 / n, 4),
    "note": "配对 McNemar 精确检验：若「替代策略救回 b 题、丢掉 0 题」，双侧 p 的下界。",
    "p_at_b_discordant": {str(b): round(mcnemar_exact(b, 0), 4) for b in (1, 2, 3, 4, 5, 6, 8, 10)},
    "questions_needed_for_p<0.05_one_sided_zero_loss":
        next(q for q in range(1, 200) if mcnemar_exact(q, 0) < 0.05),
}

# 逐题名次矩阵
matrix = []
for qi, qa in enumerate(CODE_QA_SET):
    row = {"id": qa["id"], "gold": gold[qi]}
    for name, _ in STRATEGIES:
        row[name.split("_")[0]] = per_rank[name][qi]
    matrix.append(row)

out = {"per_question_first_hit_rank": matrix, "paired_vs_S1": paired, "power": power}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("done")
