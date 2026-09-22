# -*- coding: utf-8 -*-
"""③ 两档测量 runner：2 档（自然/抽象）× 6 策略 = 12 格消融 + 配对 delta 报表（L1/L2）。

口径（见 review-probes/代码检索-两档测量规格.md）：
  - 档① 自然 = eval_data.NATURAL_QA（17 题，A组8+B组9）；档② 抽象 = CODE_QA_SET（同 id 对齐、同 gold）。
    两臂同 id 逐题配对，delta = 题面口径净效应（术语 + 措辞风格）。
  - 同一索引（synth=True 的 CodeIndex）、同一 reranker、只换两件事：题面（nat/abs）× 召回融合（S1~S6）。
  - 主口径 = S1 现行 RRF；S2~S6 只进消融表，不选优（S5 加 -1e-3·max_rank 次序项破并列，见规格 §一.2）。
  - L1 正式结论 = 配对 2×2（b/c/Δ + bootstrap CI + McNemar 精确 p），主指标 R@10（预指定），其余 k 描述性。
  - L2 补充 = ΔMRR（bootstrap CI + Wilcoxon 符号秩）+ 名次差中位数（至少一档命中的子集）。
  - 三行报表：A组(8) / B组(9) / 合计(17)，不合成一个数（ngram-lookup/logsumexp/ngram-gpu-update 三道近乎
    「标识符去下划线」，与其余题不同分布）。
  - 截断常数 21 = RECALL_N(20) + 1：gold 未进召回记 rank=21，rr=0。L1/L2 都挂在 S1（深度 20）上。

口径声明（三项，写进报告）：
  ① Δ 读「词根/符号线索值多少分」，不读「真实用户 vs 抽象」——两臂题面都看着 gold 写出来。
  ② 主指标 R@10 预指定（基线 0.44 可动空间大、不一致对多、功效最高）。
  ③ 截断常数 = 21 显式声明，不声明 ΔMRR 无法解释。

题面标注：
  - 档② dflash 题面「merges the main model's work into the same pass」是错误概括（DFlash 跑自己的
    draft model 并行草案，不并入 target 的 forward pass）；档②已冻结不改，只在报表标注。
  - rejection-sampler 的 RejectionSampler 在 vLLM 有两处定义（v1/sample/ 与 v1/worker/gpu/spec_decode/），
    但 v1/sample/ 不在 CODE_DIRS，索引里只有后者 → 命中判定无歧义；源码层命名过载，报表标注，不重写。

用法：$env:DEEPSEEK_API_KEY="sk-..."; py -3.12 eval_two_tier.py
"""
import json
import os

# 国内直连 huggingface 会超时 + 已缓存模型时跳过联网检查（同 agent.py）。
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from scipy.stats import binomtest, wilcoxon

from code_index import CodeIndex, _symbol_key, OVERVIEW_WEIGHT
from agent import get_client, _VLLM_ROOT, CODE_DIRS
from rag_baseline import QUERY_INSTRUCTION
from eval_data import CODE_QA_SET, NATURAL_QA
from eval_code import _has_symbol
from config import RECALL_N

K = 60                       # RRF 常数（同 _probe_recall_ab.py）
RERANK_MAX_CHARS = 768       # 重排截断（同 _probe_recall_ab.py）
TRUNC = RECALL_N + 1         # 21：未进召回（深度 20）的 rank 记为 21
N_BOOT = 10000
SEED = 42

A_IDS = ["medusa", "multi-mtp", "eagle-spec", "hidden-states-extract",
         "rejection-sampler", "suffix-decode", "adaptive-verify", "mtp"]
B_IDS = ["vocab-mapping", "ngram-lookup", "ngram-gpu-update", "sd-metadata",
         "logsumexp", "autoregressive", "dflash", "speculator-factory", "gemma4"]
IDS = A_IDS + B_IDS
assert list(NATURAL_QA.keys()) == IDS, "NATURAL_QA 键与 A/B 组不一致"


def rank_vec(x):
    """0-based 名次（最大值名次 0），同 _probe_recall_ab.py。"""
    return np.argsort(np.argsort(-x))


def recall_by(score, n):
    """按 score 降序取前 n 个 chunk 索引，按符号身份去重（大函数多段只占一个名额）。"""
    order = np.argsort(score)[::-1]
    seen, out = set(), []
    for i in order:
        key = _symbol_key(texts[i])
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
        if len(out) >= n:
            break
    return np.asarray(out)


# (名字, score 函数(v,b)->(score, 召回深度), 深度)
STRATEGIES = [
    ("S1 RRF(1,1) n20",         lambda v, b: (1/(K+rank_vec(v)+1) + 1/(K+rank_vec(b)+1), 20)),
    ("S2 纯向量 n20",            lambda v, b: (v, 20)),
    ("S3 加权RRF .85/.15 n20",   lambda v, b: (0.85/(K+rank_vec(v)+1) + 0.15/(K+rank_vec(b)+1), 20)),
    ("S4 RRF(1,1) n40",         lambda v, b: (1/(K+rank_vec(v)+1) + 1/(K+rank_vec(b)+1), 40)),
    ("S5 min-rank n20",         lambda v, b: (-(np.minimum(rank_vec(v), rank_vec(b)).astype(float)
                                                + 1e-3 * np.maximum(rank_vec(v), rank_vec(b))), 20)),
    ("S6 纯向量 n40",            lambda v, b: (v, 40)),
]


def query_scores(q):
    qv = idx._embed.encode([QUERY_INSTRUCTION + q], normalize_embeddings=True)[0]
    return (idx.vecs @ qv) * w, idx.bm25.scores(q) * w


def run_tier(score_fn, depth, v, b, q, gold):
    """返回 gold 在重排后完整列表中的 1-based 名次；未进召回 → depth+1（截断常数）。"""
    score, n = score_fn(v, b)
    rec = recall_by(score, n)
    idxs, scores = idx._rerank.rerank(q, rec, chunks, top_k=None, max_chars=RERANK_MAX_CHARS)
    s = scores * np.where(overview[idxs], OVERVIEW_WEIGHT, 1.0)
    full = idxs[np.argsort(-s)]
    for pos, i in enumerate(full, 1):
        if _has_symbol(texts[i], gold):
            return pos
    return n + 1


def bootstrap_delta(nat_hit, abs_hit, n_boot=N_BOOT, seed=SEED):
    """对 n 题重采样，Δ=(b-c)/n 的 95% 百分位 CI。返回 (Δ, lo, hi, b, c)。"""
    nat_hit = np.asarray(nat_hit, dtype=bool)
    abs_hit = np.asarray(abs_hit, dtype=bool)
    n = len(nat_hit)
    b = int((nat_hit & ~abs_hit).sum())
    c = int((~nat_hit & abs_hit).sum())
    delta = (b - c) / n
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(n_boot, n))
    nb = (nat_hit[draw] & ~abs_hit[draw]).sum(axis=1)
    nc = (~nat_hit[draw] & abs_hit[draw]).sum(axis=1)
    lo, hi = np.percentile((nb - nc) / n, [2.5, 97.5])
    return delta, lo, hi, b, c


def mcnemar_exact(b, c):
    if b + c == 0:
        return 1.0
    return binomtest(b, b + c, 0.5).pvalue


def bootstrap_mrr(nat_rank, abs_rank, depth, n_boot=N_BOOT, seed=SEED):
    rr_n = np.array([1/r if r <= depth else 0.0 for r in nat_rank])
    rr_a = np.array([1/r if r <= depth else 0.0 for r in abs_rank])
    delta = float((rr_n - rr_a).mean())
    n = len(rr_n)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(n_boot, n))
    deltas = (rr_n[draw] - rr_a[draw]).mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return rr_n, rr_a, delta, lo, hi


def wilcoxon_p(rr_n, rr_a):
    d = rr_n - rr_a
    if not np.any(d != 0):
        return float("nan")
    try:
        return wilcoxon(rr_n, rr_a).pvalue
    except ValueError:
        return float("nan")


# ================= 索引 + 配对题面 =================
client = get_client()
idx = CodeIndex()
n_files, n_chunks = idx.load(CODE_DIRS, synth=True, client=client, rel_root=_VLLM_ROOT)
chunks = idx.chunks
texts = [t for _, t in chunks]
overview = idx._overview_mask
w = np.where(overview, OVERVIEW_WEIGHT, 1.0)

abs_by_id = {qa["id"]: qa for qa in CODE_QA_SET}
pairs = []  # (id, group, nat_q, abs_q, gold, nat(v,b), abs(v,b))
for gid, group_ids in (("A", A_IDS), ("B", B_IDS)):
    for qid in group_ids:
        qa = abs_by_id[qid]
        nat_q, abs_q, gold = NATURAL_QA[qid], qa["q"], qa["gold"][0]
        pairs.append((qid, gid, nat_q, abs_q, gold,
                      query_scores(nat_q), query_scores(abs_q)))

# ================= 12 格消融 =================
R = {}  # R[(sname, tier)] -> list[int] rank（len 17）
for sname, fn in STRATEGIES:
    depth = 20 if "n20" in sname else 40
    for tier in ("nat", "abs"):
        ranks = []
        for qid, gid, nat_q, abs_q, gold, (vn, bn), (va, ba) in pairs:
            v, b, q = (vn, bn, nat_q) if tier == "nat" else (va, ba, abs_q)
            ranks.append(run_tier(fn, depth, v, b, q, gold))
        R[(sname, tier)] = ranks

print("=" * 100)
print("③ 两档测量：12 格消融（2 档 × 6 策略）")
print(f"索引：{n_files} 个 .py → {n_chunks} 个 chunk；配对题 {len(IDS)} 道（A组{len(A_IDS)} + B组{len(B_IDS)}）")
print(f"截断常数：S1/S2/S3/S5 深度 20 → 未召回 rank=21；S4/S6 深度 40 → 41（MRR 按各自深度截断）")
print("=" * 100)


def agg(ranks, depth):
    n = len(ranks)
    return {k: sum(1 for r in ranks if r <= k) / n for k in (1, 3, 5, 10)} | {
        "MRR": sum(1/r for r in ranks if r <= depth) / n,
        "cov": sum(1 for r in ranks if r <= depth) / n,
    }


hdr = f"{'策略':<22}{'档①自然':>10}  {'档②抽象':>10}"
print(hdr)
print(" " * 22 + "R@1 R@3 R@5 R@10 MRR  cov" + "   " + "R@1 R@3 R@5 R@10 MRR  cov")
ablation = {}
for sname, fn in STRATEGIES:
    depth = 20 if "n20" in sname else 40
    an = agg(R[(sname, "nat")], depth)
    aa = agg(R[(sname, "abs")], depth)
    ablation[sname] = {"depth": depth, "nat": an, "abs": aa}
    fmt = lambda a: f"{a[1]:>4.2f} {a[3]:>4.2f} {a[5]:>4.2f} {a[10]:>4.2f} {a['MRR']:>4.2f} {a['cov']:>4.2f}"
    print(f"{sname:<22}" + fmt(an) + "  " + fmt(aa))

# ================= L1 / L2（主口径 S1，三行） =================
S1 = "S1 RRF(1,1) n20"
nat_rank = np.array(R[(S1, "nat")])
abs_rank = np.array(R[(S1, "abs")])

print("\n" + "=" * 100)
print("L1 配对 delta（主口径 S1 现行 RRF；主指标 R@10 预指定，其余 k 描述性）")
print("Δ = 自然命中率 − 抽象命中率；正 = 自然题面更好。CI = 对 n 题重抽样的 95% 百分位区间。")
print("=" * 100)
print(f"{'分组':<8}{'n':>3}   {'k':>2}  {'R@k 自然':>9} {'R@k 抽象':>9} {'b':>3} {'c':>3} {'Δ':>8} {'CI 95%':>20} {'McNemar p':>10}")

rows = [("A组", A_IDS), ("B组", B_IDS), ("合计", IDS)]
L1 = {}
for label, grp_ids in rows:
    grp = np.isin(np.array([p[0] for p in pairs]), grp_ids)
    for k in (1, 3, 5, 10):
        nh = nat_rank[grp] <= k
        ah = abs_rank[grp] <= k
        delta, lo, hi, b, c = bootstrap_delta(nh, ah)
        p = mcnemar_exact(b, c)
        tag = "  ← 主指标" if k == 10 else ""
        n = int(grp.sum())
        print(f"{label:<8}{n:>3}   {k:>2}  {nh.mean():>9.3f} {ah.mean():>9.3f} {b:>3} {c:>3} "
              f"{delta:>+8.3f} [{lo:>+.3f}, {hi:>+.3f}]{p:>10.3f}{tag}")
        L1[(label, k)] = dict(n=n, nat=float(nh.mean()), abs=float(ah.mean()), b=b, c=c,
                              delta=delta, ci=[lo, hi], mcnemar_p=p)
    print()

print("=" * 100)
print("L2 补充（主口径 S1；截断常数 21）：ΔMRR + 名次差中位数（至少一档命中的子集）")
print("ΔMRR = 自然 MRR − 抽象 MRR；MRR 用 rank∈{1..20}，未召回 rr=0。")
print("=" * 100)
print(f"{'分组':<8}{'n':>3}   {'MRR 自然':>9} {'MRR 抽象':>9} {'ΔMRR':>8} {'CI 95%':>20} {'Wilcoxon p':>11} {'名次差中位':>10}")
L2 = {}
for label, grp_ids in rows:
    grp = np.isin(np.array([p[0] for p in pairs]), grp_ids)
    rn = nat_rank[grp]
    ra = abs_rank[grp]
    rrn, rra, d, lo, hi = bootstrap_mrr(rn, ra, RECALL_N)
    wp = wilcoxon_p(rrn, rra)
    hit_any = (rn <= RECALL_N) | (ra <= RECALL_N)
    med = float(np.median(rn[hit_any] - ra[hit_any])) if hit_any.any() else float("nan")
    n = int(grp.sum())
    print(f"{label:<8}{n:>3}   {rrn.mean():>9.3f} {rra.mean():>9.3f} {d:>+8.3f} "
          f"[{lo:>+.3f}, {hi:>+.3f}] {wp:>11.3f} {med:>+10.0f}")
    L2[label] = dict(n=n, mrr_nat=float(rrn.mean()), mrr_abs=float(rra.mean()), dmrr=d,
                     ci=[lo, hi], wilcoxon_p=wp, rank_diff_median=med)

# ================= 逐题名次表 =================
print("\n" + "=" * 100)
print("逐题名次（主口径 S1；rank=21 表未召回）")
print("=" * 100)
print(f"{'id':<22}{'组':>2} {'自然rank':>8} {'抽象rank':>8} {'Δrank':>6}   {'自然@10':>7} {'抽象@10':>7}")
per_q = []
for i, (qid, gid, nat_q, abs_q, gold, _, _) in enumerate(pairs):
    nr, ar = int(nat_rank[i]), int(abs_rank[i])
    per_q.append(dict(id=qid, group=gid, nat_rank=nr, abs_rank=ar,
                      nat_hit10=nr <= 10, abs_hit10=ar <= 10, gold=gold))
    print(f"{qid:<22}{gid:>2} {nr:>8} {ar:>8} {nr-ar:>+6}   {str(nr<=10):>7} {str(ar<=10):>7}")

# ================= 落盘 =================
out = {
    "meta": {
        "n_files": n_files, "n_chunks": n_chunks, "n_pairs": len(IDS),
        "a_ids": A_IDS, "b_ids": B_IDS, "trunc": TRUNC, "main_strategy": S1, "main_k": 10,
        "declarations": [
            "① Δ 读「词根/符号线索值多少分」，不读「真实用户 vs 抽象」——两臂题面都看着 gold 写出来",
            "② 主指标 R@10 预指定（基线 0.44 可动空间大、不一致对多、功效最高）",
            "③ 截断常数 = 21（RECALL_N+1），未进召回 rank=21、rr=0",
        ],
        "annotations": [
            "档② dflash 题面「merges the main model's work into the same pass」是错误概括（DFlash 跑自己的 draft model 并行草案，不并入 target forward pass）；档②已冻结不改，仅标注",
            "rejection-sampler 的 RejectionSampler 在 vLLM 有两处定义，但 v1/sample/ 不在 CODE_DIRS，索引里只有后者 → 命中判定无歧义；源码层命名过载，仅标注",
        ],
    },
    "ablation": {sname: {"depth": d["depth"], **d} for sname, d in ablation.items()},
    "L1": {f"{lbl}_k{k}": v for (lbl, k), v in L1.items()},
    "L2": L2,
    "per_question": per_q,
}
with open("two_tier_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n已落盘 two_tier_results.json")
