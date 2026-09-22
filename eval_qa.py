# -*- coding: utf-8 -*-
"""
评测脚本（里程碑 2 核心）：把"我感觉检索变好了"变成数字 —— Recall@k。

Recall@k：对每个问题，检索 top-k 个 chunk，能同时找到该题 3 个"答案关键词"（gold）就算命中；
Recall@k = 命中题数 / 总题数。要成套看 @1/@3/@5/@10，比"同样 k 下改动前 vs 改动后"。

当前实验：切块固定用"按结构切块"（上一实验最佳），对比五种检索——
  1. 纯向量（语义相似度）
  2. 纯 BM25（词频 / 精确关键词）
  3. 混合（RRF 倒数排名融合）—— 只到召回段
  4. 混合召回 + cross-encoder 重排 —— 和 agent 的真实 pipeline（build_search）一致
  5. 纯 BM25 召回 + cross-encoder 重排 —— 换召回源，验证瓶颈在召回段还是重排段
目标：不止看"哪种单检索最好"，而是看召回段 vs 重排段各自贡献了多少指标。

用法：
    py -3.12 eval_qa.py
"""

import math
import os
from itertools import combinations

# 国内直连 huggingface 会超时，走镜像；必须在 import sentence_transformers 之前设置。
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_baseline import load_documents, chunk_markdown, chunk_markdown_with_parent, embed
from bm25 import BM25, rrf_fusion
from rerank import Reranker, rerank_row
from eval_data import QA_SET, QA_SEMANTIC  # 题目 + gold 单一来源（见 eval_data.py）
from config import CHUNK_SIZE, RECALL_N, MAX_CONTEXT_CHARS  # 切块/检索共用常量单一来源（见 config.py）
from context_budget import pack_budget  # 字符预算装箱单一来源（和 agent.py 生产 build_search 同逻辑）

RERANK_MODEL = "BAAI/bge-reranker-base"

# bge-base-en-v1.5：英文语义模型。bge-v1.5 系列建议 query 加指令前缀、文档不加。
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def build_chunks(docs, chunk_fn):
    """把每个文档用 chunk_fn 切块，返回 [(来源, 块文本), ...]。"""
    chunks = []
    for src, text in docs:
        for c in chunk_fn(text):
            chunks.append((src, c))
    return chunks


def recall_at_k(qa_set, scores_matrix, chunks, k):
    """scores_matrix: (n_queries, n_chunks)，每行是该 query 对所有 chunk 的打分。"""
    hit = 0
    for qa, scores in zip(qa_set, scores_matrix):
        top_idx = np.argsort(scores)[::-1][:k]
        joined = "\n".join(chunks[i][1] for i in top_idx)
        if all(kw in joined for kw in qa["gold"]):
            hit += 1
    return hit / len(qa_set)


def mrr(qa_set, scores_matrix, chunks):
    """MRR：对每题找"首次凑齐 3 个 gold 关键词"的排名 rank，取 1/rank 的均值。
    比 Recall@k 多一层信息：Recall@10=1 的题，命中在第 2 名和第 10 名是两回事。"""
    rrs = []
    for qa, scores in zip(qa_set, scores_matrix):
        order = np.argsort(scores)[::-1]
        seen = set()
        rank = None
        for i, idx in enumerate(order):
            for kw in qa["gold"]:
                if kw in chunks[idx][1]:
                    seen.add(kw)
            if len(seen) == len(qa["gold"]):
                rank = i + 1
                break
        rrs.append(1.0 / rank if rank is not None else 0.0)
    return sum(rrs) / len(rrs)


def ndcg_at_k(qa_set, scores_matrix, chunks, k=10):
    """nDCG@k：每块的相关度 = 含几个 gold 关键词（0~3），DCG 按 1/log2(rank+1) 折损，
    除以理想排序（把最高相关度块排最前）的 IDCG 归一化到 [0,1]。"""
    vals = []
    for qa, scores in zip(qa_set, scores_matrix):
        order = np.argsort(scores)[::-1][:k]
        rels = [sum(1 for kw in qa["gold"] if kw in chunks[i][1]) for i in order]
        dcg = sum(r / math.log2(j + 2) for j, r in enumerate(rels))
        ideal = sorted(
            (sum(1 for kw in qa["gold"] if kw in t) for _, t in chunks), reverse=True
        )[:k]
        idcg = sum(r / math.log2(j + 2) for j, r in enumerate(ideal))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(vals) / len(vals)


def _coverable(qa, units, k):
    """该粒度下此题是否「有解」：是否存在 ≤k 个 unit 凑齐该题全部 gold 关键词。"""
    gold = qa["gold"]
    full = (1 << len(gold)) - 1
    masks = set()
    for _, t in units:
        masks.add(sum(1 << i for i, kw in enumerate(gold) if kw in t))
    masks.discard(0)
    if full in masks:
        return True
    masks = list(masks)
    for r in range(2, min(k, len(gold)) + 1):
        for combo in combinations(masks, r):
            ored = 0
            for m in combo:
                ored |= m
            if ored == full:
                return True
    return False


def ceiling(qa_set, units, k):
    """该粒度下 Recall@k 的理论上限 = 有解题数 / 总题数。

    「有解」= 存在 ≤k 个 unit 凑齐全部 gold。答案被该粒度切散（关键词跨 >k 个 unit）或
    切没（某词不在任何 unit）的题无解——再完美的检索也到不了 1.0。上限把「检索没做好」和
    「这粒度根本做不到」分开；达成率 = 实际 R@k ÷ 上限，越接近 100% 检索越接近最优。
    """
    return sum(1 for qa in qa_set if _coverable(qa, units, k)) / len(qa_set)


def report(scores_matrix, chunks, title, qa_set=QA_SET):
    """打印某套打分方法下的 Recall@k 表 + 每题命中矩阵。"""
    print(f"\n===== {title} =====")
    print("  k  |  Recall@k")
    print("-----+-----------")
    for k in (1, 3, 5, 10):
        print(f" {k:>2}  |   {recall_at_k(qa_set, scores_matrix, chunks, k):.2f}")

    print(f" MRR = {mrr(qa_set, scores_matrix, chunks):.2f}    "
          f"nDCG@5 = {ndcg_at_k(qa_set, scores_matrix, chunks, 5):.2f}    "
          f"nDCG@10 = {ndcg_at_k(qa_set, scores_matrix, chunks, 10):.2f}")

    ks = (1, 3, 5, 10)
    print("\n  每题命中矩阵（✓ = top-k 里能找全 3 个关键词）：")
    print("   " + "    ".join(f"k={k}" for k in ks) + "   问题")
    for qa, scores in zip(qa_set, scores_matrix):
        marks = []
        for k in ks:
            top_idx = np.argsort(scores)[::-1][:k]
            joined = "\n".join(chunks[i][1] for i in top_idx)
            marks.append("✓" if all(kw in joined for kw in qa["gold"]) else "✗")
        print("   " + "     ".join(marks) + "    " + qa["q"][:24])


def print_misses(qa_set, scores_matrix, chunks, k=10):
    """失败题诊断：对 Recall@k 没凑齐 3 个关键词的题，打印缺失词 + 检索到的 chunk 各自命中什么。

    把"✗"从结果变成可行动的信息：一眼看到是哪几个 gold 词漏了、被哪块勉强命中。
    """
    print(f"\n===== 失败题诊断（Recall@{k} 没凑齐）=====")
    for qa, scores in zip(qa_set, scores_matrix):
        top_idx = np.argsort(scores)[::-1][:k]
        joined = "\n".join(chunks[i][1] for i in top_idx)
        if all(kw in joined for kw in qa["gold"]):
            continue
        missing = [kw for kw in qa["gold"] if kw not in joined]
        print(f"\n❌ {qa['q'][:64]}")
        print(f"   缺失关键词: {missing}")
        for rank, i in enumerate(top_idx, 1):
            hits = [kw for kw in qa["gold"] if kw in chunks[i][1]]
            if hits:
                print(f"   top{rank:>2} 命中 {hits}: {chunks[i][1][:90]!r}")


def report_parent(scores_matrix, chunks, parents, title, qa_set=QA_SET, ceiling_note=""):
    """父段落扩展版评测（小-to-大）：检索 top-k 小块不变，命中后展开成去重父段落再判 gold。

    「拆表」：小块 Recall（纯检索）/ 父段落 Recall（返回上下文后）/ 预算 R@k（8000 字符装箱后，
    = 生产 build_search 的口径，是「预算消融」——和父段落列同 top-k、只差预算）/ 该粒度上限（ceiling）/
    达成率（父段落 R@k ÷ 上限）/ 平均上下文字符 六列并排。上限 = 存在 ≤k 个父段落凑齐 gold 的题数占比，
    把「检索没做好」和「这粒度根本做不到」分开；达成率越接近 100% 检索越接近最优。
    平均上下文字符把「返回更大上下文」的成本显性化；预算列把「被预算砍掉的那部分召回」显性化。
    """
    n = len(qa_set)
    recall = {k: 0 for k in (1, 3, 5, 10)}
    budgeted = {k: 0 for k in (1, 3, 5, 10)}
    n_trunc = 0
    mrr_sum = ndcg5_sum = ndcg10_sum = 0.0
    uniqs = []
    for qa, scores in zip(qa_set, scores_matrix):
        order = np.argsort(scores)[::-1]
        uniqs.append(list(dict.fromkeys(parents[i][1] for i in order)))

    print(f"\n===== {title} =====")
    if ceiling_note:
        print(f"  ⚠ {ceiling_note}")
    print("  k  |  小块 R@k |  父段落 R@k | 8000预算 R@k |  上限  | 达成率 |  平均上下文字符")
    print("-----+-----------+-------------+--------------+--------+--------+----------------")
    ctx_by_k = {k: [] for k in (1, 3, 5, 10)}
    for qa, scores, uniq in zip(qa_set, scores_matrix, uniqs):
        rels = [sum(1 for kw in qa["gold"] if kw in p) for p in uniq]
        for k in (1, 3, 5, 10):
            joined = "\n".join(uniq[:k])
            if all(kw in joined for kw in qa["gold"]):
                recall[k] += 1
            packed = pack_budget(list(uniq[:k]), MAX_CONTEXT_CHARS, quiet=True)
            if all(kw in "\n".join(packed) for kw in qa["gold"]):
                budgeted[k] += 1
            ctx_by_k[k].append(sum(len(p) for p in uniq[:k]))
        # 截断统计：无预算 top-10 上下文超 8000 → 生产上必有段被丢/被截
        if sum(len(p) for p in uniq[:10]) > MAX_CONTEXT_CHARS:
            n_trunc += 1
        seen, rank = set(), None
        for r, p in enumerate(uniq, 1):
            seen |= {kw for kw in qa["gold"] if kw in p}
            if len(seen) == len(qa["gold"]):
                rank = r
                break
        mrr_sum += (1.0 / rank) if rank else 0.0
        for k in (5, 10):
            dcg = sum(x / math.log2(j + 2) for j, x in enumerate(rels[:k]))
            ideal = sorted(rels, reverse=True)[:k]
            idcg = sum(x / math.log2(j + 2) for j, x in enumerate(ideal))
            val = (dcg / idcg) if idcg else 0.0
            if k == 5:
                ndcg5_sum += val
            else:
                ndcg10_sum += val

    for k in (1, 3, 5, 10):
        chunk_rec = recall_at_k(qa_set, scores_matrix, chunks, k)
        parent_rec = recall[k] / n
        budget_rec = budgeted[k] / n
        ceil_k = ceiling(qa_set, parents, k)
        rate = (parent_rec / ceil_k) if ceil_k > 0 else 0.0
        avg_ctx = sum(ctx_by_k[k]) / n
        print(f" {k:>2}  |    {chunk_rec:.2f}    |     {parent_rec:.2f}      |     {budget_rec:.2f}      |  {ceil_k:.2f}  |  {rate*100:3.0f}%  |   {avg_ctx:7.0f}")

    print(f" MRR = {mrr_sum / n:.2f}    nDCG@5 = {ndcg5_sum / n:.2f}    nDCG@10 = {ndcg10_sum / n:.2f}")
    print(f"  预算消融（{MAX_CONTEXT_CHARS} 字符，同 top-10）：R@10 {recall[10] / n:.2f} → {budgeted[10] / n:.2f}，"
          f"top-10 上下文超预算被截 {n_trunc}/{n} 题")
    lens = np.array(ctx_by_k[10])
    print(f"  top-10 上下文成本（字符/题）：均值 {lens.mean():.0f}  中位 {np.median(lens):.0f}  "
          f"p95 {np.percentile(lens, 95):.0f}  最大 {lens.max():.0f}")
    misses = [qa["q"][:34] for qa, scores, uniq in zip(qa_set, scores_matrix, uniqs)
              if not all(kw in "\n".join(uniq[:10]) for kw in qa["gold"])]
    print(f"   k=10 仍 miss: {misses if misses else '无'}")


def main():
    model = SentenceTransformer(EMBED_MODEL)
    here = os.path.dirname(os.path.abspath(__file__))
    # 语料换成真实 vLLM 文档（corpus/docs），不再用本项目自己的文件
    docs = load_documents([os.path.join(here, "corpus", "docs")], extensions=(".md",))
    # 切块固定用"按结构切块"（上一实验最佳），只换检索方法
    chunks = build_chunks(docs, lambda t: chunk_markdown(t, max_chunk=CHUNK_SIZE))
    chunk_texts = [c for _, c in chunks]
    print(f"语料：{len(docs)} 个文档 → {len(chunks)} 个 chunk（结构切块 max={CHUNK_SIZE}，模型 {EMBED_MODEL}）")

    # 父段落标签（小-to-大）：只加信息、不改切块。校验 chunk 文本与 chunk_markdown 完全一致，
    # 这样「父段落扩展」和基线共用同一套 chunk 索引，唯一变量是「命中后还什么给 LLM」。
    parents_section, parents_doc = [], []
    for src, text in docs:
        sec = chunk_markdown_with_parent(text, max_chunk=CHUNK_SIZE, parent_level=2)
        docp = chunk_markdown_with_parent(text, max_chunk=CHUNK_SIZE, parent_level=1)
        ref = chunk_markdown(text, max_chunk=CHUNK_SIZE)
        assert [c for _, c in sec] == ref, f"父段落(##节)切块与 chunk 不一致: {src}"
        assert [c for _, c in docp] == ref, f"父文档切块与 chunk 不一致: {src}"
        parents_section += [(src, p) for p, _ in sec]
        parents_doc += [(src, p) for p, _ in docp]
    print(f"父段落标签就绪：##节（parent_level=2）与整篇（parent_level=1）各 {len(parents_section)} 条，与 chunk 一一对齐")

    # 评测前体检（见 check_gold.py）：无解 = 硬失败；多文档可解 = 警告。坏题先修再跑。
    from check_gold import check
    rows = [r for qs in (QA_SET, QA_SEMANTIC) for r in check(qs, chunks)]
    unsolvable = [r for r in rows if not r["answer_docs"]]
    ambiguous = [r for r in rows if len(r["answer_docs"]) > 1]
    if unsolvable:
        print("\n🔴 评测集体检失败：以下题无解（没有任何文档凑齐 3 词），先修再跑：")
        for r in unsolvable:
            print(f"  - {r['q'][:56]}")
    elif ambiguous:
        print("\n🟡 评测集体检警告（gold 没钉住单一事实，检索到任一答案文档都命中，建议收紧）：")
        for r in ambiguous:
            print(f"  - {r['q'][:56]}  可解文档={r['answer_docs']}")
    else:
        print("\n✓ 评测集体检通过：每题唯一答案文档")

    # chunk 向量 + BM25 索引只建一次（语料不变），query 按题目集合分别算
    chunk_vecs = embed(chunk_texts, model)
    bm25 = BM25(chunk_texts)
    reranker = Reranker(RERANK_MODEL)

    for qa_set, label in ((QA_SET, "精确词题"), (QA_SEMANTIC, "语义改写题")):
        queries = [qa["q"] for qa in qa_set]
        q_vecs = embed([QUERY_INSTRUCTION + q for q in queries], model)
        vector_scores = q_vecs @ chunk_vecs.T          # (n_q, n_chunk)
        bm25_scores = np.array([bm25.scores(q) for q in queries])
        hybrid_scores = np.array([
            rrf_fusion(vector_scores[i], bm25_scores[i]) for i in range(len(queries))
        ])
        # ④/⑤ 重排段：rerank 只重排候选、不新增/删除，所以只改 MRR/nDCG、不改 Recall 上界。
        # ④ 对齐 agent 的真实 pipeline（混合召回）；⑤ 换召回源为纯 BM25，验证瓶颈在召回段还是重排段。
        def _rerank_on(base_scores):
            out = np.zeros_like(base_scores)
            for i, q in enumerate(queries):
                recall_idx = np.argsort(base_scores[i])[::-1][:RECALL_N]
                order, _ = reranker.rerank(q, recall_idx, chunks)  # 不截断，保留 20 候选全排序
                out[i] = rerank_row(base_scores[i], order)
            return out

        rerank_hybrid = _rerank_on(hybrid_scores)
        rerank_bm25 = _rerank_on(bm25_scores)
        report(vector_scores, chunks, f"① 纯向量检索（{label}）", qa_set)
        report(bm25_scores, chunks, f"② 纯 BM25 检索（{label}）", qa_set)
        report(hybrid_scores, chunks, f"③ 混合检索 RRF（{label}）", qa_set)
        report(rerank_hybrid, chunks, f"④ 混合召回 + 重排（{label}）", qa_set)
        report(rerank_bm25, chunks, f"⑤ 纯 BM25 召回 + 重排（{label}）", qa_set)
        report_parent(rerank_hybrid, chunks, parents_section,
                      f"⑥ 混合+重排 + 父段落扩展/##节（{label}）", qa_set,
                      ceiling_note="粒度上限=##节（最近二级标题，生产选择）；父段落列含「返回更大上下文」的红利，小块列才是纯检索")
        report_parent(rerank_hybrid, chunks, parents_doc,
                      f"⑦ 混合+重排 + 父文档扩展/整篇（{label}）", qa_set,
                      ceiling_note="整篇=平凡天花板：整篇文档必含全部 gold，R@1≈1.00 不是检索排序质量，仅作上下文成本上界参考，勿用于生产")
        print_misses(qa_set, rerank_bm25, chunks)


if __name__ == "__main__":
    main()
