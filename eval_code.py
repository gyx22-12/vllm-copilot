# -*- coding: utf-8 -*-
"""
代码检索评测（search_code 的 Recall@k）：把「源码检索到没」变成数字。

镜像 eval_qa.py 的 Recall@k / MRR / nDCG 三件套，但换语料、换 gold：
  - 语料 = 代码索引（CodeIndex，vLLM v1 投机解码 48 个 .py / 337 个符号 chunk）
  - gold = 实现该答案的「符号名」（类/函数/方法），不再是文档关键词
  - 命中 = top-k 符号 chunk 拼接后含全部 gold 符号名（整标识符匹配，见 _has_symbol）

一个关键差异（决定了怎么读 R@k）：
  文档 gold 是 3 个散在段落里的词，Recall@1 常为 0，靠 @5/@10 分辨；
  代码 chunk 是「自足的符号」（一个函数/类 = 一个 chunk），所以每题 gold 只钉 1 个
  符号、Recall@1 就有意义——R@1 是这里的主指标，@3/@5/@10 和 MRR/nDCG 用于分辨
  「命中了但排名靠后」的题。

gold 出题纪律（check_code_gold 的 harm 模型，同 check_gold）：
  - 可解：gold 符号必须在代码索引里（出现在 ≥1 chunk），否则题目无解。
  - 稀疏：gold 符号名要「唯一标识答案机制」。判据看「分散到几个文件」（n_files）而非
    chunk 数——类名会出现在「类概览 + 每个方法」chunk 的出处头里，这是正常放大（一个
    大类的 chunk 数可以到 20+，但 n_files=1~3，仍唯一）；真正危险的是 propose 这种
    十几个文件都有的通用名（n_files 爆炸）。>6 文件标 ⚠ 警告。
  - 不泄漏：query 不能含 gold 符号名字面，否则 BM25 字面命中，退化成「查名字」而非「查实现」。

整标识符匹配（_has_symbol）是这里和文档评测最关键的分工：文档 gold 是完整配置键/参数名
（子串碰撞少），代码 gold 是 Python 标识符，子串碰撞是常态——"MTPSpeculator" ⊂
"MultiModuleMTPSpeculator"。用 \b 式边界（前后不能紧跟 [A-Za-z0-9_]）匹配，防止把
「多模块 MTP」误算成「单模块 MTP」的命中。

检索调用对齐生产：CodeIndex.search() 含 module-overview 降权 + cross-encoder 重排
（agent.build_search_code 的同款检索段），top_k 取 10 只为测 @10，不改检索逻辑。

用法：
    $env:DEEPSEEK_API_KEY = "sk-你的key"
    py -3.12 eval_code.py
"""

import math
import os
import re
import numpy as np

# 国内直连 huggingface 会超时 + 已缓存模型时跳过联网检查（同 agent.py）。
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

from code_index import CodeIndex
from eval_data import CODE_QA_SET  # 题目 + gold 单一来源（见 eval_data.py）
from agent import get_client, _VLLM_ROOT, CODE_DIRS  # 复用生产加载路径（同 agent.load_index）

K_VALUES = (1, 3, 5, 10)
# gold 符号分散到 >6 个文件视为「太通用」（类 + 子类 + 工厂引用最多也就 3~4 个文件）。
COMMON_FILES = 6


def _has_symbol(text, symbol):
    """gold 符号名按「整标识符」匹配：前后不能紧跟 [A-Za-z0-9_]。

    防止 substring 误命中（代码标识符的常态碰撞）：
      "MTPSpeculator" 不命中 "MultiModuleMTPSpeculator"（前面是 'e'，无边界）
      "prepare_input_hidden_states_and_embeddings" 不命中
      "_prepare_input_hidden_states_and_embeddings_kernel"（前面是 '_'，无边界）
    """
    return re.search(r'(?<![A-Za-z0-9_])' + re.escape(symbol) + r'(?![A-Za-z0-9_])', text) is not None


def check_code_gold(qa_set, chunks):
    """gold 符号名体检：可解（存在）/ 稀疏（分散文件数）/ 不泄漏（query 不含 gold 字面）。"""
    rels = [rel for rel, _ in chunks]
    texts = [t for _, t in chunks]
    results = []
    for qa in qa_set:
        gold = qa["gold"]
        hit_files = {rel for kw in gold for rel, t in zip(rels, texts) if _has_symbol(t, kw)}
        n_chunks = {kw: sum(1 for t in texts if _has_symbol(t, kw)) for kw in gold}
        results.append({
            "q": qa["q"],
            "missing": [kw for kw in gold if n_chunks[kw] == 0],
            "leaks": [kw for kw in gold if kw in qa["q"]],
            "n_files": len(hit_files),
            "n_chunks": n_chunks,
        })
    return results


def check_query_discriminability(qa_set, embedder, threshold=0.80):
    """题面可区分性检查：任意两题题面 embedding 余弦相似度超阈值就报警。

    去同源去过头时（实验 18），两题的题面比答案还像——ngram「向后扫文本找可复用片段」和
    suffix「复用之前见过的序列」在向量空间里几乎重合，检索器再强也会把答案互相错位。
    这是出题问题不是检索问题，所以和 check_code_gold 一样在评测前当 harm 模型跑，把
    「题面撞车」抓出来逼你去补互相排斥的区分特征。返回 [(id_a, id_b, cos), ...] 按相似度降序。
    """
    from rag_baseline import QUERY_INSTRUCTION
    qs = [QUERY_INSTRUCTION + qa["q"] for qa in qa_set]
    vecs = embedder.encode(qs, normalize_embeddings=True, show_progress_bar=False)
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, 0.0)
    pairs = []
    for i in range(len(qa_set)):
        for j in range(i + 1, len(qa_set)):
            if sim[i, j] >= threshold:
                pairs.append((qa_set[i]["id"], qa_set[j]["id"], float(sim[i, j])))
    return sorted(pairs, key=lambda p: -p[2])


def recall_hits(ranked, gold, k):
    """top-k 拼接后是否凑齐全部 gold 符号（整标识符匹配，和文档同口径）。"""
    joined = "\n".join(t for _, t in ranked[:k])
    return all(_has_symbol(joined, kw) for kw in gold)


def recall_at_k(qa_set, ranked_all, k):
    return sum(1 for qa, ranked in zip(qa_set, ranked_all)
               if recall_hits(ranked, qa["gold"], k)) / len(qa_set)


def mrr(qa_set, ranked_all):
    """MRR：对每题找「首次凑齐全部 gold 符号」的排名 rank，取 1/rank 的均值。"""
    rrs = []
    for qa, ranked in zip(qa_set, ranked_all):
        seen, rank = set(), None
        for i, (_, t) in enumerate(ranked):
            seen |= {kw for kw in qa["gold"] if _has_symbol(t, kw)}
            if len(seen) == len(qa["gold"]):
                rank = i + 1
                break
        rrs.append(1.0 / rank if rank is not None else 0.0)
    return sum(rrs) / len(rrs)


def ndcg_at_k(qa_set, ranked_all, chunks, k=10):
    """nDCG@k：每块相关度 = 含几个 gold 符号，DCG 按 1/log2(rank+1) 折损，
    除以理想排序（把最高相关度块排最前，IDCG 来自全索引）归一化到 [0,1]。"""
    vals = []
    for qa, ranked in zip(qa_set, ranked_all):
        rels = [sum(1 for kw in qa["gold"] if _has_symbol(t, kw)) for _, t in ranked[:k]]
        dcg = sum(r / math.log2(j + 2) for j, r in enumerate(rels))
        ideal = sorted(
            (sum(1 for kw in qa["gold"] if _has_symbol(t, kw)) for _, t in chunks), reverse=True
        )[:k]
        idcg = sum(r / math.log2(j + 2) for j, r in enumerate(ideal))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(vals) / len(vals)


def print_misses(qa_set, ranked_all, k=10):
    """失败题诊断：对 Recall@k 没找全 gold 的题，打印缺失符号 + top-k 各自命中什么。"""
    print(f"\n===== 失败题诊断（Recall@{k} 没找全）=====")
    n_miss = 0
    for qa, ranked in zip(qa_set, ranked_all):
        joined = "\n".join(t for _, t in ranked[:k])
        if all(_has_symbol(joined, kw) for kw in qa["gold"]):
            continue
        n_miss += 1
        missing = [kw for kw in qa["gold"] if not _has_symbol(joined, kw)]
        print(f"\n❌ {qa['q'][:64]}")
        print(f"   缺失符号: {missing}")
        for rank, (_, t) in enumerate(ranked[:k], 1):
            header = t.split("\n", 1)[0]
            hits = [kw for kw in qa["gold"] if _has_symbol(t, kw)]
            mark = f"命中 {hits}: " if hits else "         "
            print(f"   top{rank:>2} {mark}{header}")
    if not n_miss:
        print("   无")


def main():
    client = get_client()
    idx = CodeIndex()
    n_files, n_chunks = idx.load(CODE_DIRS, synth=True, client=client, rel_root=_VLLM_ROOT)
    print(f"代码索引就绪：{n_files} 个 .py → {n_chunks} 个符号 chunk")
    chunks = idx.chunks

    # 评测前体检：无解 / 泄漏 = 硬失败；太通用 = 警告（同 eval_qa 的 check 体检）。
    rows = check_code_gold(CODE_QA_SET, chunks)
    hard = [r for r in rows if r["missing"] or r["leaks"]]
    if hard:
        print("\n🔴 代码评测集体检失败，先修再跑：")
        for r in hard:
            for kw in r["missing"]:
                print(f"  - 无解（符号不在索引）: {r['q'][:40]} → {kw}")
            for kw in r["leaks"]:
                print(f"  - 泄漏（query 含 gold 字面）: {r['q'][:40]} → {kw}")
        return

    print("✓ 代码评测集体检通过：每题 gold 符号都在索引里、query 不泄漏")
    print("  gold 稀疏度（n_files = 符号分散到几个文件）：")
    for r in rows:
        flag = "  ⚠ 太通用" if r["n_files"] > COMMON_FILES else ""
        detail = ", ".join(f"{kw}×{n}" for kw, n in r["n_chunks"].items())
        print(f"    n_files={r['n_files']:>2}  {r['q'][:40]}  [{detail}]{flag}")

    # 题面可区分性体检（实验 19）：两题题面太像 → 答案会互相错位，是出题问题不是检索问题。
    close_pairs = check_query_discriminability(CODE_QA_SET, idx._embed)
    if close_pairs:
        print("\n⚠ 题面撞车（两题题面相似度 ≥ 阈值，检索会把答案互相错排，先改题面再跑）：")
        for a, b, s in close_pairs:
            print(f"    {a:>16}  ↔  {b:<16}  cos={s:.3f}")
    else:
        print("✓ 题面可区分性通过：没有两题题面相似度超阈值")

    # 每题跑生产检索，取 top-10 排名（一次检索，@1/@3/@5/@10 从同一排名切）。
    ranked_all = [idx.search(qa["q"], top_k=10) for qa in CODE_QA_SET]

    print(f"\n===== search_code 检索 Recall@k（{n_chunks} 符号 chunk）=====")
    print("  k  |  Recall@k")
    print("-----+-----------")
    for k in K_VALUES:
        print(f" {k:>2}  |   {recall_at_k(CODE_QA_SET, ranked_all, k):.2f}")

    print(f" MRR = {mrr(CODE_QA_SET, ranked_all):.2f}    "
          f"nDCG@5 = {ndcg_at_k(CODE_QA_SET, ranked_all, chunks, 5):.2f}    "
          f"nDCG@10 = {ndcg_at_k(CODE_QA_SET, ranked_all, chunks, 10):.2f}")

    print("\n  每题命中矩阵（✓ = top-k 里找全 gold 符号）：")
    print("   " + "    ".join(f"k={k}" for k in K_VALUES) + "    gold → 问题")
    for qa, ranked in zip(CODE_QA_SET, ranked_all):
        marks = ["✓" if recall_hits(ranked, qa["gold"], k) else "✗" for k in K_VALUES]
        print("   " + "     ".join(marks) + "    " + " + ".join(qa["gold"]) + "  |  " + qa["q"][:46])

    print_misses(CODE_QA_SET, ranked_all)


if __name__ == "__main__":
    main()
