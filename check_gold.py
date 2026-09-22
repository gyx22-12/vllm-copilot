# -*- coding: utf-8 -*-
"""
gold 关键词防泄漏 + 可解性校验（评测集质量守门）。

评测判定命中 = top-k chunk 拼接后能凑齐 3 个 gold 词（子串匹配 kw in chunk）。
一道题能否测出「检索能力」取决于两个性质：

  1. 稀疏度（防泄漏）：每个 gold 词在语料里只能出现在极少数 chunk（理想 = 答案那 1 个）。
     若一个词散布在几十个 chunk，任何检索都会碰巧命中它 → 假 Recall，掩盖真实检索能力。

  2. 可解性（答案存在）：语料里必须真有一个 chunk（或至少同一文档）能同时含这 3 个词。
     否则题目「无解」，Recall 上不去是题目的锅，不是检索的锅。

check() 返回每题诊断；独立运行（py -3.12 check_gold.py）打印完整报告。
eval_qa.py 评测前会调用它做「体检」，发现坏题直接告警——防泄漏从「出题时手动跑一次」
升级成「评测的硬性前置」，不再靠「语料只加载 .md」这类隐式约定扛。
"""

import os
from collections import defaultdict

from rag_baseline import load_documents, chunk_markdown
from eval_data import QA_SET, QA_SEMANTIC
from config import CHUNK_SIZE  # 切块/检索共用常量单一来源（见 config.py）


def check(qa_set, chunks):
    """对每题返回诊断 dict 列表。

    chunks：[(文档路径, chunk文本), ...]，必须和评测用同一套切块（粒度一致才有意义）。

    harm 模型（对应评测判定「top-k 拼接后含全部 3 个 gold 词」）：
      🔴 无解（answer_docs 为空）：没有任何文档凑齐 3 词 → 题目不可能命中，必须修。
      🟡 多文档可解（answer_docs > 1）：多个文档都能凑齐 3 词 → gold 没钉住单一事实，
         检索到任一都算命中，会弱化区分度，建议收紧。
      🟡 单词跨文档（某词出现在 >1 文档）：只轻微抬高 nDCG 部分分；因为「全 3 词」的
         AND 条件里另外两词仍钉在答案文档，不产生假全命中，通常可接受。
      ⚠ query 含 gold 词（query_leaks）：题目自己就写明了答案关键词，字面检索直接命中，
         语义改写题退化成字面题——语义题必须避开，精确题也应尽量避开。
      同文档多 chunk（答案文档长、切成多块）本身不算泄漏，但意味着该题 Recall@1 恒 0、
      靠 @5/@10 + MRR/nDCG 分辨；chunk 越大越少发生（500→1000 后多数题已单块共现）。
    """
    texts = [c for _, c in chunks]
    by_doc = defaultdict(list)
    for src, t in chunks:
        by_doc[src].append(t)

    results = []
    for qa in qa_set:
        gold = qa["gold"]
        query_leaks = [kw for kw in gold if kw in qa["q"]]
        counts = {kw: sum(1 for t in texts if kw in t) for kw in gold}
        doc_hits = {kw: sorted({os.path.basename(s) for s, t in chunks if kw in t})
                    for kw in gold}
        # 答案文档 = 能同时凑齐 3 词的文档（正是评测判定的最小命中单元）。
        answer_docs = [os.path.basename(s) for s, ts in by_doc.items()
                       if all(kw in "\n".join(ts) for kw in gold)]
        single_chunk = any(all(kw in t for kw in gold) for t in texts)
        results.append({
            "q": qa["q"], "gold": gold, "counts": counts, "doc_hits": doc_hits,
            "answer_docs": answer_docs,
            "single_chunk": single_chunk, "query_leaks": query_leaks,
        })
    return results


def report(results):
    """打印可读报告。"""
    for r in results:
        print(f"\n• {r['q'][:72]}")
        for kw in r["gold"]:
            n = r["counts"][kw]
            n_docs = len(r["doc_hits"][kw])
            note = "  ⚠单词跨文档" if n_docs > 1 else ""
            print(f"    [{n:>3} chunk | {n_docs} 文档] {kw}{note}")
        problems = []
        if not r["answer_docs"]:
            problems.append("🔴 无解：没有任何文档凑齐 3 词")
        elif len(r["answer_docs"]) > 1:
            problems.append(f"🟡 多文档可解：{r['answer_docs']}")
        if r["query_leaks"]:
            problems.append(f"⚠query 含 gold 词 {r['query_leaks']}（语义题失效，改问法）")
        if not r["single_chunk"]:
            problems.append("Recall@1 恒 0（靠 @5/@10 分辨）")
        if not problems:
            problems.append(f"✓ 干净（答案文档 {r['answer_docs'][0]}）")
        print("    => " + " | ".join(problems))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    docs = load_documents([os.path.join(here, "corpus", "docs")], extensions=(".md",))
    chunks = []
    for src, text in docs:
        for c in chunk_markdown(text, max_chunk=CHUNK_SIZE):
            chunks.append((src, c))
    print(f"语料：{len(docs)} 文档 → {len(chunks)} chunk\n")
    for label, qa_set in (("精确词题 QA_SET", QA_SET), ("语义改写题 QA_SEMANTIC", QA_SEMANTIC)):
        print(f"===== {label} =====")
        report(check(qa_set, chunks))


if __name__ == "__main__":
    main()
