# -*- coding: utf-8 -*-
"""
Rerank（cross-encoder 重排）：把检索拆成"召回 + 精确打分"两阶段。

为什么需要它（对照 rag_baseline.py 里的 bi-encoder）：
  - 向量检索 / BM25 是"bi-encoder"式打分：query 和 doc 各自独立编码，再点积 / 词频统计。
    好处是快（所有 doc 向量可一次性预计算缓存），坏处是 query 和 doc 编码时"互不相见"，
    只在最后点一下，交互太粗。
  - cross-encoder 把 [query, doc] 拼成一条输入，一起过一遍 Transformer，query 和 doc 在
    同一个 attention 里逐 token 交互，输出一个 0~1 的相关度分数 —— 更准。
  - 代价：每一对 (query, doc) 都要完整 forward 一次，没法预计算，慢。
    所以只能当"第二道"：先用 bi-encoder 召回 top-N 候选，再只对这 N 个精确重排。

接口约定（这是面试要讲清的点）：rerank 只【重排】给定的候选，不新增、不删除 chunk，
所以它只改排名指标（nDCG / MRR），不改召回（Recall）—— 这正是实验 7 要验证的。
"""

import os

# 国内直连 huggingface 会超时，走镜像；必须在 import sentence_transformers 之前设置。
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from sentence_transformers import CrossEncoder


class Reranker:
    """cross-encoder 重排器。model_name 传 HF 模型名或本地路径。

    bge-reranker-base：英文 reranker，与语料（vLLM 英文文档）和 embedding
    （bge-base-en-v1.5）同属 bge 家族，sentence-transformers 原生支持，无需 FlagEmbedding。
    更强但需另装包的多语言版是 bge-reranker-v2-m3，原理一致，可无缝替换。
    """

    def __init__(self, model_name="BAAI/bge-reranker-base", batch_size=32):
        # CrossEncoder 内部已做 sigmoid 归一化，predict 返回 0~1 的相关度分数。
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def score(self, query, texts):
        """对 (query, 每个候选 text) 逐对打分，返回长度 = len(texts) 的分数数组。"""
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        return np.asarray(scores, dtype=np.float32)

    def rerank(self, query, candidate_idx, chunks, top_k=None, max_chars=None):
        """对候选 chunk 按 cross-encoder 分数重排。

        返回 (重排后的 chunk 下标数组, 对应分数数组)；top_k 不为 None 时截断。

        max_chars：打分前把候选文本截到 max_chars 字符。cross-encoder 长序列前向慢，
        而排名信号主要看开头（代码 chunk 的出处头 + 合成描述 + 函数签名都在最前面），
        截断几乎不损排序，却能省掉长序列前向。截断只影响打分，返回的还是原 chunk 下标
        ——上层拿到的仍是全文，出处/引用不受影响。
        """
        candidate_idx = np.asarray(candidate_idx)
        if len(candidate_idx) == 0:
            return candidate_idx, np.array([], dtype=np.float32)
        texts = [chunks[i][1] for i in candidate_idx]
        if max_chars:
            texts = [t[:max_chars] for t in texts]
        scores = self.score(query, texts)
        order = np.argsort(-scores)                 # 分数从高到低的下标
        idx = candidate_idx[order]
        if top_k is not None:
            idx = idx[:top_k]
        return idx, scores[order][: top_k if top_k else len(order)]


def rerank_row(base_scores_row, rerank_order):
    """把 rerank 后的顺序编码回一个 (n_chunks,) 的分数向量，供 recall/mrr/ndcg 复用。

    为什么这么做：评测函数（recall_at_k / mrr / ndcg_at_k）都吃"一整个分数矩阵，再
    argsort 取 top-k"。这里不重写评测，而是把 rerank 的结果伪装成一个分数行——
    rerank_order 里的 chunk 从高到低赋递减分，不在候选里的 chunk 赋 -inf（排最后）。
    于是 argsort(-row) 恰好 = rerank 的顺序，评测函数无感知地复用。
    """
    row = np.full_like(base_scores_row, -np.inf)
    n = len(rerank_order)
    for rank, idx in enumerate(rerank_order):
        row[idx] = n - rank   # 第 0 名给最高分 n，往后递减
    return row
