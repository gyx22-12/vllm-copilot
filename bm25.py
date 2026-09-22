# -*- coding: utf-8 -*-
"""
手写 BM25（TF-IDF 家族的检索打分公式）+ 倒数排名融合（RRF）。

BM25 和向量检索是两类互补的"相似度"：
  - 向量检索：把文本映射成语义向量，算"意思像不像"（"自动求导"和"autograd"靠得近）。
  - BM25：数"词重不重要、出现几次"，算"字面像不像"（稀有词"PagedAttention"一出现就高分）。
纯向量怕精确词，纯 BM25 不懂同义，所以混合检索（两者融合）互补。

只用标准库 + numpy，不依赖 rank_bm25 / jieba（手写才能讲清原理）。
"""

import re
import math
from collections import Counter

import numpy as np


def tokenize(text):
    """简易分词：英文/数字按词，中文按单字。

    生产环境中文会换 jieba 做词级分词（"解耦"是一个词，不是两个单字），
    这里用单字是为了保持零依赖、且原理可讲。
    """
    text = text.lower()  # 大小写归一化：BM25 是字面匹配，"PagedAttention" 和 "pagedattention"
                         # 是同一个词，不 lower 会被拆成两个 term、互相匹配不上
    tokens = re.findall(r"[a-z0-9]+", text)       # adamw、pagedattention、512
    tokens += re.findall(r"[一-鿿]", text)   # 中文逐字
    return tokens


class BM25:
    """BM25 打分器。构造时喂入语料（每篇是一个字符串），之后对 query 打分。

    BM25 三个旋钮：
      - TF 词频：query 词在文档里出现越多，分数越高（但有饱和度上限，靠 k1 控制）；
      - IDF 逆文档频率：词越稀有（出现的文档越少），区分度越高；
      - 长度归一化：文档越长，词频越被"稀释"，靠 b 控制。
    """

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(doc) for doc in corpus]
        self.N = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0
        # 文档频率 df：每个词出现在多少个文档里（一篇里重复出现只算一次）
        self.df = Counter()
        for doc in self.docs:
            for term in set(doc):
                self.df[term] += 1

    def _idf(self, term): #逆文档频率，词越稀有（n越小），IDF 越大，区分度越高。
        n = self.df.get(term, 0)
        # 平滑 IDF：ln((N - n + 0.5) / (n + 0.5) + 1)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def scores(self, query):
        """返回 query 对每个文档的 BM25 分数（numpy 数组，长度 = 文档数）。"""
        q_tokens = tokenize(query)
        out = np.zeros(self.N)
        for idx, doc in enumerate(self.docs):
            tf = Counter(doc)
            dl = self.doc_len[idx]
            s = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                f = tf[t]
                num = f * (self.k1 + 1)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self._idf(t) * num / denom
            out[idx] = s
        return out


def rrf_fusion(vec_scores, bm25_scores, k=60):
    """倒数排名融合（Reciprocal Rank Fusion）：合并两个 ranker，无需归一化分数。

    向量分和 BM25 分量纲不同（一个是余弦、一个是词频统计），直接加权会打架；
    RRF 只关心"排名"：每个文档的融合分 = 1/(k+向量排名) + 1/(k+BM25排名)。
    k 越大越"温和"（排名靠后的文档权重衰减越慢），60 是常用值。
    """
    vec_rank = np.argsort(np.argsort(-vec_scores))     # 0 基排名，0 = 最高
    bm25_rank = np.argsort(np.argsort(-bm25_scores))
    return 1.0 / (k + vec_rank + 1) + 1.0 / (k + bm25_rank + 1)
