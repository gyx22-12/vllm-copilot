# -*- coding: utf-8 -*-
"""
代码索引（里程碑 5 第 1 步）：把 Python 源码按「符号」切块后建独立索引。

为什么和文档索引分开（不混进 agent.py 的 CHUNKS/CHUNK_VECS/BM25_IDX）：
  文档索引伺候「概念题」（LoRA / FP8 KV / preemption ...），代码索引伺候「实现题」
  （MTP 底层怎么调度）。混在一起，代码 chunk 会把那 5 道文档评测题污染掉——
  「What is the default preemption mode」会捞出 preemption 的源码实现而不是说明文档。
  分开 + 后面加 router（第 3 步），agent 才能按「问的是概念还是实现」选对索引。

复用已验证的两段式检索（和 agent.build_search 同构）：
  混合召回（向量+BM25→RRF）top-RECALL_N → cross-encoder 重排 top_k。

chunk 文本里已带 [source 路径:行号 | 类型 符号名] 出处头（code_chunker 产出），
所以检索结果直接可逐行 cite 到源码。

docstring 合成（synth=True）：vLLM 这类生产代码 docstring 稀疏，embedding 靠自然
语言匹配，没 docstring 的符号语义信号≈0，会被有 docstring 的特定模型反超。
用 LLM 给符号 chunk 合成一句英文描述补上语义锚点——见 docstring_synth.py。
"""

import os

# 国内直连 huggingface 会超时 + 已缓存模型时跳过联网检查（同 agent.py）。
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

import glob
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_baseline import embed, QUERY_INSTRUCTION
from bm25 import BM25, rrf_fusion
from rerank import Reranker
from code_chunker import chunk_python
from config import RECALL_N  # 召回候选数单一来源（见 config.py）

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_MAX_CHARS = 768   # 重排前把候选截到 768 字符（≈200 token，远低于 bge-reranker 512 上限）
OVERVIEW_WEIGHT = 0.3    # module overview（文件自述）的降权系数：代码索引面向「实现题」，
                         # 自述 docstring 字面和 query 对齐但信息密度最低，会压过真正的符号实现
MAX_CHUNK_CHARS = 1600   # 单个 chunk 嵌入前的字符上限。bge-base-en 的 512 token 上限 ≈ 2000 字，
                         # 超长符号（实测 342 块里 91 个 >2000 字，最长 11770）只会被静默编码前半截，
                         # 后半截检索时完全看不见还不报错。超了切成多段（见 _split_long_chunk）。
                         # 但 1600 按字符切仍挡不住 token 密度高的代码：measure_chunk_tokens.py 用 bge
                         # tokenizer 实测 523 块里 98 块（18.7%）切完仍 >512 token，被静默截断。
                         # 是否改按 token 切（MAX_CHUNK_TOKENS≈480）待定，测量见 chunk_tokens.json。


def _inject(text, desc):
    """把合成描述插到出处头之后、源码之前（作为检索锚点，不影响正文出处）。"""
    if not desc:
        return text
    header, _, rest = text.partition("\n")
    return header + "\n# " + desc + "\n" + rest


def _split_long_chunk(rel, text, max_chars=MAX_CHUNK_CHARS):
    """把超长的代码 chunk 按行切成 ≤max_chars 的多段，每段保留出处头 + [part i/n] 标记。

    背景：embedding（bge 512 token）上限约 2000 字，超长符号只编码前半截（静默截断，不报错）。
    切成多段后每段独立成向量，完整覆盖长函数/长类；出处头保留，检索结果仍能逐段 cite 到源码。
    """
    if len(text) <= max_chars:
        return [(rel, text)]
    lines = text.split("\n")
    header = lines[0]
    body = lines[1:]
    parts, cur = [], []
    for ln in body:
        cur.append(ln)
        if sum(len(x) for x in cur) + len(cur) >= max_chars:
            parts.append("\n".join(cur))
            cur = []
    if cur:
        parts.append("\n".join(cur))
    n = len(parts)
    return [(rel, f"{header} [part {i}/{n}]\n{p}") for i, p in enumerate(parts, 1)]


def _symbol_key(chunk_text):
    """chunk 的「符号身份」= 出处头去掉 [part i/n] 段号。

    超长符号被 _split_long_chunk 切成 N 段后，各段共享同一出处头（同路径:行号 + 同符号名），
    段号是唯一差异。用这个作合并键，让一个符号无论切成几段，检索时都只占一个候选名额——
    否则大函数被切成 8 段就占 8 个 top-k 格子，把小符号挤出去（实验 19 修的 R@10=0.33 主因之一）。
    """
    header = chunk_text.split("\n", 1)[0]
    return re.sub(r" \[part \d+/\d+\]$", "", header)


class CodeIndex:
    """一组 Python 源码文件的检索索引（embedding + BM25 + rerank）。"""

    def __init__(self):
        self.chunks = []      # [(相对路径, chunk 文本)]
        self.vecs = None      # (n_chunks, dim) 归一化向量
        self.bm25 = None      # BM25 打分器
        self._embed = None    # SentenceTransformer
        self._rerank = None   # cross-encoder

    def load(self, src_dirs, extensions=(".py",), max_files=None, synth=False, client=None,
             embed_model=None, reranker=None, rel_root=None):
        """加载目录下所有 .py，按符号切块，建索引。返回 (文件数, chunk数)。

        synth=True 时给「符号 chunk」（函数/类/方法）用 LLM 合成一句英文描述，
        补上 docstring 稀疏导致的语义缺口。client 需传 DeepSeek 客户端。
        embed_model / reranker 可传入外部已加载的模型（复用，省一份内存）。
        rel_root：chunk 出处头的路径相对这个根计算（而非相对 CWD）。docstring 合成
        缓存键 = chunk 内容哈希（含首行路径）——不传 rel_root 时换目录启动缓存全 miss，
        会重新烧一轮 LLM 合成。传源码根（如 vllm-0.29.0）让缓存键与启动目录解耦。
        """
        files = []
        for d in src_dirs:
            for p in glob.glob(os.path.join(d, "**", "*.py"), recursive=True):
                if "__pycache__" in p:
                    continue
                files.append(p)
        files = sorted(set(files))
        if max_files:
            files = files[:max_files]

        chunks = []
        types = []
        for p in files:
            try:
                with open(p, encoding="utf-8") as f:
                    src = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            rel = os.path.relpath(p, rel_root) if rel_root else os.path.relpath(p)
            # ↑ 相对 rel_root（源码根）而非 CWD：出处头/合成缓存键不随启动目录漂移
            for text, _line in chunk_python(rel, src):
                for rel2, text2 in _split_long_chunk(rel, text):
                    chunks.append((rel2, text2))
                    types.append("module overview" if "| module overview]" in text2 else "symbol")

        self.chunks = chunks
        self._overview_mask = np.array([t == "module overview" for t in types], dtype=bool)
        if synth:
            self._synthesize(client)
        self._embed = embed_model or SentenceTransformer(EMBED_MODEL)
        self.vecs = embed([c for _, c in self.chunks], self._embed)
        self.bm25 = BM25([c for _, c in self.chunks])
        self._rerank = reranker or Reranker(RERANK_MODEL)
        return len(files), len(self.chunks)

    def _synthesize(self, client):
        """给符号 chunk 合成描述并注入（module overview 已有 docstring，跳过）。"""
        from docstring_synth import synthesize
        texts = [t for _, t in self.chunks]
        targets = [i for i, t in enumerate(texts) if "| module overview]" not in t]
        try:
            descs = synthesize([texts[i] for i in targets], client)
        except Exception as e:
            # 失败隔离：LLM 合成是「锦上添花」的语义锚点，不是索引的必要条件。
            # 崩溃/限流时跳过合成、索引照常建，别让一个网络错误拖垮整个 load()。
            print(f"[code_index] docstring 合成失败，跳过（索引仍可用）：{type(e).__name__}: {e}")
            return
        for i, d in zip(targets, descs):
            rel, text = self.chunks[i]
            self.chunks[i] = (rel, _inject(text, d))

    def search(self, query, top_k=5):
        """两段式检索，返回 [(相对路径, chunk文本), ...]（按相关度降序）。"""
        q = embed([QUERY_INSTRUCTION + query], self._embed)[0]
        vec_scores = self.vecs @ q
        bm25_scores = self.bm25.scores(query)
        # 召回阶段降权：module overview 是「文件自述」非「实现」，压低向量分和字面分，
        # 避免 gemma4 这类自卖自夸的 docstring 在 top-20 召回里就挤掉符号实现。
        w = np.where(self._overview_mask, OVERVIEW_WEIGHT, 1.0)
        fused = rrf_fusion(vec_scores * w, bm25_scores * w)
        # 合并同符号多段（实验 19）：_split_long_chunk 把超长符号切成 N 段后，每段都是独立
        # 候选，大函数在 recall/top-k 里占 N 个格子、把小符号挤出去（SpecDecodeBaseProposer
        # 一个类的多段曾占 18 题 top-10 里 19% 的格子）。按符号身份去重、只留 fused 分最高的
        # 一段，让每个符号最多占一个名额——recall 和最终 top_k 都自然去重。
        order = np.argsort(fused)[::-1]
        seen, recall = set(), []
        for i in order:
            key = _symbol_key(self.chunks[i][1])
            if key in seen:
                continue
            seen.add(key)
            recall.append(i)
            if len(recall) >= RECALL_N:
                break
        recall = np.asarray(recall)
        # 重排阶段再降一次：cross-encoder 独立打分同样会被自述 docstring 带偏，
        # 所以 rerank 之后对 module overview 的分数再打折，再取 top_k。
        idx, scores = self._rerank.rerank(query, recall, self.chunks, top_k=None,
                                          max_chars=RERANK_MAX_CHARS)
        s = scores * np.where(self._overview_mask[idx], OVERVIEW_WEIGHT, 1.0)
        idx = idx[np.argsort(-s)][:top_k]
        return [self.chunks[i] for i in idx]


if __name__ == "__main__":
    # 自检：索引本项目自己的 .py，验证「加载→切块→embed→BM25→rerank→检索」管道不崩。
    # 注意：embedding 是英文模型，中文注释的源码检索语义不准，这里只验管道机制，不验质量。
    idx = CodeIndex()
    n_files, n_chunks = idx.load([os.path.dirname(os.path.abspath(__file__))])
    print(f"索引就绪：{n_files} 个 .py → {n_chunks} 个符号 chunk")

    q = "how does rerank work"
    print(f"\n查询：{q}\n")
    for rel, text in idx.search(q, top_k=3):
        header = text.split("\n", 1)[0]
        print(f"  {header}")
    print("\n（管道跑通即算自检通过）")
