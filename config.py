# -*- coding: utf-8 -*-
"""
检索/切块共用常量的单一来源。

这些值之前在 agent.py / eval_qa.py / check_gold.py / eval_rerank.py（及 code_index.py 的
RECALL_N）各有一份拷贝，靠注释「与 XX 一致」手工同步——结果 eval_rerank.py 漏改，
停在 CHUNK_SIZE=500 / TOP_N=50。这里收拢成一份，各处 import，改一次全局生效，
从根上杀掉「N 份拷贝漂移」这类 bug（用户 review 指出的 #1）。

只放「切块 + 检索」的数值常量。模型名（EMBED_MODEL/RERANK_MODEL）和 QUERY_INSTRUCTION
也在多处重复，但改动频率低，暂不纳入；要收再一并收进来。

用法：
    from config import CHUNK_SIZE, RECALL_N, TOP_N, MAX_CONTEXT_CHARS, MAX_CODE_CONTEXT_CHARS
"""

# 结构切块的最大字符数。500→1000 实测 Recall 全线大涨（chunk 散是主要瓶颈），锁定 1000。
CHUNK_SIZE = 1000

# 两段式检索：混合召回先粗筛出的候选数，交给 cross-encoder 重排（agent.build_search 与
# code_index.CodeIndex.search 同构，共用此值）。
RECALL_N = 20

# build_search 最终返回给 LLM 的 chunk/父段落数。
TOP_N = 5

# build_search 返回上下文的字符预算：父段落展开会把上下文撑大（eval_50 实测 ##节 top-10
# 均值 3.5 万字符、整篇 14 万），超预算按排名截断装箱（pack_budget）：整段装得下就装、
# 装不下截到剩余预算再装、停（截断保留段首答案，优于「跳过整段」——数据见 eval_qa 预算消融列）。可调。
MAX_CONTEXT_CHARS = 8000

# build_search_code 返回源码上下文的字符预算：代码 chunk（符号粒度+docstring 合成）也可能
# 很大（实测 top-5 曾到 1.8 万字符），同 build_search 按排名截断装箱（pack_budget）。代码密度高，预算比文档宽。
MAX_CODE_CONTEXT_CHARS = 12000
