# -*- coding: utf-8 -*-
"""验证实验：给 BM25 的 tokenize 加「小写归一化 / 标识符拆词」后，
改写 query（用户不知道类名）的检索是否回升。

只 patch 我的探针进程里的 bm25.tokenize，不碰项目文件；embedding 只算一次，
BM25 索引按变体重建（tokenize 只影响 BM25 通道）。
"""
import json
import os
import re
import sys

RAG = r"C:\Users\GYX\rag-project"
sys.path.insert(0, RAG)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import bm25 as bm25_mod  # noqa: E402
from bm25 import BM25  # noqa: E402
from code_index import CodeIndex  # noqa: E402
from agent import get_client, _VLLM_ROOT, CODE_DIRS  # noqa: E402
from eval_code import _has_symbol  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tokenize_ablation.json")

PAIRS = [
    ("medusa",
     "Which proposer implements MEDUSA-style drafting using several classification heads stacked on the target's hidden states?",
     "Which component implements the multi-head-forecast drafting approach that stacks several classification heads on top of the main model's internal representations?",
     "MedusaProposer"),
    ("multi-mtp",
     "How does the multi-module multi-token-prediction path prepare per-module input hidden states and embeddings for several draft modules at once?",
     "How does the path that drives several drafting modules simultaneously build each module's starting activations and vector representations?",
     "prepare_input_hidden_states_and_embeddings"),
    ("eagle-spec",
     "Which speculator subclass specializes the autoregressive drafting path for EAGLE-style draft models?",
     "Which subclass of the drafting component specialises the one-token-at-a-time generation path for tree-attention-based assistant models?",
     "EagleSpeculator"),
    ("hidden-states-extract",
     "Which proposer extracts hidden states from the target model to feed a draft model for EAGLE-style speculative decoding?",
     "Which component pulls internal activations out of the main model so they can be handed to a smaller assistant model?",
     "ExtractHiddenStatesProposer"),
    ("rejection-sampler",
     "Which class verifies draft tokens against the target model's token probabilities, accepting or rejecting each one?",
     "Which class checks candidate tokens against the main model's probability distribution, keeping or discarding each one?",
     "RejectionSampler"),
    ("suffix-decode",
     "How does the suffix-matching proposal method look up a suffix of the current tokens against previously seen sequences to propose a continuation?",
     "How does the tail-matching proposal method look up the closing part of the current tokens against earlier-seen sequences to suggest what comes next?",
     "SuffixDecodingProposer"),
    ("adaptive-verify",
     "How does vLLM adaptively decide how many draft tokens to verify for each request based on measured per-step cost curves?",
     "How does vLLM dynamically decide how many tentative tokens to check for each request based on measured per-step cost curves?",
     "AdaptiveVerificationManager"),
    ("mtp",
     "How does the multi-token-prediction path reuse the target model's prefill step to seed the draft model's decode?",
     "How does the path that predicts several tokens ahead reuse the main model's prompt-processing step to seed the assistant model's generation?",
     "MTPSpeculator"),
]

CJK = r"[\u4e00-\u9fff]"


def tk_base(text):
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    tokens += re.findall(CJK, text)
    return tokens


def tk_lower(text):
    t = re.findall(r"[A-Za-z0-9]+", text.lower())
    t += re.findall(CJK, text)
    return t


def tk_split(text):
    """小写归一化 + 标识符拆词（camelCase/PascalCase/数字边界），原 token 同时保留。"""
    out = []
    for w in re.findall(r"[A-Za-z0-9]+", text):
        out.append(w.lower())
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", w)
        if len(parts) > 1:
            out += [p.lower() for p in parts]
    out += re.findall(CJK, text)
    return out


def rank_of(ranked, gold):
    for i, (_, t) in enumerate(ranked, 1):
        if _has_symbol(t, gold):
            return i
    return None


def summarize(rows):
    def agg(key):
        n = len(rows)
        mrr = sum((1.0 / r[key] if r[key] else 0.0) for r in rows) / n
        r1 = sum(1 for r in rows if r[key] == 1) / n
        r10 = sum(1 for r in rows if r[key]) / n
        return {"MRR": round(mrr, 3), "R@1": round(r1, 3), "R@10": round(r10, 3)}
    return {"orig": agg("orig_rank"), "para": agg("para_rank")}


def main():
    client = get_client()
    idx = CodeIndex()
    n_files, n_chunks = idx.load(CODE_DIRS, synth=True, client=client, rel_root=_VLLM_ROOT)
    print(f"index ready: {n_files} files / {n_chunks} chunks", flush=True)

    corpus = [c for _, c in idx.chunks]
    all_res = {}
    for name, tk in [("baseline", tk_base), ("lower", tk_lower), ("lower+split", tk_split)]:
        bm25_mod.tokenize = tk          # BM25.__init__ / scores 都是模块全局查找，patch 生效
        idx.bm25 = BM25(corpus)         # 只重建 BM25（embedding 不受影响，复用）
        rows = []
        for qid, orig, para, gold in PAIRS:
            rows.append({
                "id": qid, "gold": gold,
                "orig_rank": rank_of(idx.search(orig, top_k=10), gold),
                "para_rank": rank_of(idx.search(para, top_k=10), gold),
            })
        all_res[name] = {"summary": summarize(rows), "rows": rows}
        print(f"[{name}] {json.dumps(all_res[name]['summary'], ensure_ascii=False)}", flush=True)
        for r in rows:
            print(f"   {r['id']:<22} orig={r['orig_rank']}  para={r['para_rank']}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
