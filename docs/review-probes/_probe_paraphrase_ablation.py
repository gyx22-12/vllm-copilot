# -*- coding: utf-8 -*-
"""消融实验：把 query 里与 gold 符号共享的词根换成同义表达，重跑生产检索，
看 gold 的排名掉多少 —— 量化「R@1 有多少来自词形命中，而非语义理解」。

纯本地检索（embedding + BM25 + cross-encoder rerank），不调 LLM API。
"""
import json
import os
import sys

RAG = r"C:\Users\GYX\rag-project"
sys.path.insert(0, RAG)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from code_index import CodeIndex  # noqa: E402
from agent import get_client, _VLLM_ROOT, CODE_DIRS  # noqa: E402
from eval_code import _has_symbol  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_paraphrase_ablation.json")

# (id, original query, 去词根改写 query, gold)
# 改写纪律：语义等价，但抹掉与 gold 符号共享的实词（专有名词/词根）→ 只留语义描述。
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


def rank_of(ranked, gold):
    for i, (_, t) in enumerate(ranked, 1):
        if _has_symbol(t, gold):
            return i
    return None


def main():
    client = get_client()
    idx = CodeIndex()
    n_files, n_chunks = idx.load(CODE_DIRS, synth=True, client=client, rel_root=_VLLM_ROOT)
    print(f"index ready: {n_files} files / {n_chunks} chunks", flush=True)

    rows = []
    for qid, orig, para, gold in PAIRS:
        r1 = idx.search(orig, top_k=10)
        r2 = idx.search(para, top_k=10)
        rows.append({
            "id": qid, "gold": gold,
            "orig_query": orig, "para_query": para,
            "orig_rank": rank_of(r1, gold),
            "para_rank": rank_of(r2, gold),
            "orig_top3_headers": [t.split("\n", 1)[0] for _, t in r1[:3]],
            "para_top3_headers": [t.split("\n", 1)[0] for _, t in r2[:3]],
        })
        print(f"{qid:<22} orig_rank={rows[-1]['orig_rank']}  para_rank={rows[-1]['para_rank']}", flush=True)

    # 汇总
    def mrr(key):
        vals = [(1.0 / r[key] if r[key] else 0.0) for r in rows]
        return sum(vals) / len(vals)

    def hit1(key):
        return sum(1 for r in rows if r[key] == 1) / len(rows)

    def hit10(key):
        return sum(1 for r in rows if r[key]) / len(rows)

    summary = {
        "n": len(rows),
        "orig": {"MRR": round(mrr("orig_rank"), 3), "R@1": round(hit1("orig_rank"), 3),
                 "R@10": round(hit10("orig_rank"), 3)},
        "para": {"MRR": round(mrr("para_rank"), 3), "R@1": round(hit1("para_rank"), 3),
                 "R@10": round(hit10("para_rank"), 3)},
        "dropped_to_none": [r["id"] for r in rows if r["orig_rank"] and not r["para_rank"]],
        "rank_worsened": [{"id": r["id"], "orig": r["orig_rank"], "para": r["para_rank"]}
                          for r in rows if r["orig_rank"] and r["para_rank"] and r["para_rank"] > r["orig_rank"]],
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
