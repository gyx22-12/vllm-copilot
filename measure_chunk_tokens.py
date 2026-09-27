# -*- coding: utf-8 -*-
"""②a-2 测量：用 bge 自己的 tokenizer 逐 chunk 数 token，验证按 token 切（MAX_CHUNK_TOKENS=460）
之后还有多少块顶到 bge 512-token 上限（会被静默截断，后半截检索时看不见）。

历史：原先 _split_long_chunk 按 1600 **字符**切，字符数换算 token 数误差大（代码 token 密度
高，1600 字符可能是 400 也可能是 800 token），实测 523 块里 98 块（18.7%）切完仍 >512 被静默
截断。改成按 token 切后，本脚本用来确认超线降到 0。

不烧 LLM、不加载 embedder 权重：只复刻 CodeIndex.load 的切块循环（chunk_python →
_split_long_chunk(tokenizer)），再用 AutoTokenizer 只加载 bge 的 tokenizer（vocab 级，无权重）。
注意：这里数的是「切块本体」（header+[part i/n]+body），不含 synth 注入的描述；460 已给描述
和 [CLS]/[SEP] 留了 ~52 的裕量。

输出：总块数、token 分布（含/不含 [CLS][SEP]）、超线块数与比例、最长块的出处头抽样。
用法：
    py -3.12 measure_chunk_tokens.py
"""

import glob
import json
import os

# 国内直连 huggingface 会超时 + 已缓存模型时跳过联网检查（同 agent.py）。
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from transformers import AutoTokenizer

from code_index import MAX_CHUNK_TOKENS, _split_long_chunk
from code_chunker import chunk_python
from agent import CODE_DIRS, _VLLM_ROOT

EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def build_chunks(src_dirs, rel_root, tokenizer):
    """复刻 CodeIndex.load 的切块循环（不含 embedding/BM25/rerank/synth），只产 chunk 文本。"""
    files = []
    for d in src_dirs:
        for p in glob.glob(os.path.join(d, "**", "*.py"), recursive=True):
            if "__pycache__" in p:
                continue
            files.append(p)
    files = sorted(set(files))
    chunks = []
    for p in files:
        try:
            with open(p, encoding="utf-8") as f:
                src = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        rel = os.path.relpath(p, rel_root) if rel_root else os.path.relpath(p)
        for text, _line in chunk_python(rel, src):
            for rel2, text2 in _split_long_chunk(rel, text, tokenizer):
                chunks.append((rel2, text2))
    return chunks


def main():
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    chunks = build_chunks(CODE_DIRS, _VLLM_ROOT, tok)

    full_lens = []     # 含 [CLS]/[SEP]：真正撞 512 墙的长度
    content_lens = []  # 不含特殊 token：纯正文长度
    for _, t in chunks:
        full_lens.append(len(tok.encode(t, add_special_tokens=True)))
        content_lens.append(len(tok.encode(t, add_special_tokens=False)))

    a = np.array(full_lens)
    b = np.array(content_lens)
    over_512 = int((a > 512).sum())    # 会被 bge 静默截断
    over_510 = int((b > 510).sum())    # 正文 >510 = [CLS]/[SEP] 都塞不下

    print(f"MAX_CHUNK_TOKENS = {MAX_CHUNK_TOKENS}   块数 = {len(chunks)}")
    print(f"token 分布（含 [CLS]/[SEP]，撞 512 墙的口径）：")
    for p in (0, 50, 90, 95, 99, 100):
        print(f"  p{p:>2} = {np.percentile(a, p):.0f}")
    print(f"  mean = {a.mean():.1f}")
    print(f"超线（>512，会被静默截断）      : {over_512}/{len(chunks)} = {over_512/len(chunks):.1%}")
    print(f"顶线（正文 >510，特殊 token 都塞不下）: {over_510}/{len(chunks)} = {over_510/len(chunks):.1%}")

    if over_512:
        idx = np.argsort(-a)[:10]
        print("\n最长 10 块（token 数 / 出处头）：")
        for i in idx:
            print(f"  {a[i]:>4}  {chunks[i][1].split(chr(10), 1)[0]}")

    summary = {
        "max_chunk_tokens": MAX_CHUNK_TOKENS,
        "n_chunks": len(chunks),
        "full_tokens_pct": {str(p): float(np.percentile(a, p)) for p in (0, 50, 90, 95, 99, 100)},
        "full_tokens_mean": float(a.mean()),
        "over_512": over_512,
        "over_512_ratio": over_512 / len(chunks),
        "over_510": over_510,
        "over_510_ratio": over_510 / len(chunks),
    }
    with open("chunk_tokens.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n已落盘 chunk_tokens.json")


if __name__ == "__main__":
    main()
