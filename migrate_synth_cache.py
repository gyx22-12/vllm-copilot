# -*- coding: utf-8 -*-
"""一次性：把 code_synth_cache.json 从旧键（首行 header）迁移到新键（内容哈希）。

②a part 1 把 docstring_synth._chunk_key 改成内容哈希后，旧缓存的键全部失效，下次
load(synth=True) 会重烧全部 desc（~35 批 LLM 调用；即使 temperature=0 仍可能有微漂移，
让 ③ 的索引与历史抽象档数字不可比）。本脚本不烧 LLM：用旧键（header）从旧缓存取出 desc，
用新键（sha1 内容哈希）写回，desc 一字不动。

迁移前先备份旧缓存到 code_synth_cache.json.bak（可回退）。
用法：py -3.12 migrate_synth_cache.py
"""
import glob
import hashlib
import json
import os
import re
import shutil

from code_chunker import chunk_python
from code_index import _split_long_chunk
from agent import CODE_DIRS, _VLLM_ROOT
from transformers import AutoTokenizer

CACHE = "code_synth_cache.json"


def build_chunks():
    """复刻 CodeIndex.load 的切块循环（不含 embedding/BM25/rerank/synth），只产 chunk 文本。"""
    tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    files = []
    for d in CODE_DIRS:
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
        rel = os.path.relpath(p, _VLLM_ROOT)
        for text, _line in chunk_python(rel, src):
            for rel2, text2 in _split_long_chunk(rel, text, tok):
                chunks.append((rel2, text2))
    return chunks


def main():
    with open(CACHE, encoding="utf-8") as f:
        old = json.load(f)
    shutil.copy(CACHE, CACHE + ".bak")

    new, migrated, missing = {}, 0, 0
    for rel, text in build_chunks():
        if "| module overview]" in text:
            continue  # module overview 无合成 desc
        header = text.split("\n", 1)[0]
        old_key = re.sub(r":\d+-\d+", "", header)
        new_key = hashlib.sha1(re.sub(r":\d+-\d+", "", text).encode("utf-8")).hexdigest()
        if old_key in old:
            new[new_key] = old[old_key]
            migrated += 1
        else:
            missing += 1

    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=1)
    print(f"旧缓存 {len(old)} 条 → 新缓存 {len(new)} 条（迁移 {migrated}，旧缓存缺失待重烧 {missing}）")
    print("已备份旧缓存到 code_synth_cache.json.bak")


if __name__ == "__main__":
    main()
