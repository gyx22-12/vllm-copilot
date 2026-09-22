# -*- coding: utf-8 -*-
"""
docstring 合成：给无 docstring 的代码符号生成一句英文描述，作为向量检索的语义锚点。

背景：vLLM 这类生产代码 docstring 稀疏（其 AGENTS.md 明说「keep docstrings brief」），
embedding 靠自然语言匹配，没 docstring 的符号（如 class MTPSpeculator）语义信号≈0，
会被有 docstring 的特定模型（如 gemma4.py）反超。用 LLM 从「签名+函数体」推断一句话
描述补上锚点，通用实现就能被语义命中。

设计：
  - 批量合成（一次调 LLM 处理 batch_size 个符号），省时省 token。
  - temperature=0 保证可复现。
  - 结果按 chunk 内容哈希（剥行号区间）作键缓存到 code_synth_cache.json，重复建索引不重复
    调 LLM；符号体一被改写就重烧，行号漂移不触发 miss。
  - 描述只是「检索锚点」，不是最终答案的出处——最终答案仍逐行 cite 到真实源码，
    所以即使描述略有偏差也不污染忠实度（里程碑 4 的 judge 只看源码上下文）。
"""

import hashlib
import json
import os
import re

from agent import get_client  # 复用 DeepSeek 客户端（含 API key 检查）

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_synth_cache.json")
BATCH_SIZE = 15
TRUNCATE = 1200   # 每个符号喂给 LLM 的最多字符（签名+docstring+开头函数体足够判断用途）


def _chunk_key(text):
    """缓存键 = 整段 chunk 的内容哈希（先剥掉行号区间再哈希）。

    旧键只取出处头去行号，得到 [路径 | 类型 符号名]：非切分块的键里没有段号，符号体被
    改写（路径和符号名没变）时键不变 → 复用旧 desc，描述在讲一段已经改掉的代码。内容
    哈希让「正文一变就重建」；行号漂移仍不影响（:12-73 先剥掉再哈希，改注释挪行号不
    miss）。别把这个理由套到 _symbol_key 上——那里要的正是「路径 + 符号名」稳定（跨段去重
    靠它），不能用内容哈希替换。

    调用点在 desc 注入（_inject）之前，哈希里不含 desc，无循环依赖。
    """
    return hashlib.sha1(re.sub(r":\d+-\d+", "", text).encode("utf-8")).hexdigest()


def _load_cache():
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except OSError:
        pass  # 缓存写失败不该拖垮索引构建（失败隔离）


def _synthesize_batch(batch, client):
    """一次调 LLM 给 batch 里的符号各写一句英文描述。"""
    system = (
        "你是代码阅读助手。给定一组 Python 符号（函数/类/方法）的源码片段，"
        "每个片段以 [序号] 开头。为每个符号写一句英文描述（不超过 20 词），"
        "说明它做什么、在什么上下文中使用。只依据给出的源码，不要臆测没有的信息。"
        '只输出 JSON：{"descriptions": ["...", "..."]}，顺序与输入一致。'
    )
    items = [t[:TRUNCATE] for t in batch]
    user = "\n\n---\n\n".join(f"[{i}] {s}" for i, s in enumerate(items))
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        descs = json.loads(resp.choices[0].message.content).get("descriptions", [])
    except Exception:
        # 失败隔离：网络抖动 / 限流 / 非法 JSON 都不该让单个 batch 崩掉整个索引构建，
        # 返回空描述（对齐 batch 长度），上层拿到的仍是等长 list。
        descs = []
    # 对齐：数量不符时补空/截断，保证与 batch 等长
    descs = list(descs) + [""] * len(batch)
    return descs[:len(batch)]


def synthesize(chunks, client, batch_size=BATCH_SIZE):
    """给每个 chunk 生成一句英文描述，返回与 chunks 对齐的 list[str]（带缓存）。"""
    cache = _load_cache()
    keys = [_chunk_key(t) for t in chunks]
    descriptions = [cache.get(k) for k in keys]
    missing = [i for i, d in enumerate(descriptions) if not d]

    for start in range(0, len(missing), batch_size):
        idxs = missing[start:start + batch_size]
        descs = _synthesize_batch([chunks[i] for i in idxs], client)
        for i, d in zip(idxs, descs):
            if d:
                descriptions[i] = d
                cache[keys[i]] = d
    _save_cache(cache)
    return descriptions
