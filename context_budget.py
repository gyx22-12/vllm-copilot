# -*- coding: utf-8 -*-
"""共享「字符预算装箱」：按排名顺序把文本段装进 max_chars（截断装箱，单一来源）。

build_search（文档）/ build_search_code（源码）/ eval_qa 的预算消融 三处共用——
抽成单一来源，避免「N 份拷贝漂移」（改了一处忘了另一处，TOP_N 悄悄退化成 1 也发现不了）。

规则（截断装箱，而非贪心跳过）：
  - 整段装得下就整段装；装不下就把该段截到剩余预算再装、然后停（预算已满，更低排名段装不下）。
  - 为什么不用「跳过装不下的大段」：超长 top-1 往往是答案段，跳过等于把最相关的段整段丢掉、
    拿更低排名的小段顶替。实测（eval_qa 预算消融列）截断 R@10 0.86/0.67 vs 跳过 0.82/0.60。
  - 截断的段尾加「…（超出预算截断）」标记，让 LLM/人知道这段没给全。
  - 有段被截断时打一行 stderr（返回 N/M 段）观测，否则退化静默无从发现。
"""

import sys

_TRUNC_MARK = "\n…（超出预算截断）"


def pack_budget(items, max_chars, label="", quiet=False):
    """items: 已按排名排序的文本段（含出处头）。返回装下的段列表（预算内）。

    label: stderr 观测行的前缀；quiet=True 关掉观测（评测里每 QA 每 k 都调，避免刷屏）。
    """
    out, used = [], 0
    for it in items:
        if used + len(it) <= max_chars:
            out.append(it)
            used += len(it)
            continue
        room = max_chars - used
        out.append(it[: room - len(_TRUNC_MARK)] + _TRUNC_MARK if room > len(_TRUNC_MARK) else it[:room])
        break
    if not quiet and len(out) < len(items):
        print(f"[{label or 'pack_budget'}] 预算 {max_chars} 字：返回 {len(out)}/{len(items)} 段（末段截断）",
              file=sys.stderr)
    return out
