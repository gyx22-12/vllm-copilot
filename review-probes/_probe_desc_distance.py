# -*- coding: utf-8 -*-
"""题面 vs 索引自然语言锚点的「对抗距离」。

事实前提（已核实）：code_index._inject() 把 LLM 合成的一句描述以 `# <desc>` 插在
chunk 出处头之后、源码之前 —— 所以每个符号 chunk 里唯一的自然语言段落就是这句 desc，
它同时是 embedding 和 BM25 的语义锚点。

实验 18 的题面纪律是「避开 docstring/源码里的实现措辞」，而合成 desc 正是 docstring 同源物。
本脚本量化：现行题面与 desc 的内容词重合度 —— 越大说明题面越贴近索引语言；
接近 0 说明题面是「逐词反着写」的对抗性改写。
"""
import json
import os
import re
import sys
from collections import Counter

BASE = r"C:\Users\GYX\rag-project"
sys.path.insert(0, BASE)
from eval_data import CODE_QA_SET  # noqa: E402

OUT = r"C:\Users\GYX\WorkBuddy\2026-09-17-10-52-55\rag-review-scripts\_desc_distance.json"
TOK = re.compile(r"[a-z0-9]+")
STOP = set("""the a an of to in on for and or not is are was were be been being it its this that these those
as at by with from into over under out up down all any both each few more most other some such no nor only own
same so than too very can will just should now if then else when while where which who whom what how why
given does do did done has have had using use used
into based own with without within across during after before between
""".split())


def content(text):
    return [t for t in TOK.findall(text.lower()) if t not in STOP and len(t) > 2]


cache = json.load(open(os.path.join(BASE, "code_synth_cache.json"), encoding="utf-8"))

rows = []
for qa in CODE_QA_SET:
    g = qa["gold"][0]
    keys = [k for k in cache if f" {g}]" in k]
    if not keys:
        rows.append({"id": qa["id"], "gold": g, "found_desc": False})
        continue
    # 优先带行号的「符号本体」chunk，其次类概览
    specific = [k for k in keys if re.search(r":\d+-\d+", k)] or keys
    k = specific[0]
    desc = cache[k]
    q = qa["q"]

    qt, dt = set(content(q)), set(content(desc))
    shared = qt & dt
    # desc 里「被题面避开」的内容词：desc 有、题面没有
    avoided = dt - qt

    rows.append({
        "id": qa["id"], "gold": g, "found_desc": True,
        "desc": desc,
        "n_q_terms": len(qt), "n_desc_terms": len(dt),
        "n_shared": len(shared),
        "shared": sorted(shared),
        "jaccard": round(len(shared) / len(qt | dt), 4) if (qt | dt) else 0.0,
        "query_recall_of_desc": round(len(shared) / len(dt), 4) if dt else 0.0,
        "n_avoided": len(avoided),
        "avoided_sample": sorted(avoided)[:18],
    })

ok = [r for r in rows if r.get("found_desc")]
n = len(ok)
summ = {
    "n_questions": len(rows),
    "n_with_desc": n,
    "mean_jaccard": round(sum(r["jaccard"] for r in ok) / n, 4),
    "mean_query_recall_of_desc": round(sum(r["query_recall_of_desc"] for r in ok) / n, 4),
    "n_zero_shared": sum(1 for r in ok if r["n_shared"] == 0),
    "n_shared_le_1": sum(1 for r in ok if r["n_shared"] <= 1),
    "mean_avoided": round(sum(r["n_avoided"] for r in ok) / n, 1),
}
json.dump({"summary": summ, "rows": rows},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1), flush=True)
for r in ok:
    print(f"\n[{r['id']}] shared={r['n_shared']}/{r['n_desc_terms']}  avoided={r['n_avoided']}", flush=True)
    print(f"  desc : {r['desc'][:150]}", flush=True)
    print(f"  query: {[qa['q'][:150] for qa in CODE_QA_SET if qa['id'] == r['id']][0]}", flush=True)
    print(f"  被避开的 desc 词: {r['avoided_sample']}", flush=True)
print("done")
