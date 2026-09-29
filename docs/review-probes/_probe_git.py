# -*- coding: utf-8 -*-
"""git init 前的体量体检：哪些必须进 .gitignore。纯只读。"""
import json
import os

ROOT = r"C:\Users\GYX\rag-project"
OUT = r"C:\Users\GYX\WorkBuddy\2026-09-17-10-52-55\rag-review-scripts\_git_probe.json"


def dir_stat(p):
    n = 0
    total = 0
    for r, ds, fs in os.walk(p):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(r, f))
                n += 1
            except OSError:
                pass
    return n, total


res = {"gitignore_exists": os.path.exists(os.path.join(ROOT, ".gitignore")),
       "git_dir": os.path.exists(os.path.join(ROOT, ".git")),
       "dirs": {}, "big_files": [], "counts": {}}

for name in ["corpus", "corpus/docs", "corpus/src", "workspace", "mingpt", "__pycache__"]:
    p = os.path.join(ROOT, name)
    if os.path.exists(p):
        n, b = dir_stat(p)
        res["dirs"][name] = {"files": n, "bytes": b, "mb": round(b / 1048576, 1)}

top = []
n_top = 0
b_top = 0
for f in os.listdir(ROOT):
    fp = os.path.join(ROOT, f)
    if os.path.isfile(fp):
        s = os.path.getsize(fp)
        n_top += 1
        b_top += s
        top.append({"name": f, "kb": round(s / 1024, 1)})
res["counts"]["root_files"] = n_top
res["counts"]["root_bytes_mb"] = round(b_top / 1048576, 1)
res["big_files"] = sorted(top, key=lambda x: -x["kb"])[:20]

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("done")
