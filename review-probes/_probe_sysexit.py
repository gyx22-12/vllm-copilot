"""机制验证：build_run_file 用 runpy 进程内执行 + `except Exception`，能否接住子文件里的 SystemExit？

等价复现 agent.build_run_file 的 try 结构（不改 rag-project 里任何文件）。
"""
import contextlib
import io
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(HERE, "_sys_exit_case.py")
with open(target, "w", encoding="utf-8") as f:
    f.write("import sys\nprint('before exit')\nsys.exit(7)\nprint('after exit')\n")

out_buf, err_buf = io.StringIO(), io.StringIO()
try:
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        ns = runpy.run_path(target, run_name="__main__")
except Exception as e:  # noqa: BLE001  ← agent.py 用的就是这个
    print("CAUGHT_BY_EXCEPTION:", type(e).__name__, e)
    sys.exit(0)
print("NO_EXCEPTION_RAISED — 正常返回", out_buf.getvalue().strip())
