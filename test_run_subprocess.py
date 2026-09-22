# -*- coding: utf-8 -*-
"""run_file 子进程化（P0-2）的冒烟测试：7 例，本地跑，不烧 LLM API。

验收的核心是 #2/#3 —— sys.exit 和死循环都必须只杀子进程、harness 存活。
import agent 只加载模块（不建索引、不调 get_client），所以这里不用 DEEPSEEK_API_KEY。

用法：
    py -3.12 test_run_subprocess.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402


def _write(name, content):
    """在 workspace/ 下写一个被测文件，返回相对路径。"""
    os.makedirs(agent._WRITE_ROOT, exist_ok=True)
    p = os.path.join(agent._WRITE_ROOT, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return name


def main():
    results = []

    def ok(cond, label):
        results.append(cond)
        print(("  ✓ " if cond else "  ✗ ") + label)
        return cond

    # 1. 普通输出
    r = agent.build_run_file(_write("smoke1.py", 'print("hi")\n'))
    ok("hi" in r and "运行失败" not in r, f"1 普通输出 → {r!r}")

    # 2. sys.exit(7)：run_ok=False、err=SystemExit:7、harness 存活（能打印出这条结果就证明没死）
    r = agent.build_run_file(_write("smoke2.py", 'import sys\nsys.exit(7)\n'))
    ok("SystemExit: 7" in r, f"2 sys.exit(7) → {r!r}")

    # 3. 死循环：临时把超时压到 2s，直接走 helper；返回 timed_out=True、harness 存活
    fd, tmp = tempfile.mkstemp(suffix=".py", prefix="smoke3_")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("while True:\n    pass\n")
    try:
        _, timed_out = agent._run_subprocess(tmp, None, timeout=2)
        ok(timed_out, f"3 死循环超时 → timed_out={timed_out}")
    finally:
        os.remove(tmp)

    # 4. 文件内 assert 失败：运行失败、AssertionError
    r = agent.build_run_file(_write("smoke4.py", 'assert 1 == 2\n'))
    ok("运行失败" in r and "AssertionError" in r, f"4 assert 失败 → {r!r}")

    # 5. __name__ == "__main__" 块要执行（证明 run_name 语义没变）
    r = agent.build_run_file(_write("smoke5.py", 'if __name__ == "__main__":\n    print("MAIN")\n'))
    ok("MAIN" in r, f"5 __main__ 语义 → {r!r}")

    # 6. 正确 add() + 通过断言
    r = agent.build_run_file(
        _write("smoke6.py", 'def add(a, b):\n    return a + b\n'),
        check="assert add(1, 2) == 3",
    )
    ok("check 通过" in r, f"6 check 通过 → {r!r}")

    # 7. 正确 add() + 错误断言：check 失败，且与 #4（运行失败）措辞可区分
    r = agent.build_run_file(
        _write("smoke7.py", 'def add(a, b):\n    return a + b\n'),
        check="assert add(1, 2) == 4",
    )
    ok("check 失败" in r and "AssertionError" in r, f"7 check 失败(≠运行失败) → {r!r}")

    print(f"\n{sum(results)}/7 通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
