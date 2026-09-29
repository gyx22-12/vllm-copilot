# 落地顺序与两项决策

> 回答两个问题：**① 两档用 A 还是 B → 选 A（A−）**；**② run_file 超时现在做不做 → 现在做，且和 sys.exit 是同一刀。**
> 另附：你的 ①~⑤ 顺序有三处要改，以及我需要更正自己上一轮的一条建议。
> 全部结论基于源码核实 + 实测，`rag-project` 未改动。

---

## 决策一：两档选 **A**，具体是 A−

### 判 A 的三条理由

**① 你要的量是「配对差值」，B 求的是「两个独立集合的均值差」。**
档② 的产出不是第二个分数，是「题面换成自然语域后分数回升了多少」。这个量只有在**同一批题、同一套配置**下逐题配对才有意义。B 用 8 题独立集 × 均值差，会把「题目本身的难度」和「题面口径」混成一个数——而在 18 题规模下，前者比后者大得多。

**② B 的功效不足以回答它自己的问题（这条是决定性的）。**
档② 要测的效应量在 Δ0.05~0.11 之间（上一轮 6 组策略 A/B 的实测范围）。而 n=8 时 ±1 题 = **±0.125**，比效应量本身还大。也就是说 B 的分辨力**低于它要测的现象**——跑完也分不清 0.125 的差是真信号还是噪声。

| | n=8（B） | n=18（A−） |
|---|---|---|
| ±1 题引起的指标抖动 | ±0.125 | **±0.056** |
| 能否分辨 Δ0.056 的效应 | 不能 | 勉强（配对检验仍不显著，见下） |

顺带把上一轮的功效结论接过来：**即便 n=18，任何融合改动也只能算「方向性证据」**（McNemar 实测 p=0.50~1.00）。所以 A− 的价值不在「能定论」，而在「给出可解释的配对差 + 逐题名次，把效应量算准，为扩题到 40~50 题做准备」。

**③ B 的「省工作量」不成立。**
档② 那 18 题**已经写好、已经过两项体检**（gold 符号名字面 0 泄漏、题间可区分性 0 对超阈值）。B 只留 8 道 = 丢掉 10 道已完成的题，省不下任何工作时间，而那 10 道恰好是配对差的信息所在。

### 但 B 有一处是对的，要吸收

**别做成「并排两张分数表」。** 那样读者会以为有两个「能力分」。
A− 的报告主体应该是**配对表**：

| 表 | 内容 | 作用 |
|---|---|---|
| 表 1 | 档① 主指标（R@1/3/5/10 + MRR + nDCG） | 唯一的「能力分」 |
| 表 2 | 逐题配对 rank（档① vs 档②） | 主体。看每题被术语影响了多少名 |
| 表 3 | **术语依赖度** = 平均 ΔR@1、ΔR@10 | 档② 的真正产出 |

并挂一个**口径标记**：query ↔ 目标 chunk 那句 desc 的（词级）重合度。档② 实测 **3.8%**、7/18 题零重合；档① 随报。
**脱离这个数，0.17 和 0.44 都无法解读**——这是最省事也最防误读的一招。

### A− 的落地规格

- 18 题 × 2 版 = 36 个 query，**共享同一 gold**（`eval_data.py` 已钉死 `gold` 与 `answer_docs`，不用重做 ground truth）
- **档① 自然提问：新写**（保留 `draft model` / `speculative decoding` / `KV cache` 这类领域词）
- **档② 沿用现有 18 题原文，一个字不改**（它已过体检，是干净的对照组）
- 两档**同一套配置**（同一融合、同一重排、同一 RECALL_N）
- **档① 出题纪律**：先想「用过 vLLM 的人会怎么问」→ 写题面 → 事后只做两项检查（无 gold 符号名字面 / 题间不撞车）
  - **禁令：不要对着 desc 改题面。** 实验 18 就是这么写的（题面 ∩ desc 内容词重合度 0.038、平均规避 9.1 个 desc 内容词），所以那批题不是「公平题面」而是**对抗性题面**
  - 也**别回头对着档② 题面写反义词**——那样差值会被压平。从「问题意图 + 你自己的 vLLM 领域知识」写

### 预注册判据（写下来，跑完照它读，避免事后解释）

| 实测结果 | 结论 |
|---|---|
| 档① R@10 − 档② R@10 **≥ 0.11**（≥2 题） | 「BM25 在自然语域有信号」成立 → **RRF 保留**，P0-1 降级为「档② 的配置问题」 |
| 差值 **< 0.06** | 领域词不是主因，瓶颈在 embedding/索引/重排 → **融合之争降级**，别动融合 |
| 档① R@1 仍 **< 0.30** | R@1 不可用于「能定位实现」的表述，对外只用 R@10 / MRR |
| 任何情况 | **扩到 ≥40 题之前，不对融合做任何改动**（沿用上轮功效结论） |

---

## 决策二：`run_file` 超时 **现在就做**，而且和 sys.exit 是**同一刀**

### 先纠正提问里的一个分法

「P0-2（sys.exit）」和「run_file 超时」看起来是两件事，实际是**同一个缺陷的两面**：*运行器与被测代码共享同一个进程*。因为共享，所以被测代码既能（a）用 `sys.exit()` 杀掉运行器，也能（b）用死循环挂住运行器。修 (a) 的正确做法（子进程）自动修了 (b)。

### 我核到的实际风险面：**三个**，不是一个

| 位置 | 代码 | 后果 |
|---|---|---|
| `agent.py:264` | `runpy.run_path()` 包在 `except Exception` 里 | `SystemExit` 继承 `BaseException` → **接不住 → agent 进程当场退出** |
| `agent.py:274` | `exec(compile(check,...))` 包在 `except Exception` 里 | **同一个洞**：`check` 是 LLM 给的字符串，也可能带 `sys.exit()` |
| `eval_codegen.py:80` | `run_python_check` 里 `exec(compile(code,...))` + `except Exception` | **同一个洞，但这次死的是评测本身**：agent 写的代码一句 `sys.exit()` → 异常穿过 `eval_one` → 穿过 for 循环 → 穿过 `main()` → **进程退出，已跑的题全丢、汇总不打印** |

第三条你可能没注意：**现在 `eval_codegen.py` 里有两个独立的「exec 代码再 exec check」实现**（`agent.build_run_file` 一份、`eval_codegen.run_python_check` 一份）。而 `eval_codegen.py:153-157` 的修复轮 prompt 是把**同一个 `run_check` 字符串**交给 agent 当 `check` 用的——也就是说，agent 用 `run_file` 验证它以为的判据，而 harness 用另一套实现来判分。**两套实现一旦分叉（超时、隔离、命名空间），pass@1 就是用一个 agent 复现不了的路径算出来的。** 这是归因隐患，现在顺手合并掉。

### 为什么「一行 `except BaseException`」不是答案

它只能拦住 `SystemExit` 和 `KeyboardInterrupt`。拦不住：
- `os._exit(0)` —— 绕过一切 Python 异常处理
- `while True: pass` —— 没有异常可拦，就是挂着
- C 扩展段错误、内存爆掉 —— 进程直接没了

**能拦住这三样的只有进程边界。** 所以子进程是承重的部分，`except BaseException` 只是子进程内把 `sys.exit(7)` 翻译成一句可读回报的手段。

### 设计规格

**① 新常量**（与 `RUN_PYTHON_TIMEOUT = 10` 分开，`agent.py:54` 附近）
```python
RUN_FILE_TIMEOUT = 30   # run_file 跑的是完整脚本，比 run_python 的一段代码宽一些
```

**② 子进程驱动**（`runpy` + `check` 都搬进子进程，语义与现在完全一致，只换进程边界）
```python
# 子进程内执行的驱动，argv: [target_path, check, result_file]
try:
    ns = runpy.run_path(target, run_name="__main__")      # 保持 __main__ 语义
    run_ok = True
except BaseException as e:                                # SystemExit 也在这里
    run_ok, err = False, f"{type(e).__name__}: {e}"
if run_ok and check:
    try:
        exec(compile(check, "<check>", "exec"), ns)       # 与现在同一套命名空间语义
        check_ok = True
    except BaseException as e:
        check_ok, err = False, f"{type(e).__name__}: {e}"
# 结果写 result_file，不混进 stdout/stderr
```
**结果走临时文件，不走 stdout 哨兵**——否则被测文件自己打印一行 `__RESULT__` 就能污染解析。

**③ 父进程**
```python
subprocess.run([sys.executable, "-c", DRIVER, target, check or "", result_file],
               capture_output=True, text=True, timeout=RUN_FILE_TIMEOUT)
```
- `TimeoutExpired` → 返回**独立措辞**的「执行超时」消息
- `cwd` **保持项目根**（与现状一致）→ 这一刀只改一个变量，方便定位基线变化。想改成「文件所在目录」是对的，但留到下一次单独做
- `os._exit()` 依然逃得出 `except`，但**现在只杀子进程**——这正是要的

**④ `eval_codegen.run_python_check` 改为调用同一个 helper**，不要再留第二份 in-process `exec`。返回 `(check_ok, err)`，并新增 `timed_out` 字段。

**⑤ 汇总里 timeout 单列**：`pass@1` 的失败要能区分「写了死循环」和「逻辑写错」——这是两种不同的 agent 缺陷，混成一个 `✗` 就丢了诊断。

**⑥ 已知限制（现在不修，写进注释）**：`subprocess.run` 的超时只杀**直接子进程**，孙进程（`multiprocessing`）可能残留。runnable 的题是纯 Python，影响小；要严格就加 `CREATE_NEW_PROCESS_GROUP`（Windows）/ `start_new_session=True`。

### 动手前的冒烟测试（本地，不烧 API，7 例）

这 7 例才是 P0-2 的**验收标准**，尤其第 2、3 例是「harness 存活」的直接证明：

| # | 输入 | 期望 |
|---|---|---|
| 1 | `print("hi")` | run_ok，输出含 `hi` |
| 2 | `sys.exit(7)` | run_ok=False，err=`SystemExit: 7`，**harness 存活** |
| 3 | `while True: pass`（timeout 临时设 2s） | 「执行超时」，**harness 存活** |
| 4 | 文件里 `assert 1 == 2` | run_ok=False，AssertionError |
| 5 | `if __name__ == "__main__": print("MAIN")` | 输出含 `MAIN`（证明 `run_name` 语义没变） |
| 6 | 正确 `add()` + `check="assert add(1,2)==3"` | check_ok=True |
| 7 | 正确 `add()` + `check="assert add(1,2)==4"` | check_ok=False，**且与 #4 可区分**（一个是运行失败、一个是断言失败） |

### 为什么不是「先修 sys.exit、超时以后再补」

- 两次改动会**重写同一段代码**，而且中间那个状态（in-process + `except BaseException`）会给人「已经安全了」的错觉
- 更要紧的是时序：**②b 要赶在下一次跑 `eval_codegen` 基线之前完成**。否则你先拿到一个旧的 pass@1 基线，改完执行语义又得重跑一次——多一个基线，多一份混淆。现在 `answer_eval_results.json` 已删、`eval_codegen` 还没重跑，正是换轨的最佳窗口

---

## 对你 ①~⑤ 顺序的三处修正

### ① git init —— 必须一起写 `.gitignore`，否则 145MB 语料进库

实测体量（`_git_probe.json`）：

| 路径 | 文件数 | 大小 | 处理 |
|---|---|---|---|
| `corpus/src` | **6817** | **144.3 MB** | **忽略**（vLLM 0.29.0 源码，可重新下载，是依赖不是源码） |
| `corpus/docs` | 25 | **0.35 MB** | **必须保留**——25 篇官方文档是 `answer_eval` / `check_gold` 的 ground truth |
| 根目录 23 个文件 | 23 | 0.5 MB | 保留 |
| `workspace/` | 13 | 16 KB | 保留（agent 答卷） |
| `__pycache__` | 21 | 0.2 MB | 忽略 |
| `code_synth_cache.json` | 1 | **192 KB** | **建议提交**——重建要烧约 47 次 LLM 调用 |

最小可用 `.gitignore`：
```gitignore
corpus/src/
__pycache__/
*.pyc
```
忽略后仓库从 145MB 降到 **≈1MB**。

> **由此产生一个耦合（别忘）**：忽略 `corpus/src` → **必须在 README（第 ⑤ 步）写清怎么还原**（vLLM 0.29.0 源码放哪个路径），否则别人 clone 下来跑不了评测。① 和 ⑤ 是绑在一起的。

> **第一版提交就按现状提交，不要先清理。** 那些陈旧文件、临时脚本先在第一个 commit 里留个档，第 ② 步删掉时你才有一条可读的删除 diff。

### ② 拆成 ②a / ②b，两个 commit

**②a 纯 bug / 注释（低风险，实测过）**
- `P2-3` `eval_codegen.py:135` 清 workspace 不判类型 → 遇子目录 `IsADirectoryError`
- `P2-1` `MAX_CHUNK_CHARS` 1600 → 1200~1400（实测 **189/523 = 36% 超线**，最长 1803）
- `P2-1` 缓存键改成**内容哈希**（见下一节，这里要更正我上轮的建议）
- 陈旧注释：`337 个符号 chunk` → 523；`code_index.py:47` 那句「超长符号罕见」→ 实测 26.6% 超线

**②b `run_file` 子进程化**（= P0-2）+ 超时 + 合并 `eval_codegen.run_python_check`

**为什么必须分开**：②a 会改 chunk 数、会触发缓存重建；②b 会改执行语义。合成一个 commit，之后分数变化你无法归因。

### ④ 拆成 ④a / ④b

- **④a（随时可做，纯已有数据）**：`experiments.md` 里 dedup 那条归因——实验 19 把 `R@10 0.33→0.44` 记成「同符号多段合并的主因」，我做的 A/B 显示**只开关合并时 R@1/3/5/10/MRR 完全相同**（贡献 ≈ 0），那 +0.11 来自同时做的题面改写。**归因错了要早改**，不然被自己再引用一次就很难看。
- **④b（依赖 ③ 的数字）**：实验 18 那句「R@10 0.33 比 0.75 更低 → 真本事比温和改写暴露得还要弱」——**不成立**，0.75 vs 0.33 是题面口径差不是能力差。这条要等 ③ 的 Δ 数字出来才能改写准确。同时把实验 19 的「公平的代码检索能力 = R@1 0.17 / R@10 0.44」改成「**术语陌生用户提问下的下界**」。

### 修正后的顺序

```
①  git init + .gitignore（按现状首次提交）
②a 纯 bug / 注释 / 缓存键 / MAX_CHUNK_CHARS   ← 一次缓存重建解决两件事
②b run_file 子进程化 + 超时 + 合并 run_python_check（7 例冒烟测试先过）
③  A−：写 18 条档① 自然提问 → 跑两档 → 建配对报告
④a experiments.md 的 dedup 归因（可与 ② 并行）
④b 实验 18/19 口径重写（等 ③ 的 Δ）
⑤  hygiene：README（含 corpus/src 还原方法）+ requirements.txt + archive/ + 清残渣
```

---

## 我要更正自己上一轮的一条建议

上轮「P2-1」我写的是：**缓存键只留段号 `[part i]`，去掉 `/n`**。这条**是错的**，我当时的推理只看了「避免重烧」，没看后果。

核实后的实际结构：

```
code_index.py:79   header + f" [part {i}/{n}]"        ← 段的固定形式，第一行
docstring_synth.py:37  re.sub(r":\d+-\d+", "", header) ← 现在的键 = header 去行号，保留了 [part i/n]
```

三种键的对比：

| 键方案 | 改行号 | 改 `MAX_CHUNK_CHARS` | 结果 |
|---|---|---|---|
| 现状 `[part i/n]` | 不重建 ✓ | 105 个多段符号键变 → 重建 | 现状 |
| 我上轮说的 `[part i]` | 不重建 ✓ | **不重建，但复用了错误描述的缓存** ✗ | **有害**：段边界变了，同一键指向的正文不同，而 desc 生成时只看前 1200 字符（`TRUNCATE = 1200`）→ **缓存里的描述在讲另一段代码**。desc 是以 `# <desc>` 注入 chunk 正文的**唯一自然语言锚点**（`code_index.py:51-56`），描述错了直接伤向量检索 |
| **内容哈希**（推荐） | 不重建 ✓ | 重建 ✓（内容真的变了，该重建） | **正确** |

推荐写法：
```python
def _chunk_key(text):
    # 内容哈希：行号漂移不影响，正文一变就重建。desc 只在前 1200 字符上生成，
    # 段边界一动 desc 就该重烧——用段号当键会把旧描述贴到新正文上。
    return hashlib.sha1(re.sub(r":\d+-\d+", "", text).encode("utf-8")).hexdigest()
```
调用点没问题：`code_index.py:155` 传的是 `_inject` **之前**的原始 chunk（`_inject` 在 163 行才做），所以不会出现「哈希里含 desc」的循环依赖。

**成本实测**：`BATCH_SIZE = 15`，缓存约 700 条 → **一次全量重建约 47 次 LLM 调用**。很便宜。所以**别为了省缓存而牺牲键的正确性**——把键改成内容哈希、同时把 `MAX_CHUNK_CHARS` 一起改，一次重建两个问题一起解决。

**另外注意别改错函数**：`code_index.py:83` 的 `_symbol_key`（去 `[part i/n]` 求符号身份，给去重用）是**另一个函数，逻辑正确，不要动**。要改的是 `docstring_synth.py:30` 的 `_chunk_key`。

---

## 本轮改动清单（给 `rag-project` 的接口，不含代码）

| # | 文件 | 改动 | 依据 |
|---|---|---|---|
| 1 | `.gitignore`（新建） | 忽略 `corpus/src` / `__pycache__` / `*.pyc` | 实测 144.3MB / 6817 文件 |
| 2 | `agent.py` 常量区 | 新增 `RUN_FILE_TIMEOUT = 30` | 与 `RUN_PYTHON_TIMEOUT` 分开 |
| 3 | `agent.py:241-278` | `build_run_file` 换子进程驱动；`runpy`+`check` 都进子进程；结果走临时文件 | P0-2，三处同源洞之一 |
| 4 | `agent.py` 新增 helper | 供 `build_run_file` 与 `eval_codegen.run_python_check` 共用 | 消掉判分与工具的两套实现 |
| 5 | `eval_codegen.py:69-81` | `run_python_check` 改为调用 helper；返回带 `timed_out` | P0-2 第三处洞 |
| 6 | `eval_codegen.py` 汇总 | timeout 单列，不与断言失败混计 | 保住失败模式诊断 |
| 7 | `eval_codegen.py:135` | 清 workspace 时判类型 | P2-3 |
| 8 | `docstring_synth.py:30-37` | `_chunk_key` 改内容哈希 | 更正上轮建议 |
| 9 | `code_index.py:46` | `MAX_CHUNK_CHARS` 1600 → 1200~1400 | P2-1，实测 36% 超线 |
| 10 | `code_index.py:47`、`eval_data.py` 注释 | 陈旧注释改实测值 | 337 → 523；「罕见」→ 26.6% |

---

*本文档只做决策与规格，未改动 `rag-project` 任何文件。体量数据来自 `rag-review-scripts/_git_probe.json`（本轮新跑），机制结论来自源码核实（`agent.py:241-278`、`eval_codegen.py:69-81,135,153-157`、`code_index.py:46-56,79,83-90,151-163`、`docstring_synth.py:27-37,83-90`）。*
