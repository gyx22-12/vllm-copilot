# review-probes —— 审查过程产生的探针与文档

这个目录不是项目源码的一部分，是**只读审查过程**留下的东西：诊断脚本、它们跑出的原始结果、以及几份审查/规格文档。放进来是为了让协作方（另一个会话 / 另一个 AI）能直接读到，不必在别处找。

生成日期：2026-09-22

---

## 一、重要使用须知（先读这段）

1. **脚本的运行路径已经写死**为 `C:\Users\GYX\rag-project`（`sys.path.insert` + `os.chdir`），所以从任何目录调用都能跑，不需要 cd。
2. **输出路径两种行为，注意区分**：
   - `_probe_recall_ab.py`、`_probe_git.py` 的 `OUT` 常量**写死指向原工作区** `C:\Users\GYX\WorkBuddy\2026-09-17-10-52-55\rag-review-scripts\` —— 重跑会写到那边，**不会覆盖本目录的 json**。
   - `_probe_paraphrase_ablation.py` 等用 `os.path.dirname(__file__)`，重跑会写到**本目录同名文件**，**会覆盖**。
   - 所以：**本目录的 json 是历史快照，重跑前先备份**。
3. 这些脚本**不烧 LLM API 的只有一部分**。需要 `get_client()` 的（要建索引的）会调 embedding 与合成描述 LLM；纯本地检索类（`_probe_paraphrase_ablation.py`）不烧 API。
4. 部分脚本需要环境变量 `DEEPSEEK_API_KEY`。

---

## 二、脚本与结果

| 脚本 | 结果 json | 干什么 | 关键结论 |
|---|---|---|---|
| `_probe_recall_ab.py` | `_recall_ab.json` | 6 种召回融合策略 A/B（S1~S6），其它一切固定（同索引、同重排器、同 top-10），只换召回 | 见下方「S1~S6 定义」与「6 格结果」 |
| `_probe_paraphrase_ablation.py` | `_paraphrase_ablation.json` | 8 道题把与 gold 共享词根的实词换成同义表达，重跑生产检索，看 gold 名次掉多少 | 量化出 R@1 里相当一部分来自**词形命中**而非语义理解 |
| `_probe_tokenize_ablation.py` | `_tokenize_ablation.json` | 分词消融（大小写归一、分词粒度对 BM25 的影响） | 对应「BM25 大小写未归一」这一项 |
| `_probe_paired_power.py` | `_paired_power.json` | 配对 McNemar 精确检验 + bootstrap CI + 功效估算，含**逐题名次** | 现有 n=18 下所有融合改动均不显著；按当前速率需 ≈50 题 |
| `_probe_desc_distance.py` | `_desc_distance.json` | 题面 vs 索引内合成描述的内容词重合度（Jaccard / 覆盖率 / 规避词数） | **档①题面的语域审计工具**，用来防止档①写成「贴 docstring 的题面」 |
| `_probe_stage_diag.py` | `_stage_diag.json` | 逐题分阶段诊断：gold 在「向量名次 / BM25 名次 / 融合名次 / 重排名次」各是多少 | **8/18 题所有策略全 miss**，瓶颈不在融合 |
| `_probe_recoverable_bridge.py` | `_recoverable_bridge.json` | 核查 18 道题的原始题面有多少能逐字复原 | 只有 **8/18** 能复原，另 10 道需重写 |
| `_probe_lexical_bridge.py` | `_lexical_bridge.json` | 词面桥梁分析（题面与 chunk 的稀有词重合） | 支撑「去同源去过头」的判断 |
| `_probe_sysexit.py` | （无 json，打印到 stdout） | P0-2 的最小复现：`sys.exit()` 击穿 harness | 证明 `except Exception` 接不住 SystemExit |
| `_probe_git.py` | `_git_probe.json` | git 仓库体量体检 | 决定 `.gitignore` 该忽略什么（corpus/src = 6817 文件 / 144MB） |

---

## 三、S1~S6 定义（源码见 `_probe_recall_ab.py:97-104`）

公共前置：

```python
rank_vec(x) = np.argsort(np.argsort(-x))          # 0-based 名次，0 = 分数最高
w           = np.where(overview_mask, OVERVIEW_WEIGHT, 1.0)
vec_all     = (idx.vecs @ qv) * w                 # 向量分，已乘 overview 权重
bm_all      = idx.bm25.scores(q) * w              # BM25 分，已乘 overview 权重
K           = 60
```

| 代号 | 名称 | 打分函数 | 召回深度 |
|---|---|---|---|
| S1 | 现行 RRF(1,1) | `1/(K+rank_vec(v)+1) + 1/(K+rank_vec(b)+1)` | 20 |
| S2 | 纯向量 | `v`（向量原始分） | 20 |
| S3 | 加权 RRF 0.85/0.15 | `0.85/(K+rank_vec(v)+1) + 0.15/(K+rank_vec(b)+1)` | 20 |
| S4 | 现行 RRF 但加深召回 | `1/(K+rank_vec(v)+1) + 1/(K+rank_vec(b)+1)` | **40** |
| S5 | 并列名次融合（取较好者） | `-np.minimum(rank_vec(v), rank_vec(b))` | 20 |
| S6 | 纯向量 + 加深召回 | `v`（向量原始分） | **40** |

三者关系：**S3 = S1 的加权版**（削弱 BM25 话语权）；**S4 = S1 只改召回深度**；**S6 = S2 只改召回深度**。
即 S3/S4/S6 分别是「换融合权重 / 换召回深度 / 向量+换召回深度」三个单变量对照。

**两处容易踩的实现细节**：

1. 召回阶段的 `recall_by(score, n, dedup=True)` 用 `_symbol_key` 对同一符号的多段**去重**后再取前 n。
2. 命中判定 `_has_symbol(texts[i], qa["gold"][0])` **只看 `gold[0]`**，不是全部 gold 词 —— 与文档评测「3 个关键词全中」的口径**不同**，读分时别混淆。

---

## 四、6 格结果（抽象档，n=18，历史快照）

| 策略 | R@1 | R@3 | R@5 | R@10 | MRR | 召回覆盖 |
|---|---|---|---|---|---|---|
| S1 现行 RRF(1,1) n20 | 0.167 | 0.222 | 0.278 | 0.444 | 0.232 | 10/18 |
| S2 纯向量 n20 | 0.111 | 0.333 | 0.333 | **0.556** | 0.236 | 13/18 |
| S3 加权 RRF 0.85/0.15 n20 | 0.111 | 0.278 | 0.389 | **0.556** | 0.237 | 12/18 |
| S4 现行 RRF n40 | 0.111 | 0.222 | 0.222 | 0.389 | 0.192 | 13/18 |
| S5 min-rank OR n20 | **0.222** | 0.222 | 0.333 | 0.500 | **0.271** | 12/18 |
| S6 纯向量 n40 | 0.111 | 0.222 | 0.389 | 0.444 | 0.200 | **14/18** |

读法（三条，别过度解读）：

- **R@10 上 S1 最低（0.444），S2/S3 最高（0.556）** → 在抽象档上，现行 RRF 确实把已经排好的向量信号稀释掉了。
- **R@1 上 S5 最高（0.222）**，但 S2/S3/S4/S6 都是 0.111 —— n=18 时 ±1 题 = ±0.056，这个差距在噪声范围内。
- **召回覆盖 S6 最高（14/18）**，但仍低于满值 → 说明有相当一部分题的 gold **连候选池都进不去**，换融合救不了。

---

## 五、文档

| 文件 | 内容 |
|---|---|
| `代码评测审查.md` | 源码检索 + 代码生成的评测审查（条目编号 P0-x / P1-x / P2-x） |
| `代码检索-两档测量规格.md` | 两档测量的完整规格：S5 定义与 4 个实现细节、2×6 矩阵、三层配对报表格式、三条口径声明、8/18 原料核查 |
| `代码检索分档方案.md` | A− 方案的裁决过程与规格 |
| `rag-project-落地顺序与两项决策.md` | 落地顺序、`sys.exit`/超时的子进程化规格、缓存键更正、10 项改动接口清单 |

---

## 六、这些结论的边界（诚实声明）

- 全部结果基于 **n=18** 的抽象档，**任何融合改动都不具备统计显著性**（详见 `_paired_power.json`）。
- `_recall_ab.json` 是**抽象档**结果，不是自然档，也不是最终能力分。
- 命中判定只看 `gold[0]`，与文档评测口径不同。
- 工作区那边可能还有更新版本的文件；本目录是 2026-09-22 的快照副本。
