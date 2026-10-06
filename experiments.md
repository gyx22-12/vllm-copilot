# 里程碑 2 · 消融实验记录

> 目标：把每次改动都用 Recall@k 量化，证明"为什么这么做有效"，而不是"我感觉"。
> 指标：Recall@k = 对 10 道评测题，检索 top-k 个 chunk，能找全每题 3 个答案关键词的题数占比。
> 语料：interview_qa.md + agent_roadmap.md（只加载 .md，排除 .py 代码）。
> 评测脚本：eval_qa.py（`py -3.12 eval_qa.py` 一键复现）。

## 实验 1：分块策略（固定字数 vs 按结构切块）

| 方案 | Recall@1 | @3 | @5 | @10 |
|---|---|---|---|---|
| 固定字数（500 字 / overlap 50） | 0.50 | 0.80 | 0.90 | 1.00 |
| 按结构切块（标题前缀 / max 500） | **0.80** | **1.00** | **1.00** | 1.00 |

### 结论
- 按结构切块把 Recall@1 从 0.50 → 0.80、Recall@3 从 0.80 → 1.00。
- **机制**：固定字数会在 500 字处硬切，把"问题标题"和"答案正文"劈开；长答案被切成的后续块里没有标题，成了"孤儿块"，检索时漏检（这正是里程碑 1 时 AdamW 题"答出来了但说资料截断"的根因）。
- 按结构切块在标题/段落边界切，并给每块挂上标题前缀，让长答案的每一块都"自报家门"，检索能命中尾块（含"解耦/二阶矩"）。
- 最典型案例：temperature 题从"k=5 都救不回"→"k=1 就命中"。

### 仍存的硬约束
- BPE 题（答案含表格，~600 字）超过 bge 的 512 token 上限，任何切块都装不进一块，Recall@1 对它天然到不了 1。
- 教训：Recall@1 的上限不是 1，取决于"答案是否比 embedding 窗口长"。

### 踩坑（写进 README 用）
1. **评测集关键词不能混进语料**：eval_qa.py 是 .py，若被当语料加载，答案关键词泄漏 → 分数虚高。所以语料只加载 .md。
2. **query 写法要贴近文档**："根号 d" 和符号 "√d" 在向量空间里不相等，会导致检索错位（本实验已把 query 改成"为什么除以 √d？"）。
3. **不要只看总分**：结构切块让总分涨了，但掩盖了 2 处局部回退（一处是 query 瑕疵、一处是答案超长硬约束）。逐题命中矩阵才能定位真凶。

---

## 实验 2：检索方法（纯向量 vs 手写 BM25 vs 混合 RRF）

| 方案 | Recall@1 | @3 | @5 | @10 |
|---|---|---|---|---|
| 纯向量（余弦点积） | 0.80 | 1.00 | 1.00 | 1.00 |
| 纯 BM25（手写） | 0.80 | 1.00 | 1.00 | 1.00 |
| 混合（RRF 倒数排名融合） | 0.80 | 1.00 | 1.00 | 1.00 |

### 结论：三种打平，不是失败，是两个真实结论
1. **指标饱和**：语料只有 54 个 chunk、题目是域内"为什么…"，实验 1 的结构切块已经把向量检索推到 1.00@3，Recall 到顶了，检索方法之间没有头寸。排名诊断（`_debug_hybrid.py`）显示三种方法 top-3 几乎相同，BM25 排到的也是同一批正确 chunk——说明 BM25 代码是对的，只是"找到正确 chunk"这件事向量已经做完了。
2. **剩下的 2 个 @1 失分是"答案跨 2 块"，检索救不了**：
   - AdamW 题：`解耦` 在 chunk#32、`二阶矩/动量` 在 chunk#31；
   - BPE 题：`合并表/OOV` 在 chunk#29、`OOV/byte-level` 在 chunk#30。
   - 检索只能把 1 个 chunk 排第一，关键词分在两块永远凑不齐 → Recall@1 数学上到不了 1。这是**切块 / embedding 窗口**约束，不是检索问题。

### 混合检索什么时候才"有用"（留给下一步）
- BM25 的强项是**精确稀有词**（API 名、版本号、报错信息）和 **OOD 改写**；当前"为什么…"域内题没有这两样，所以它 ≈ 纯向量。
- Recall@k 是"命中没命中"的二值，到顶后就看不出排名差别了——下一步要加 **MRR / nDCG** 这类排名指标。
- 要证明混合检索的价值，得让评测能"区分"检索器：换大而模糊的真实语料（vLLM/transformers 文档）＋精确词对抗题。

---

## 实验 3：换真实语料（vLLM 文档）后的三检索对比

语料：vLLM 官方文档 25 个核心文件（344KB，只挑 design/features/configuration 概念文档，跳过安装/部署流水账）→ **930 个 chunk**（结构切块 max=500）。
评测：10 道英文题，gold 全是精确技术词（环境变量/类名/参数名，如 `VLLM_ALLOW_RUNTIME_LORA_UPDATING`、`CudagraphDispatcher`），已用 `_check_gold.py` 校验每个词只出现在 1~2 个文件。

| 方案 | Recall@1 | @3 | @5 | @10 |
|---|---|---|---|---|
| 纯向量（bge-small-zh） | 0.20 | 0.30 | 0.30 | 0.50 |
| 纯 BM25（手写） | 0.20 | 0.40 | 0.40 | **0.80** |
| 混合（RRF 融合） | 0.20 | 0.40 | 0.60 | 0.70 |

### 三个真实结论
1. **指标不再饱和**：语料 54 → 930 chunk，top-k 从"覆盖 18% 语料"变成"1%"，Recall 掉到 0.2~0.8，检索方法终于有区分度了（实验 1/2 打平的问题解决了）。
2. **BM25 反超向量**：两个原因叠加——(a) 语料是英文，bge-small-zh 是中文优化模型，英文语义匹配弱；(b) gold 是精确技术词，正是 BM25（词频/精确匹配）的主场。**评测集怎么选 gold，就倾向哪边赢**，这是评测设计必须主动承认的偏置（不是 bug，真实 RAG 查询确实常带精确词）。
3. **朴素 RRF 混合反而拖后腿**（@10 0.70 < 纯 BM25 0.80）：当两个检索器一强一弱，RRF 把强检索器的排名被弱检索器稀释。**混合检索只在"互补且实力相近"时加分，不是万能。**

### 3 道"必挂题"是"答案跨块"，不是检索问题
诊断 `_debug_miss.py`：FP8 / disagg / preempt 三题的 3 个 gold 关键词散落在 2~6 个 chunk（如 `llm-compressor` 散在 chunk 855~870）。top-10 只占 930 块的 1%，凑不齐跨块关键词 → 必然挂。这是切块/embedding 窗口（max 500 字）约束，任何检索器都救不了。

## 实验 4：换英文 embedding（bge-base-en-v1.5）后的三检索对比

改动：embedding 从 `bge-small-zh-v1.5` 换成 `BAAI/bge-base-en-v1.5`（768 维、英文语义模型，query 加指令前缀 "Represent this sentence for searching relevant passages: "）。语料/切块/BM25/gold 全部不动，只换向量这一层。

| 方案 | Recall@1 | @3 | @5 | @10 |
|---|---|---|---|---|
| 纯向量（bge-base-en） | 0.20 | 0.30 | 0.40 | 0.60 |
| 纯 BM25（手写） | 0.20 | 0.40 | 0.40 | **0.80** |
| 混合（RRF 融合） | 0.20 | 0.40 | **0.60** | 0.70 |

（对比实验 3：纯向量 0.20/0.30/**0.30**/**0.50**，BM25 不变，混合不变）

### 三个结论
1. **换英文模型只修好了"英文向量弱"这一半**：向量 @5 0.30→0.40、@10 0.50→0.60，证明 bge-small-zh 的英文语义确实拖了后腿。但向量 @10（0.60）仍追不上 BM25（0.80）——说明"BM25 反超"还有另一半是**评测选词偏置**（gold 全是精确技术词，是 BM25 主场），光换模型补不掉。现在能给出定量比例：模型错配贡献约 +0.10，剩 0.20 的差距来自选词偏置。
2. **BM25 完全不变**（0.20/0.40/0.40/0.80）——它是纯词频统计，跟 embedding 模型无关。这条是 sanity check：数值复现 = 实验稳定，也证明向量那 +0.10 确实来自换模型、不是别的变量。
3. **朴素 RRF 的稀释问题，换更强向量也没修好**：hybrid @10 0.70 仍 < BM25 0.80。但看全 k 会发现更细的规律——hybrid @5（0.60）> BM25 @5（0.40）> vector @5（0.40）：**RRF 在小 k 是真赢、在大 k 才被稀释**。机制：小 k 时两个检索器捞到的正确 chunk 互补（并集更大）；大 k 时 BM25 的精确词长尾（第 6~10 名）才是最佳来源，被向量排进来的"噪声 chunk"挤出去 → 掉 1 题。稀释不是"向量不够强"，是"无脑按排名融合"的毛病，得换加权融合或两阶段。

### 逐题命中里的"硬地板"
- 三种方法共同必挂只剩 2 题：**FP8（Q2）、disagg（Q3）**——答案跨 2~6 块，top-10 只占 930 块的 1%，凑不齐，这是切块/窗口约束，检索器救不了。
- **mm 题（Q10）**是"向量救不回、BM25 @10 能中"：gold 是 `Dummy Input Text`/`PromptUpdate` 这类精确多词短语，向量切不准、BM25 精确匹配主场——又一个"精确词偏置"的实锤。
- preempt 题（Q8）三法 @10 都能凑齐，不是必挂题（跨块跨度比 FP8/disagg 小）。

### 下一步
- 修"稀释"：加权 RRF / 两阶段（BM25 召回 + 向量或 cross-encoder 重排）。里程碑 2 的最后一项 rerank 就在这条线上。
- 修"跨块硬地板"（FP8/disagg）：换长窗口 embedding（bge-m3 8192 token）或 parent-child 层级切块，让答案段装进 1 块。

---

## 实验 5：语义改写题 A/B —— 证明"BM25 领先"是评测偏置



设计：从 10 道精确词题里挑 4 道"向量本来就能命中"的（LoRA/推理/cudagraph/prefix），把 query 改写成同义表达、刻意去掉 gold 精确词（如 "load/unload LoRA adapters at runtime" → "swap in or remove fine-tuned adapters on a live server"），gold 完全不变。语料/切块/模型/检索全部不动，只换 query 写法。

| 题目类型 | 方法 | @1 | @3 | @5 | @10 |
|---|---|---|---|---|---|
| 精确词题 | 纯向量 | 0.20 | 0.30 | 0.40 | 0.60 |
| 精确词题 | 纯 BM25 | 0.20 | 0.40 | 0.40 | **0.80** |
| 精确词题 | 混合 RRF | 0.20 | 0.40 | **0.60** | 0.70 |
| 语义改写题 | 纯向量 | 0.00 | 0.25 | 0.25 | **0.50** |
| 语义改写题 | 纯 BM25 | 0.00 | 0.00 | 0.00 | 0.25 |
| 语义改写题 | 混合 RRF | 0.00 | 0.25 | 0.25 | 0.50 |

### 结论
1. **赢家随题目类型翻转**：精确词题 BM25 0.80 > 向量 0.60；语义题向量 0.50 > BM25 0.25。同样的检索器、同样的语料，只换 query 写法赢家就互换——直接证明实验 3/4 的"BM25 反超"主要是**评测选词偏置**，不是 BM25 内在更强。
2. **BM25 崩得比向量狠**：精确词→语义，BM25 @10 0.80→0.25（-0.55），向量 0.60→0.50（只 -0.10）。本质区别：**BM25 无语义兜底、向量有**——query 里没有精确词，BM25 就没了抓手。
3. **别过度吹向量**：语义题向量也只有 0.50（4 中 2）。cudagraph、prefix 两题改写后两法都挂——把高度技术概念翻译成大白话后，连向量也丢信号。准确说法是"BM25 语义更弱"，不是"向量语义超强"。
4. **混合在语义题上 == 纯向量**（0.50）：弱的 BM25 被 RRF 压低，融合退化回纯向量。再次印证"朴素 RRF 只会稀释、不会凭空增益"。

### 逐题（语义改写题）
- LoRA：向量 @10 中、BM25 全程 ✗——改写后 BM25 彻底瞎，向量靠"swap/remove ≈ load/unload"摸到。
- reasoning：向量 @3 中、BM25 @10 才中——向量更快。
- cudagraph、prefix：双挂——改写太"白"、概念太专。

### 对面试的价值
现在能给出**定量、可控**的证据链："BM25 擅长精确词查找、向量擅长语义改写、混合在精确词 @5 处互补"——这是工程师最想要的"在什么条件下用谁"，而不是一句"BM25 更好"。

---

## 实验 6：加排名指标 MRR / nDCG —— Recall@10 和 nDCG@10 打架了

动机：前面三次撞到 Recall@k 是"进没进 top-k"的二值，看不到排名质量；而 rerank 只改排名不改召回，必须用排名指标才能测出它。给 eval 加 **MRR**（首次凑齐 3 词排名的倒数）和 **nDCG@k**（每块含几个关键词做分级相关度，除以理想排序归一化）。

| 题目类型 | 方法 | Recall@10 | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| 精确词题 | 纯向量 | 0.60 | 0.31 | 0.46 | 0.55 |
| 精确词题 | 纯 BM25 | **0.80** | 0.36 | 0.55 | 0.60 |
| 精确词题 | 混合 RRF | 0.70 | 0.36 | **0.60** | **0.63** |
| 语义改写题 | 纯向量 | 0.50 | 0.16 | 0.20 | 0.26 |
| 语义改写题 | 纯 BM25 | 0.25 | 0.06 | 0.19 | 0.26 |
| 语义改写题 | 混合 RRF | 0.50 | 0.16 | **0.27** | **0.33** |

### 核心发现：同一个"精确词题"，不同指标说谁是赢家不一样
- **Recall@10 说 BM25 赢**（0.80 > 混合 0.70）：BM25 能在 top-10 里凑齐更多题的完整 3 词。
- **nDCG@5/10 说混合赢**（0.60/0.63 > BM25 0.55/0.60）：混合把相关块排得更靠前。
- 不矛盾——**两个指标测的东西不同**：Recall 是"全不全"（3 词是否都在 top-10），nDCG 是"靠不靠前"（相关块排第几，给部分分）。混合把最相关的块顶到最前（nDCG 高），却把"第三个关键词"挤出了 top-10（Recall 掉）——**完整性换排名质量**。

### 顺带修正实验 4 的"稀释"说法
- 之前说"朴素 RRF 被弱检索器稀释"，现在看太糙。精确说法：**RRF 不是在稀释，是在拿 Recall@10 换 nDCG**——排名质量上升、完整命中率下降。选哪个看下游：LLM 需要证据凑齐 → 看 Recall；LLM 能靠最相关一块作答 → 看 nDCG/MRR。

### 语义题上排名指标也分离
- **MRR**：向量 0.16 vs BM25 0.06（2.7 倍），比 Recall（0.50 vs 0.25）更锋利地显示向量的语义优势。
- **nDCG@10**：向量 0.26 = BM25 0.26 打平（BM25 靠"散落关键词"拿部分分），但混合 0.33 明显最高——语义题上混合的 nDCG 是最好的。

### 下一步：上 rerank（cross-encoder）
- 现在有 nDCG 这把尺子了，rerank 的收益（纯改排名）终于能被测出来。里程碑 2 最后一项，可以动手。

---

## 实验 7：接入 rerank（cross-encoder）—— keyword 指标测不出 rerank，反而暴露了"指标错配"

改动：新增 `rerank.py`（Reranker 类，封装 `CrossEncoder("BAAI/bge-reranker-base")`）+ `eval_rerank.py`（两阶段：召回 top-50 → cross-encoder 重排 → 评测取 top-k）。语料/切块/检索全部不动，只加"第二道重排"。**一句话原理**：bi-encoder 各自独立编码（快、可预计算），cross-encoder 把 [query, doc] 一起喂进 Transformer 逐 token 交互（准、但慢），所以只对 top-N 候选重排。

| 题目类型 | 方法 | 指标 | 召回(前) | rerank(后) |
|---|---|---|---|---|
| 精确词题 | 向量 | nDCG@5 | 0.46 | **0.56** |
| 精确词题 | 向量 | nDCG@10 | 0.55 | **0.58** |
| 精确词题 | 向量 | Recall@10 | 0.60 | 0.60 |
| 精确词题 | BM25 | nDCG@5 | 0.55 | 0.57 |
| 精确词题 | BM25 | nDCG@10 | 0.60 | 0.58 |
| 精确词题 | BM25 | Recall@10 | 0.80 | **0.60** ⚠️ |
| 精确词题 | 混合 | nDCG@10 | 0.63 | 0.59 |
| 精确词题 | 混合 | Recall@10 | 0.70 | 0.60 |
| 语义改写题 | 向量 | Recall@10 | 0.50 | 0.25 |
| 语义改写题 | BM25 | nDCG@5 | 0.19 | **0.29** |
| 语义改写题 | BM25 | Recall@10 | 0.25 | **0.50** |
| 语义改写题 | 混合 | Recall@10 | 0.50 | 0.25 |

### 结论 1（纠错）：我之前说"rerank 不改 Recall"是错的
精确说法：rerank **只重排 top-N 候选池、不增删 chunk**，所以 **Recall@N(=50) 不变**；但对 **k < N**，rerank 重排了池子内部顺序，Recall@k **会变、可升可降**。实验里 BM25 精确词题 Recall@10 从 0.80 → 0.60，就是被重排"挤出 top-10"。这句话以后面试别再说错。

### 结论 2（核心发现）：keyword 指标与 rerank 的目标【错配】
逐块 debug（`_debug_rerank.py`，LoRA 题）证明 **reranker 是"对"的**：它把真答案 `### Using API Endpoints / Loading a LoRA Adapter: To dynamically load` 从第 5 名提到第 2~3 名，把备选方案 `### Using Plugins / LoRAResolver` 从第 3~4 名压到第 8~9 名。但 gold 关键词 `LoRAResolver` 恰好住在被压下去的"Plugins"块里，于是 keyword-Recall 掉了。
- 本质：**rerank 优化"语义相关"，keyword 指标奖励"字面覆盖"**，两者会打架。gold 关键词是"散落的 magic string"，被 cross-encoder 合理地当成了次要信息 → 指标把它记成失分。
- 不是 reranker 弱（它排序判断是对的），是**离线 keyword 指标已经测不出 rerank 的真实价值**。

### 结论 3：rerank 干净的赢面在两处（"在什么条件下用谁"）
- **弱检索器上**：向量 nDCG@5 0.46 → 0.56（+0.10）。向量召回的相关块排得靠后，rerank 把它们顶到前面——这是 rerank 的经典主场。
- **语义改写题 + BM25 上**：BM25 Recall@10 0.25 → 0.50、nDCG@5 0.19 → 0.29。改写 query 里没有精确词，BM25 字面匹配瞎掉，但正确块还在 top-50 里（被埋住），cross-encoder 靠语义把它捞回。**rerank 救活了无语义兜底的 BM25**。
- 对照：语义改写题上向量/hybrid 的 Recall 反而降（0.50→0.25）——它们的语义排序本来已经不错，rerank 再按语义重排，和"keyword gold"更不对齐。

### 结论 4：下一步必须是 LLM-as-judge
keyword 指标（Recall/nDCG）本质奖励"精确字符串凑齐"，天生偏 BM25、也天生测不准 rerank（实验 5 已证偏 BM25，本实验再证测不准 rerank）。要证明 rerank 的**端到端**价值（答案更相关、更忠实），得换 **LLM-as-judge** 对最终答案打分——这正好接里程碑 4 的"忠实度评测"。在换之前，别再用 keyword 指标给 rerank 下"没用"的结论。

---

## 实验 8（bug 修复）：代码块里的 `#` 注释被误判成标题

问题：`_split_by_heading` 把所有 `#` 开头行都当 markdown 标题。vLLM 文档代码块里满是 `#` 注释
（Python/bash/yaml），导致代码块被切碎、还凭空多出 `# config.py` 这类"假标题前缀"污染 chunk。

修复：追踪代码块状态（``` ... ``` 翻转 `in_code_block`），只在非代码块内识别标题；fence 行保留在 body 里，保证代码块在 chunk 里仍闭合。

影响（重跑 eval_qa.py，语料 25 篇）：
- chunk 数 930 → **903**（少了 27 个被假标题切出来的碎块）
- 向量 精确词 Recall@10 0.60 → **0.70**（+0.10）
- 混合 语义改写题 MRR 0.16 → **0.29**、nDCG@5 0.27 → **0.32**、Recall@1 0.00 → **0.25**
- BM25 精确词不变（0.80）——它靠词频，切块边界对它影响小
- 唯一微降：混合 精确词 MRR 0.36 → 0.34（噪声级）

说明：实验 3~7 的记录用的是修复前的 chunker（930 chunk）。修复只让分数小幅上移、不改变任何结论
（混合检索/rerank 的机制性结论与具体切块边界无关）。

---

## 实验 9：RRF 的 k 值消融 —— k=60 落在稳健区（null result）

动机：RRF 的 `k` 是"排名权重衰减"旋钮，60 是论文常用值但没验证过。扫 k ∈ {5,10,20,30,60,100,200}（`_sweep_rrf_k.py`）。

| k | 精确词 Recall@10 | 语义题 Recall@10 | 语义题 nDCG@5 |
|---|---|---|---|
| 5 | 0.70 | 0.25 ⚠️ | 0.38 |
| 10 | 0.70 | 0.50 | 0.35 |
| 20~60 | 0.70 | 0.50 | 0.32 |
| 100~200 | 0.70 | 0.50 | 0.26 |

结论（有证据的"默认值没问题"）：
- 精确词题对 k **完全不敏感**（两个检索器都强，融合排名几乎不动）。
- 语义题 k=5 太小：Recall@10 掉一整题（0.25），"赢家通吃"挤掉了互补结果。
- 语义题 nDCG@5 随 k 增大单调下降（0.38→0.26）：k 越大越平滑、正确块排名被抹后——反向印证实验 6 的"稀释"。
- **k=30~60 是稳健区，保持默认 60**。面试可讲"我扫过 k，60 是稳健区而非拍脑袋"。

附：eval_qa.py 新增 `print_misses()` 失败题诊断（把 ✗ 变成"漏了哪个词、被哪块接住"）。用它确认：剩余全部失败都是"答案跨块"（实验 3/4 的结论），另暴露 mm 题的 gold 精度问题（`PromptUpdate` 类名 vs 正文 `Prompt Update Detection`，两者不在同一块）。

---

## 实验 10：chunk 大小 500 → 1000 —— 主导瓶颈是「chunk 切散」

（结论已落到 `config.py` 的 `CHUNK_SIZE=1000`；`eval_rerank.py` 里的 500 是当时快照，已标废弃。）
- 500 字会把答案段落切散——3 个 gold 词散到 2~6 块，而 top-10 只占语料的 ~1%，永远凑不齐。
- 换 1000 后 Recall 全线大涨：**chunk 大小是唯一真正起作用的杠杆**。此前假设的混合检索 / rerank /
  router / 更大 embedder 四个方向，逐个被数据否掉（这是整个里程碑 2 最重要的一条反直觉结论；
  其中「混合检索被否掉」在实验 20 被修正为「按题面分档」——BM25 只在无术语线索的抽象题上是噪声）。
- 代价：chunk 变大 → 命中块更长 → 端到端上下文膨胀，引出实验 11 的「小-to-大」和字符预算。

---

## 实验 11：小-to-大（parent-child）——「返回更大上下文」的真实收益与代价

动机：实验 10 把 chunk 调到 1000 后检索质量上去了，但 RAG 要的是「回答」而非「找到块」。把命中的小块
展开成父段落（## 节）能还 LLM 一个完整答案段。问题是这「提升」有多少水分，得拆开看。

设计：检索仍用小 chunk（精确召回），命中后展开成父段落（##节 `parent_level=2`）或整篇（`parent_level=1`），
用「小块 Recall（纯检索）/ 父段落 Recall（返回上下文后）/ 平均上下文字符」三列分开计量。

| 题集 | 小块 R@10（纯检索）| ##节 R@10（无预算·离线理想）| ##节 平均上下文 | 预算消融 R@10（8000 预算）| 整篇 R@10 | 整篇 平均上下文 |
|---|---|---|---|---|---|---|
| 精确词题（50）| 0.94 | 0.98 | 35,549 字符 | **0.86** | 1.00 | 142,337 字符 |
| 语义改写题（15）| 0.60 | 0.93 | 34,153 字符 | **0.67** | 1.00 | 135,287 字符 |

> ⚠ **预算消融 ≠ 离线理想值**：把 top-10 去重父段落按 `MAX_CONTEXT_CHARS=8000` 截断装箱（和
> `build_search` 同逻辑，`eval_qa.py` 的「8000预算 R@k」列可复现），R@10 掉到 **0.86 / 0.67**
> （无预算 0.98/0.93）。表里 0.98/0.93 是**无预算**离线理想值，讲实验必须报 0.86/0.67——上下文预算
> 吃掉了 0.12/0.26，这是「生产把 3.5 万字符压到 8 千」的真实代价，不能拿 0.98 冒充生产召回。

三个结论：
1. **##节的收益分两种**：精确题 0.94→0.98（+0.04，边际）；语义题 0.60→0.93（+0.33，大）——
   语义题的大头是「小块检索对改写 query 本来就弱（0.60）」，返回整段 ## 才捞回来，不是父段落魔法。
2. **整篇 R@10=1.00 是平凡天花板**：整篇文档必含全部 gold，R@1≈1.00 不代表检索排序好，只作
   「上下文成本上界」参考，勿用于生产吹嘘。
3. **代价是上下文膨胀，预算消融比离线理想值低一截**：##节也要 3.5 万字符/题、整篇 14 万，生产上
   `build_search` 压到 `MAX_CONTEXT_CHARS=8000`（`config.py`），R@10 从 0.98/0.93 掉到 **0.86/0.67**——
   这是「上下文预算」的真实代价。装箱用的是「截断」而非「跳过」：超长 top-1 往往是答案段，截它的前
   8000 字（段首含答案句）比整段丢掉、拿更低排名小段顶替更优——这个选择是拿数据验过的（截断
   0.86/0.67 vs 跳过 0.82/0.60），已用 `eval_qa.py` 复现。忠实度 A/B（同一检索只换装箱，5 题）截断
   0.99 vs 跳过 1.00，无质量差——截断在召回和忠实度上都占优或持平，坐实截断是正确默认。

→ 离线 keyword 指标到这里已到「测不准」的边界（实验 7 已证它与 rerank 的目标错配）。端到端质量
交给 `judge.py`（忠实度，里程碑 4）和 `agent.py`（多步检索 + 源码索引，里程碑 3/5）去验证。

---

## 实验 12：答案正确性评测 —— 补齐 RAG 评测「三件套」的最后一条腿

动机：`judge.py` 只测「忠实度」（回答里的每个声明有没有上下文支撑），keyword Recall 只测「检索到没」，
**都没有回答「答对没」**——一个 agent 可以检索到正确 chunk、忠实复述它，却漏说关键事实或答非所问
（关键词命中 ≠ 回答正确）。这是面试官会直接戳的软肋。

设计：从 50 道精确词题挑 15 道，每题手写 2~3 条「必须事实」（`eval_data.ANSWER_QUESTIONS`，共 33 条）。
`answer_eval.py` 跑完整 agent 拿答案 → LLM judge 逐条判「该事实是否被正确陈述」（语义等价即命中）→
正确率 = 命中事实 / 总事实。

**两条写 facts 的纪律（都是踩坑后总结的）**：
1. **facts 必须是「文档里讲的机制/因果/约束」，不是「gold 关键词换句话」**。第一版 facts 基本就是把
   三个 gold 关键词换个说法（如 lora 三条 = `VLLM_ALLOW_RUNTIME_LORA_UPDATING`/`load_lora_adapter`/
   `LoRAResolver`）——这样正确性就和 keyword Recall 重复了（检索到关键词 + 忠实复述 ≈ 必然命中，测不出
   新东西）。flag/类名题答案本身含标识符，就把标识符和它的作用/条件/默认值绑在一起，让 judge 测「讲对没」
   而非「出现没」。
2. **每条 fact 都要 grep 文档核对**，别凭记忆写（下面「反噬」一节就是凭记忆写翻车的实锤）。

### 结果：33/33 = 1.00（单跳基线，不是满分幻觉）

| 维度 | 脚本 | 测什么 | 结果 |
|---|---|---|---|
| 检索到没 | `eval_qa.py` | Recall@k | 0.86 / 0.67（精确词/语义，8000 预算 R@10） |
| 编没编 | `judge.py` | 忠实度（逐 claim 核对上下文） | 0.99 |
| 答对没 | `answer_eval.py` | 正确性（逐 fact 核对参考答案） | **1.00（33/33）** |

三层正交：检索到了 + 没编造 + 答对了，三者同时高才是真的能答。1.00 是「单跳文档事实题 + 检索准 +
忠实复述」的**应有基线**——这批题 agent 一轮工具调用就答完，本来就不该错；正确性这层的分辨力要等
「多跳 / 需源码 / 陷阱」题进来才显形（见「观察」）。

### 评测反手抓出 2 处参考答案错误（它的价值证明）

第一版 keyword-facts 跑出 34/37，3 处「漏」。逐条回语料核验，**2 处是参考答案自己错了**，只有 1 处是 agent 真漏：

1. **chunked-prefill（参考答案错）**：把「改善的延迟指标」写成 TTFT。语料 `optimization.md:57` 是
   **ITL**（inter-token latency，第 65 行才说调大 `max_num_batched_tokens` 才改善 TTFT）。agent 答对了
   （明确写 ITL + 解释 ITL/TTFT 权衡），是我把两个延迟指标记混了。
2. **structured（参考答案偏题）**：`StructuredOutputsParams` 确是 doc 里的类（`structured_outputs.md:312`），
   但题问的是「支持哪些 backend」，agent 答全了 4 个（xgrammar/guidance/outlines/lm-format-enforcer），
   这条 fact 不在题点上。
3. **paged-attn（agent 真漏）**：第一版要求点名函数名 `paged_attention_kernel`，agent 只给了描述性说法
   「multi-head query attention kernel」+ 文件路径。这条本身就是「关键词型 fact」，已随重写改成机制型。

**这个「反噬」正是加这一层评测的意义**：keyword Recall 和忠实度都不会暴露「参考答案写错」，正确性对照才会。

### 保留的坑
- **同模型偏见**：agent 和 judge 都用 deepseek-chat。用「逐条 fact、具体判 stated 是/否」而非模糊总分来
  缓解；严格版应换 deepseek-reasoner 做 judge 交叉验。
- **facts 要独立于 agent 答案写**：本批 facts 是「grep 文档后写的」，但写作时已经看过 agent 答案，独立性
  打折。以后新增题，先写 facts、再跑 agent。

### 观察
15 题全部一轮工具调用就答完（router 预检索 + 1 次 search，未进源码）——这批题是单跳文档事实题，
测的是「检索→直接答」链路，与 Recall 那列高度对齐。

### 多跳题的实测：逼出了多步，但 agent 仍然全对
又加了两道「答案跨两个文档」的题（chunked prefill↔CUDA graph、multimodal↔prefix caching），并刻意把
第一跳藏进题干、不让题面透露机制（第一版把「chunked prefill 会混合 prefill/decode」写进题干，等于送掉
第一跳，agent 一次 search 就答完——重写后才真逼出多步）。结果：

- Q16（chunked prefill↔cudagraph）用了 **3 次工具调用**（search + search_code）、Q17（multimodal↔prefix
  caching）用了 **5 次**（search/search_code 交替）——多步确实逼出来了。
- 但两道都 **3/3 答对**：agent 正确完成了跨文档 + 跨「文档/源码」的合成，连 `BatchDescriptor`、自动降级、
  `FULL_AND_PIECEWISE` 默认、`multi-modality input hashes` 这些边角都补对了（Q16 还引了源码
  `CudagraphDispatcher.dispatch(...)` 的调用形态）。

结论：**多跳文档题让 agent 多走几步，但还不够让它答错**——这个 agent 的检索 + 合成能力足以覆盖跨文档事实。
要让正确性这层真正掉到 1.00 以下，题必须戳到检索够不着的地方：**只写在源码里的实现细节**（search_code 的
符号粒度检索比文档粗，容易漏），或 **陷阱题**（judge.py 里那道「编造 flag」题就是现成的）。这是下一步。

→ 至此 RAG 评测三件套闭环：Recall（检索到没）+ 忠实度（编没编）+ 正确性（答对没）。

---

## 实验 13：源码检索评测（search_code 的 Recall@k）

> 动机：三件套只评了「文档检索」（search）这一条腿；agent 还有 `search_code`（源码实现检索，
> 48 个 .py / 337 个符号 chunk，复用同一套混合召回 + 重排 + module-overview 降权 + docstring 合成）。
> 它从没跑过 recall 数，正好接上实验 12 的尾巴——「要掉到 1.00 以下，题必须戳到源码实现细节」，
> 那源码检索本身到底好不好，先量化出来。
> 评测脚本：eval_code.py（`py -3.12 eval_code.py` 一键复现）；题目/gold：eval_data.CODE_QA_SET（18 题）。

### 和文档评测的三个关键差异（决定怎么读分）
1. **gold 换成「符号名」**：不再 3 个关键词，而是「实现该答案的类/函数/方法名」（如 `RejectionSampler`）。
2. **chunk 自足 → R@1 是主指标**：代码 chunk = 一个符号（函数/类），一个符号名就在一个 chunk 里；
   所以 R@1 就有意义（文档题因 3 词散落、R@1 常为 0 要靠 @5/@10）。@3/@5/@10 和 MRR/nDCG 用来分辨
   「命中了但排名靠后」的题。
3. **整标识符匹配（_has_symbol）替代朴素子串**：文档 gold 是完整配置键，子串碰撞少；代码 gold 是 Python
   标识符，子串碰撞是常态——`MTPSpeculator` ⊂ `MultiModuleMTPSpeculator`。用 `\b` 式边界（前后不能紧跟
   `[A-Za-z0-9_]`）匹配，否则「多模块 MTP」会被误算成「单模块 MTP」的命中（第一版体检就抓出了这个假阳性）。

### 结果：R@10 = 0.94，R@1 = 0.67

| 指标 | 数值 |
|---|---|
| Recall@1 / @3 / @5 / @10 | 0.67 / 0.83 / 0.83 / 0.94 |
| MRR | 0.76 |
| nDCG@5 / @10 | 0.54 / 0.54 |

18 题里 17 题在 top-10 内找全 gold 符号；12 题 R@1 即命中。对比文档检索（精确词 0.86 / 语义 0.67，@10）：
**源码检索「找得到」（0.94）比文档还稳，但「排第一」（0.67）更弱**——符合直觉：符号粒度 chunk 语义更稀疏，
靠 docstring 合成补锚点，能把答案捞进 top-10 但不够精准钉到第 1。

### 唯一的 miss：验证/接受类查询被「propose」簇抢走
`RejectionSampler` 这道「哪个类负责 accept/reject 验证」的题，top-10 里完全没有 rejection_sampler.py，
前几名全是 proposer（`SpecDecodeBaseProposer` / `NgramProposer` / `MedusaProposer.propose` …）。

- **不是数据坏**：换 query 探测，`verify draft tokens against target model probability`、`accept or reject
  draft tokens` 都能让 `RejectionSampler._verify` 浮到 top-1——符号本身可检索。
- **是真弱点**：语料里「propose/draft/sample」这个簇极大（每个 proposer 都有 `propose` 方法 + docstring），
  而「verify/reject」只集中在 rejection_sampler.py 一个小子目录。题干只要同时出现「draft tokens + 验证」，
  向量就被大簇拽走。这是 search_code 的**鲁棒性短板**：验证型查询词被提议型查询词淹没。

### 踩坑
1. **题干里冗余的「proposed」会毁掉检索**：第一版写「verifies the draft tokens *proposed* during speculative
   decoding」——「proposed」纯属多余修饰，却把 query 向量硬拽向 proposer 簇（体检复现）。删掉后题干才干净；
   但干净题干下仍 miss，才确认是检索弱点而非出题瑕疵。教训：代码题题干里的动词要克制，别让无关的
   「propose/draft/sample」这类高频词污染语义。
2. **稀疏度看「分散到几个文件」（n_files）不看 chunk 数**：类名会出现在「类概览 + 每个方法」chunk 的出处头，
   `AutoRegressiveSpeculator`（18 个方法）在索引里出现 19 个 chunk，但 n_files=1，仍唯一——按 chunk 数判
   「太通用」是误报。真正危险的是 propose 这种十几个文件都有的名（n_files 爆炸）。eval_code 用 n_files>6 报警。

### 结论
- search_code 的召回是「捞得回来、排不准」：R@10 0.94 但 R@1 0.67、MRR 0.76。对 agent 的多轮场景
  「捞回 top-10 再让 LLM 挑」够用；但对「直接 cite 第一个结果」的场景，验证/接受类实现是已知短板。
- 这条腿补上后，agent 的检索面（文档 + 源码）都有了数字；下一步按计划转「代码生成正确性」——评
  write_file/edit_file 写出代码对不对（见用户方向：先 2 后 1，最后延申到 3）。

---

## 实验 14：代码生成正确性评测（write_file 写代码对不对）

> 动机：按用户方向「先 2（search_code 检索）后 1（代码生成）」，评 agent 的**写代码工具**。判法采用
> 用户批准的「静态事实核对」：每题手写 2~4 条「代码事实」（代码里必须出现的 API 元素），静态匹配判对。
> 评测脚本：eval_codegen.py（`py -3.12 eval_codegen.py` 一键复现）；题目/事实：eval_data.CODE_GEN_QUESTIONS（8 题）。
> 只开 `VLLM_COPILOT_ALLOW_WRITE=1` 暴露 write_file，agent 走生产 loop，代码落盘到 workspace/ 再读回来判。

### 三层判分（对「写代码」这个对象逐层收紧）
1. **ast.parse 语法门**：能不能加载（`import` 它不报 SyntaxError）。
2. **facts 静态核对**：API 用对没——`contains`（必须出现的子串）全命中才算过，`not_contains`（必须不出现）抓
   废弃 API（`guided_json`/`guided_decoding_backend` 已换成 `structured_outputs`）。
3. **run_check 真执行**（仅纯 Python 题）：把函数 exec 进干净命名空间再跑断言，比静态匹配强得多。

### 8 题构成（5 API 用法 + 3 纯 Python 可跑）
API 题（不可跑，只静态判）：fp8+prefix-caching、chunked-prefill（max_num_batched_tokens）、LoRA、
structured outputs、n-gram 投机解码。纯 Python 题（可跑）：RRF 融合、KV cache 大小、next_power_of_2。

**facts 是 ground truth，写前 grep 文档核对 API 面**（同实验 12 的纪律）：`kv_cache_dtype="fp8"` /
`enable_prefix_caching=True` / `max_num_batched_tokens=16384` / `from vllm.lora.request import LoRARequest`
+ `enable_lora=True` + `LoRARequest(...)` / `StructuredOutputsParams(choice=...)` +
`SamplingParams(structured_outputs=...)`（guided_* 已废弃）/ `speculative_config={"method":"ngram",
prompt_lookup_min/max, num_speculative_tokens}`。

### 结果：8/8 语法、21/21 facts、3/3 真执行

| 维度 | 数值 |
|---|---|
| 语法可加载（ast.parse） | 8/8 |
| fact 命中 | 21/21（1.00） |
| 纯 Python 题真执行通过 | 3/3 |

### 这不是满分幻觉，是「跟文档走」的应有天花板
和实验 12 的 1.00 同理：这批题是「API 用法照文档写」，API 面 grep 得明明白白、agent 又有文档检索兜底，
**本来就该全对**。它的价值不在「让人吃惊的数字」，在：
1. **证明了 write_file 这条链路端到端可用**——agent 能 1~2 次工具调用写出可直接 `ast.parse`、`exec` 的
   自足脚本（不是「给个大概」）。
2. **代码质量高得出乎预期**：8 个文件全部自带 docstring 且内容准确（chunked-prefill 还引了 optimization.md
   的 ITL/TTFT 权衡，ngram 题解释了 lookup 窗口语义）；API 用全对的现代接口——`from vllm.sampling_params import
   StructuredOutputsParams`（不是废弃的 guided_*）、`LoRARequest("sql_adapter", 1, path)`、`speculative_config`
   里 `"method": "ngram"` 三键齐全。
3. **确立了基线**：以后换模型/改 system prompt/改工具描述，重跑这个 21/21 就是回归防线。分辨力要等
   「戳源码实现 / 多文件 / 边角」的硬题进来才显形（同实验 12 的结论）。

### 意外发现：route() 分不清「写代码」和「跑 Python」
route 的启发式 `\b(run|write|execute)\b.{0,30}\b(python|code|script|snippet)\b` 只认两种去向：run_python /
search。**「Write a Python script…」会被误路由到 run_python**（hint 让 agent「用 run_python 执行、别凭空猜」），
但写代码任务该走 write_file——router 根本没有 write_file 这一路。这是 agent 对「代码生成」这类任务的一个真实
短板：它的路由表里没有「生成代码」这个意图。本实验靠改题面措辞（「Write a program/function」避开触发词）绕过，
但生产里用户自然会说「帮我写个 Python 脚本」，就会进错分支。→ 值得给 router 加一路「代码生成」识别
（或至少把 write 类的去向从 run_python 里拆出来）。

### 踩坑
1. **静态 contains 的子串碰撞**：`"fp8"` ⊂ `"fp8_e4m3"/"fp8_e5m2"`、`"True"` 是弱信号（任意 `if True:` 都命中）。
   对「必须写对这个值」的强判断，纯 Python 题靠 run_check 真执行兜底；API 题目前接受这个轻微不精确
   （`kv_cache_dtype` + `fp8` 同时出现已是强证据）。
2. **facts 要 grep 核对，别凭记忆**：`structured` 题的 import 路径（`vllm.sampling_params.StructuredOutputsParams`）
   和 LoRA 的 `LoRARequest(name, id, path)` 三参序，都是文档里钉死的——记错 import 路径会把对的代码判错。
3. **落盘优先、答案回退**：extract_code 先读 workspace 落盘文件，没落盘才回退答案里的 ```python 块。评测前
   清空 workspace，避免上一轮产物被当成本轮结果（读错文件比读不到更糟）。

### 结论
- 代码生成这条腿通了：write_file + 文档检索能把「API 用法」题写出 21/21 全对的可加载代码。
- 但这不是评测的终点，是起点：8 题全是「跟文档走」，测不出 agent 的真本事。要测「会不会写代码」，题得
  戳到文档够不着、要读源码才能写对的地方（比如「按 vLLM 源码里 `next_power_of_2` 的实现风格写一个块对齐
  工具」），或「多文件 + 要改既有代码」的 edit_file 场景——那才是步骤 1 该有的难度，也是延申到步骤 3
  （端到端写 + 跑）的垫脚石。
- 顺带抓出一个真实短板（router 无「代码生成」意图），值得单独修。

---

## 实验 15：代码生成评测加「硬题」+ route 修掉 write/run 混淆

### 一、route 修复：拆出 write_code 一路（实验 14 抓出的短板）
实验 14 发现「Write a Python script」被误路由到 run_python（hint 让 agent 跑代码而非写文件）。修复：route()
从「run_python / search」两路拆成三路——
- **write_code**：write/generate/create/implement/produce + python/code/script/snippet/function/program/class/module
- **run_python**：run/execute/compute/calculate + python/code/script/snippet（**去掉了 write**）
- **search**：其余，预检索文档

写代码题不再进 run_python。`test_route.py` 重写成 5 个用例锁三类意图（65 题全 search + write/run 各开关组合），
hint 也改成「写/改代码，用 write_file 或 edit_file」。

### 二、硬题：4 道「读源码才能写对」的题
前 8 题是「照文档写 API」（实验 14 全对、无分辨力）。加 4 道 facts/run_check 钉在「只写在源码里」的细节：
| 题 | 复刻什么 | 判据 |
|---|---|---|
| spec-metadata | SpecDecodeMetadata 数据类 | 7 字段名 + `__post_init__` 静态 facts |
| accept-decision | 拒绝采样接受判定 | greedy 比 argmax / 非 greedy 比 log-ratio，run_check |
| max-chunk-logits | get_max_chunk_logits | 1 GiB 常量 2**30 / 4 字节 FP32，run_check |
| fix-next-pow2 | edit_file 修 off-by-one | seed 预置 bug 文件，not_contains 抓残留 bug 行 |

（eval 新增 `seed` 机制：题可预铺一个有 bug 的文件进 workspace，让 agent read_file 读回再改。）

### 三、结果：8 易题全过，4 硬题 2 过 2 挂（分辨力出来了）

| 维度 | 数值 |
|---|---|
| 语法可加载 | 11/12 |
| fact 命中 | 26/31（0.84）|
| 纯 Python 题真执行 | 5/6 |

硬题：accept-decision ✓（search_code 捞回 log-ratio 公式）、fix-next-pow2 ✓（read_file+edit_file 一次改对）、
**spec-metadata ✗、max-chunk-logits ✗**。两个挂点都戳到真短板，且都能根因到代码索引的切块策略。

### 四、两个失败都根因到「代码索引只切符号，不切常量/字段」
1. **max-chunk-logits：模块级常量不在索引里**。`chunk_python` 只切 FunctionDef/ClassDef/方法，`MAX_CHUNK_BYTES =
   2**30` 是模块级 `ast.Assign`，**整个不进索引**（`_module_overview` 只有 docstring、不含常量）。agent 捞回了
   `get_max_chunk_logits` 函数体（`max(1, MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES))`），但公式里只有常量名、
   没有值，于是 agent 猜 256 MiB（真值 1 GiB），run_check 断言 2684 vs 671 当场挂——agent 的 docstring 自己承认
   "exact value ... not retrievable from the source index"。
2. **spec-metadata：类字段声明不在索引里 + agent 检索死循环**。`_class_overview` 只收 docstring + 方法签名，
   数据类的字段注解（`ast.AnnAssign`）不收；字段名只「偶然」出现在 `make_dummy` 的 `cls(draft_token_ids=...)`
   调用里。agent 12 次工具调用（8 次 search_code）没凑出一个干净的字段表，撞 max_steps 没写文件。帮凶：
   read_file 只读 workspace/，agent 想 read_file `vllm/v1/spec_decode/metadata.py` 直读源码，返回「文件不存在」。

### 结论
- 加硬题让代码生成评测第一次有分辨力：8 易题 0 失败、4 硬题 2 失败。失败点不是「不会写代码」，而是
  **「代码索引给不全」**——符号粒度索引对「模块常量 / 类字段」这类非函数/非类的顶层元素是盲区（`ast.Assign` /
  `ast.AnnAssign` 都不切）。
- 这是 search_code 索引的结构性缺口，不是 agent 的推理短板：函数/类能切块，常量/字段切不到，而「照着实现
  复刻一个函数」恰恰最需要常量的值、数据类的字段。
- 下一步方向：a) 修 chunker 把模块级 Assign / 类字段 AnnAssign 也切进索引（补常量/字段盲区）；b) 或先记录不动，
  继续推步骤 3（端到端写+跑）。

### 五、补记：修掉 chunker 盲区（选 a）
`code_chunker.py` 三处改动 + 自检回归：
1. **模块级常量**：`chunk_python` 新增 Assign/AnnAssign 分支，只认全大写名（`name.isupper()`，PEP8 常量约定），
   每个常量单独成 chunk（`[source … | constant MAX_CHUNK_BYTES]` + 完整赋值）。不切 `scheduler = Scheduler()`
   这类小写初始化。
2. **类字段**：`_class_overview` 收进 `ast.Assign`/`ast.AnnAssign`（用 `get_source_segment` 取完整段），作为
   `# 字段` 一节排在方法签名前。
3. 自检样例加了 `MAX_STEPS = 4` / `_FP32_BYTES = 4` / 类字段 `draft_batch_size: int = 32`，断言它们都进 chunk。

在真实文件上核对通过：`rejection_sampler.py` 现在索引 `MAX_CHUNK_BYTES = 2**30` 和 `_FP32_BYTES = 4` 两个
常量 chunk；`metadata.py` 类概览现在带 `@dataclass` + 7 个字段声明（`draft_token_ids: torch.Tensor` 等）。
（这两处盲区正是 max-chunk-logits / spec-metadata 的失败根因。）重跑 12 题评测看收敛结果。

### 六、重跑结果：12/12 全过（盲区修掉后收敛）
| 维度 | 修前 | 修后 |
|---|---|---|
| 语法可加载 | 11/12 | 12/12 |
| fact 命中 | 26/31 (0.84) | 31/31 (1.00) |
| 纯 Python 题真执行 | 5/6 | 6/6 |

两个失败题都翻绿：
- **max-chunk-logits**：agent 多打一次 `search_code("_FP32_BYTES constant definition")` 就捞到常量值，写出
  `MAX_CHUNK_BYTES = 2**30` / `_FP32_BYTES = 4`，run 通过（2684/256/1 三断言全过）——上一轮它只能猜 256 MiB。
- **spec-metadata**：3 次工具调用就写出完整数据类（上一轮 12 次撞 max_steps 没落盘），类概览 chunk 的
  `# 字段` 一节把 7 个字段一次给全。

结论：实验 15 的「硬题」真正兑现了价值——它们不是考「会不会写代码」，而是逼出了代码索引的结构性盲区
（模块常量 / 类字段不切块），修掉后 12/12 全过。评测从「会写 API」升级到「能复刻源码实现」。

---

## 实验 16：代码判分从「查字符串」换成「解析 AST」（code_facts.py）

> 动机：用户 review 实测戳穿旧判分的假——`contains`/`not_contains` 就是「这些字符串在文件里出现了吗」：
> ① 参数值改错、正确词写进注释 → 4 项全过；② 整个文件全是注释 → 4 项全过（纯注释是合法 Python，语法检查拦不住）；
> ③ 6 个不运行题全换成注释空壳 → 6 个满分；④ 反过来，正确代码 + 一行提废弃 API 的注释 → 3 项被判挂。
> 大白话：假货能过、真货可能被判挂。

### 设计：注释进不了语法树，一个改动同时堵死两个方向
- `code_facts.py` 用 `ast.parse` 把代码解析成语法树，fact 从「子串」改成「AST 结构断言」——注释和字符串
  字面量根本进不了语法树，所以「注释里的正确词」救不了「写错的参数值」、「注释里的旧 API」也冤不了「正确的代码」。
- check 类型（kind）：`import/imports/call/kwarg/dict_entry/pair/func/class/field/assign/name/compare/pow2_buggy`。
- 支持 `"not": true` 取反（抓废弃 API、抓 bug 形 `1<<n.bit_length()`）+ 名字常量解析（`max_num_batched_tokens=
  MAX_NUM` 且 `MAX_NUM=16384` 也算命中）。
- 31 条 facts 全量迁移到 `{"desc", "checks": [...]}`（eval_data.py），eval_codegen.py 改 import `code_facts.check_facts`。

### 结果：数字没变，但这次判分是真的
| 维度 | 旧（子串）| 新（AST）|
|---|---|---|
| 语法可加载 | 12/12 | 12/12 |
| fact 命中 | 31/31 | 31/31 |
| 纯 Python 题真执行 | 6/6 | 6/6 |

数字不变（agent 本来就把代码写对了），**变的是可信度**：`code_facts.py` 的 self-test 把用户实测的三种攻击
场景钉成回归断言——纯注释空壳 0 分、注释里的正确值不救写错的值、注释提旧 API 不冤枉正确代码。今后判分
不再能被注释骗。

### 踩坑：pair 派发把 key 当成 name
`pair`（kwarg 或 dict_entry 任一命中）派发到 `kwarg` 分支时只改了 `kind` 没把 `key` 映射成 `name`，
`c["name"]` KeyError 崩掉整个评测。修：`{**c, "kind": "kwarg", "name": c["key"]}`。

### 结论
- 判分这一层现在是「结构真值」，和实验 14/15 的「写对没」数字有了公信力背书。
- 配合实验 17 的 run_file（真执行），代码生成评测从「静态核对」升级到「静态结构 + 真执行」双层，不再是字符串玩具。

---

## 实验 17：run_file 工具 + pass@1/fix-rate + 9 个真 bug 修复

> 三件事：review 的「第三件」（写文件≠能运行）+「其余 9 条真 bug」，顺带补上 coding agent 两个此前完全
> 空缺的核心指标。

### 一、run_file 工具：补上「写文件 → 运行」闭环（review 第三件）
`run_python` 只能跑一段字符串，agent 写完文件想验证只能把代码再抄一遍塞进去（绕路、开环）。新增 `run_file`：
- `build_run_file(path, check=None)`：`runpy.run_path(target, run_name="__main__")` 进程内跑已落盘文件，
  `check` 是可选断言片段，跑完 `exec` 进文件命名空间验证「跑出来的值对不对」。
- 和 run_python 一样受 `VLLM_COPILOT_ALLOW_RUN_PYTHON` 门控；`execute()`/`_get_tools()`/`tool_desc`/write_code hint 全接上。
- 意义：这是「能写代码」和「能写对代码」之间的坎——报错是确定性、免费的真值，比 LLM 自评靠谱得多。

### 二、pass@1 / fix-rate 两个核心指标（此前完全是空的）
`eval_codegen.py` 加一轮「修复轮」：runnable 题首轮 run 没过 → 把 run_check + 报错喂回 agent，让它用 run_file
自测自纠后再判。汇总输出：
- **pass@1** = 首轮 run 就过的占比；**fix-rate** = 首轮失败里修复轮通过的占比。
- 当前这批题 agent 首轮全对（pass@1 = 6/6），fix-rate 无样本——指标管道已就位，分辨力要等能答错的硬题进来。

### 三、9 个真 bug 逐条修（都实测复现过）
| # | bug | 根因 | 修法 |
|---|---|---|---|
| 1 | BM25 分词不分大小写 | tokenize 不 lower，`RejectionSampler` 被劈成两个词 | `text.lower()`（R@10 0.875→1.000）|
| 2 | LLM 非法 JSON 崩掉整个任务 | `execute()` 的 `json.loads` 不防错 | try/except 接住，错误喂回 LLM 重试 |
| 3 | 长代码块静默截断 | 342 块里 91 个 >2000 字，embedding 只编码前半截不报错 | `_split_long_chunk` 切成 ≤460 token 多段，带 `[part i/n]`（按 1600 字切仍挡不住 token 密度，实测 18.7% 块超 512；改按 token 切后 0% 超线）|
| 4 | 空段（只有标题没正文）被丢 | `_chunk_section` 对空 body 返回 []，标题文本蒸发 | 空 body 但有标题时单独成 chunk |
| 5 | edit_file 空串 | `old_string=""` 匹配任意位置，`replace("", …)` 插到文件头 | 拒绝空 old_string |
| 6 | 缓存键含行号 | 出处头 `path:12-73` 进缓存键，改一行就 700+ 缓存全 miss 重烧 LLM | `re.sub(r":\d+-\d+", "", header)` 去行号 |
| 7 | 合成无失败隔离 | LLM 网络/限流崩掉 `load()` 整个索引 | `_synthesize_batch`/`_synthesize`/`_save_cache` 三层 try/except |
| 8 | 嵌套类丢失 | `chunk_python` 只遍历顶层，class 套 class 整类漏 | 递归 `_chunk_nodes`，嵌套类/方法带全限定名 |
| 9 | 默认参数不一致 | `chunk_markdown` 默认 `max_chunk=500`，与 `CHUNK_SIZE=1000` 漂移 | 默认改 `CHUNK_SIZE`（config 单一来源）|

全部 smoke-tested 通过：split 段数 / 缓存键去行号 / 空段保留标题 / 非法 JSON 返回错误串 / 空串拒绝 /
嵌套类自检断言。

---

## 实验 18：把两个虚高分数打回原形（去同源泄漏 + 补难题）

> review「第二件」：代码检索 R@1 靠「题面泄漏」，答案 33/33 零信息量。这实验只做一件事——把分数里
> 的水分挤掉，让「能对外说」的数字和「真本事」对上号。

### 一、代码检索 R@1 是假的：去「同源词」后直接崩到 0.06

**泄漏机制**（比 check_code_gold 查的更深）：旧题面不是只含 gold 符号名字面（那关 check 早防了），
而是拿 docstring/源码的实现措辞近义改写——`RejectionSampler` 题说 "accepting or rejecting"、
`_find_longest_matched_ngram_and_propose_tokens` 题说 "prompt-lookup … longest … matches the current
suffix"。这些「同源词」在向量/BM25 上几乎必然命中合成描述（docstring 合成本来就是用 LLM 写一句
"这符号做什么"），于是 R@1 测的是「题面和合成描述字面重合」，不是「检索能理解代码」。

**修法**：18 题全部改用「高层角色描述」措辞，避开 gold 符号名和源码实现词（reject / prompt-lookup /
hidden states / log-sum-exp / autoregressive / EAGLE / MEDUSA …）。

| 指标 | 泄漏版 | 去同源后（实测） | review 温和改写估计 |
|---|---|---|---|
| R@1  | 0.625 | **0.06**（1/18） | 0 |
| R@10 | 0.875 | **0.33**（6/18） | 0.75 |
| MRR  | —    | 0.10 | — |

**结论**：
- R@1 = 0.06 基本归零——「能不能排第一」确实全靠题面里的词，不是能力。
- R@10 = 0.33 比 review 估的 0.75 还低：因为我这版改写更狠（连 "several-tokens-at-once" 这种
  半同源词也换成了更抽象的「角色描述」），说明「能不能捞回来」的真本事比温和改写暴露得还要弱。
- 但没归零：MTPSpeculator 仍 R@1 命中（"several-tokens-at-once" 确实唯一），Gemma4/DSpark/Vocab
  /Adaptive/AutoRegressive 仍能进 top-10——说明检索能拿到「符号的邻域」（如 ngram 题 top-2 捞出
  NgramProposer.propose，只是没钉到具体方法），只是钉不准具体符号。
- **对外口径**：代码检索只能说「R@10 ≈ 0.33 能捞回邻域」，不能说「R@1 能定位实现」。

### 二、答案 33/33 零信息量：补两道「能答错」的题

旧 15 单跳 + 2 多跳全是「文档一句话能答」，agent 一轮全对，全对和全错一样没信息量。补两道（ANSWER_QUESTIONS 17→19）：
- **陷阱题**：「哪个环境变量能一键开启投机解码？」——正确答案是「没有这个变量，走 speculative_config 参数」。
  戳 agent 编 env 变量名的幻觉倾向。
- **源码题**：「GPU rejection sampler 里固定 buffer 多大、每 chunk 行数上限怎么算？」——答案只在
  rejection_sampler.py（`MAX_CHUNK_BYTES = 2**30`、`get_max_chunk_logits`），文档搜不到，逼走 search_code。

**口径修正**：正确性这层从此只能当「回归网」（盯住别退化），不能当「能力分」；能力分交给
代码检索（R@10）和 pass@1/fix-rate（实验 17）去扛。

---

## 实验 19：两个评测 artifact 修复（同符号多段合并 + 题面去撞车）

> review 又揪出两个真问题——都不是检索能力问题，是「评测本身」的 artifact：切分让大函数占满
> top-10 格子、去同源去过头让题面互相撞车。修掉后，实验 18 的 0.33 其实把真本事算低了。

### 一、同符号多段合并（R@10 0.33→0.44 的主因）

实验 17 的 `_split_long_chunk` 解决了截断，但引入副作用：超长符号切成 N 段后，每段都是独立候选。
大函数切 8 段 = 8 个候选，小函数仍 1 个。数 18 题 × 10 = 180 个 top-10 位置：SpecDecodeBaseProposer
一个类的方法多段占了 34 个 = **19%**，小符号根本挤不进 top-10。

**修法**：`CodeIndex.search()` 在召回阶段按「符号身份」（出处头去掉 `[part i/n]`）去重，只留 fused
分最高的一段，每个符号最多占一个名额 → recall 和 top-k 都自然去重。
**一个偏离 review 的点**：review 说「重排后合并」，我改在**召回阶段**合并——RECALL_N=20 的召回同样
被大函数多段挤占（8 段占 8/20 格子，小符号连重排都进不去），只重排后合并治标不治本。

### 二、题面去撞车 + 可区分性检查

去同源去过头了：ngram「向后扫文本找可复用片段」和 suffix「复用之前见过的序列」在向量空间里几乎
重合——`_find_longest_matched_ngram_and_propose_tokens` 跑进 suffix 题的 top-4/5、掉出自己的 top-10；
eagle 的答案跑进 rejection 题的 top-6。**题面差异小于答案差异时，检索器再强也会错排**——这是出题
问题，不是检索问题。

**修法**：给撞车的对补互相排斥的区分特征（不泄漏 gold 名字面）：
- ngram ↔ suffix：ngram 强调「有界窗口 + prompt 前缀」，suffix 强调「已生成序列的缓存」。
- eagle ↔ rejection：rejection 强调「提案**之后**比对两模型概率、留/弃」，eagle 强调「用主模型隐状态**增广**输入」。
- autoregressive ↔ eagle：autoregressive 强调「独立 draft 模型逐 token」，eagle 去掉「逐 token」、只留「隐状态增广」。

**加 harm 模型**：`check_query_discriminability()` —— 任意两题题面 embedding 余弦 >0.80 就报警，
和 `check_code_gold` 一起在评测前跑。修完 18 题 0 对超阈值。

### 三、结果

| 指标 | 泄漏版 | 去同源（实验18，含两 artifact） | 修 artifact 后（实验19） |
|---|---|---|---|
| R@1  | 0.625 | 0.06 | **0.17** |
| R@3  | —     | 0.06 | **0.22** |
| R@5  | —     | 0.17 | **0.28** |
| R@10 | 0.875 | 0.33 | **0.44** |
| MRR  | —     | 0.10 | **0.23** |

### 结论
- **公平的代码检索能力 = R@1 0.17 / R@10 0.44**（这是「不点名词」的抽象档）。比泄漏版低得多（该低），
  比实验 18 的 0.33 高（0.33 被「多段挤占 + 题面撞车」两个 artifact 压低，不是真本事）。实验 20 补上
  另一半口径：题面点名词的自然档 R@10 = 0.94，差值 +0.53 就是「术语线索溢价」——两档合起来才是完整的
  能力图景（抽象档 0.44 更接近真实用户「不点名」时的能力）。
- 教训：去同源方向对，但「去多少」要留一手——线索全抹掉，题目自己就分不开了。可区分性检查把
  这个「留一手」固化成了可执行的 harm 模型。

---

## 实验 20：两档测量（自然 vs 抽象题面）——「术语线索」值多少分

> 动机：实验 18/19 把代码题改写成「高层角色描述」（去同源）后 R@1 0.06→0.17、R@10 0.33→0.44，
> 但这一刀可能砍过头——真实用户提问时，知道类名的会说「MEDUSA 那个 proposer」，不知道的才说
> 「那种叠几个分类头的」。把「题面里有没有术语线索」拆成两档、逐题配对，量化它到底值多少分，
> 顺便回答实验 10 遗留的「混合检索到底有没有用」。
> 评测脚本：eval_two_tier.py（`py -3.12 eval_two_tier.py`）；题目：eval_data.NATURAL_QA（档①自然，
> 17 题）与 CODE_QA_SET（档②抽象，同 id 对齐、同 gold）；主口径 S1（现行 RRF），截断常数 21。

### 设计：配对差值，只换题面口径
同一索引、同一 reranker、同一 top-10，只换两件事：题面（自然/抽象）× 召回融合（S1~S6 六种）。
两臂同 id 逐题配对，Δ = 自然命中率 − 抽象命中率，用 McNemar 精确检验（binomtest）+ bootstrap CI
（按题重抽样百分位）+ Wilcoxon 符号秩判显著性。

### 12 格消融（2 档 × 6 策略；R@k / MRR / cov=召回覆盖率）

| 策略 | 档①自然 R@1 R@3 R@5 R@10 MRR cov | 档②抽象 R@1 R@3 R@5 R@10 MRR cov |
|---|---|---|
| S1 RRF(1,1) n20 | 0.71 0.76 0.82 0.94 0.75 0.94 | 0.18 0.24 0.29 0.41 0.24 0.53 |
| S2 纯向量 n20 | 0.65 0.71 0.82 0.82 0.71 0.88 | 0.12 0.29 0.29 0.53 0.24 0.71 |
| S3 加权RRF .85/.15 n20 | 0.65 0.71 0.82 0.82 0.71 0.88 | 0.12 0.24 0.35 0.53 0.24 0.65 |
| S4 RRF(1,1) n40 | 0.71 0.76 0.82 0.94 0.76 1.00 | 0.12 0.24 0.24 0.41 0.22 0.71 |
| S5 min-rank n20 | 0.71 0.76 0.82 0.88 0.76 0.88 | 0.24 0.24 0.29 0.47 0.29 0.65 |
| S6 纯向量 n40 | 0.65 0.71 0.76 0.88 0.71 0.94 | 0.12 0.24 0.35 0.41 0.22 0.76 |

### L1 配对差值（主指标 R@10，主口径 S1；b=自然中抽象不中，c=抽象中自然不中）

| 分组 | n | 自然 R@10 | 抽象 R@10 | b | c | Δ | 95% CI | McNemar p |
|---|---|---|---|---|---|---|---|---|
| A组 | 8 | 1.000 | 0.375 | 5 | 0 | +0.625 | [+0.250, +0.875] | 0.062 |
| B组 | 9 | 0.889 | 0.444 | 4 | 0 | +0.444 | [+0.111, +0.778] | 0.125 |
| **合计** | **17** | **0.941** | **0.412** | **9** | **0** | **+0.529** | **[+0.294, +0.765]** | **0.004** |

L2 补充：ΔMRR = +0.508（Wilcoxon p=0.003），名次差中位 −10。

### 三个结论（回看实验 10 / 18 / 19）

1. **「术语线索」值 +0.53 R@10，全程 c=0**（没有一道「抽象命中而自然不中」）。自然档 0.94 里大部分
   是题面里词根/符号名的字面重合，不是检索能力。**对外口径修正**：代码检索「不点名词时 R@10 ≈ 0.41
   （≈ 实验 19 的 0.44）」，点名词时 0.94，差值 0.53 是术语溢价，不是真实用户提问时的能力（真实用户
   不点名的概率更高）。
2. **缺口在召回不在重排**：抽象档召回覆盖率仅 0.53（自然 0.94），8/17 题 gold 没进 top-20。语义召回
   （不知道类名、只描述功能）才是当前真短板，抽象档 0.41 才是「当前能力」更诚实的估计。
3. **BM25 的角色按档翻转——实验 10 的「混合检索被否掉」要加限定**：消融表里抽象档 R@10 纯向量 0.53 >
   RRF 0.41（BM25 是噪声）；自然档 RRF 0.94 > 纯向量 0.82（BM25 是 +0.12 资产）。精确说法：**BM25 只在
   query 没有词面锚点时是噪声，有术语线索时是资产**——这不是「否掉混合检索」，是「混合检索的价值取决于
   query 有没有词面锚点」，两条腿都要，融合要按题型切（router 的第 4 步）。

### 逐题名次的几个尖点
- **autoregressive 是唯一「抽象档名次反而更好」的题**（自然 rank 10、抽象 rank 2，两档都 @10 命中）：
  抽象题面用「independent draft model」点中了它的核心机制，反而比自然题面更接近符号；其余 16 题
  自然 ≥ 抽象。
- **sd-metadata 两档都 miss**（rank 21/21）：题面点不点名都捞不到，是唯一「术语线索也救不回」的题——
  也是自然档 0.94 里唯一失分的那道，独立于术语溢价的真实盲区。

### 口径声明（写进任何对外表述）
① Δ 读「词根/符号线索值多少分」，不读「真实用户 vs 抽象」——两臂题面都看着 gold 写出来。② 主指标 R@10
预指定。③ 截断常数 21（RECALL_N+1）。另有两条题面标注：档② dflash 题面「merges the main model's work into
the same pass」是错误概括（DFlash 跑自己的 draft model），档②已冻结不改仅标注；RejectionSampler 在 vLLM
有两处定义但索引里只有 GPU spec_decode 那处，命中无歧义。
