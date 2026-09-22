# Agent 项目技术清单（走 ③ 高价值段：造 Agent 基础设施）

> 目标：把 `minimal_rag.py` 从"调包式 RAG"演进成一个**能自主决策的 Agent**。
> 核心价值在"手写 Agent 循环 + 工具调用 + 评测"，而不是调 LangChain。
> 面向岗位：DeepSeek Agent Harness（靶心）、Moonshot 后端（兜底）。

---

## 当前进度（2026-09-18）

里程碑 1~5 已全部落地，里程碑 6（原计划的部署/README/博客）待做：

| 里程碑 | 内容 | 状态 |
|---|---|---|
| 1 | RAG 基线 `rag_baseline.py` | ✅ |
| 2 | 检索底座 + 消融实验 `eval_qa.py` / `experiments.md` | ✅ |
| 3 | Agent 主循环 + 工具调用 `agent.py` | ✅ |
| 4 | LLM-as-judge 忠实度 `judge.py` | ✅ |
| 5 | Agent 增强：代码索引 `code_index.py` + `search_code`/`run_python`/写文件 `read_file`·`write_file`·`edit_file`（workspace 白名单）+ 显式 router + `profiler.py` 成本打点 | ✅ |
| 6 | FastAPI/Gradio 部署 + README + 技术博客（原计划里程碑 5） | ⏳ |

> 注：实际「里程碑 5」做的是 Agent 增强（代码索引 + 工具 + router + profiler），
> 不是下方第 5 条写的「部署」——部署顺延为里程碑 6。下方里程碑列表保留原计划，进度以本表为准。

---

## 一条判断标准（先记住）

**手写 = Agent 的"决策与编排"逻辑；用库 = "模型与底座"。**
面试官只会问你手写的那部分。用库的部分调对了就行，不配占你时间。

---

## 一、整体架构

```
用户问题
   │
   ▼
┌─────────── Agent 主循环（手写 ReAct）─────────────┐
│ ① 思考：调 LLM，让它输出"下一步动作"                │
│ ② 解析：判断是"调工具"还是"给最终答案"              │
│ ③ 执行：调用下面注册好的工具                        │
│ ④ 观察：把工具结果拼回上下文，回到 ①                │
│ ⑤ 终止：达到 max_steps 或 LLM 给出 final_answer    │
└───────────────────────────────────────────────────┘
        │ 调用                         │ 读写
        ▼                             ▼
  ┌── 工具集（手写注册+协议）──┐   ┌── 上下文 / Memory ──┐
  │ search(query)  混合检索     │   │ 对话历史（短期）     │
  │ rerank(docs,q) 重排        │   │ 检索缓存/摘要（长期） │
  │ verify(claim)  溯源校验     │   └────────────────────┘
  │ final_answer(text)         │
  └────────────────────────────┘
        │
        ▼
  ┌── 检索底座（部分用库）─────────┐
  │ 文档解析(用库) · 分块(手写策略)│
  │ BM25(手写/rank_bm25)         │
  │ 向量检索(Chroma→Qdrant)       │
  │ embedding(bge-m3 API)         │
  │ reranker(bge-reranker API)    │
  └───────────────────────────────┘
```

---

## 二、必须手写（差异化核心，面试官问的就是这些）

### 1. Agent 主循环（ReAct 循环）
一个 while 循环：`LLM → 解析动作 → 执行工具 → 拼结果 → 再 LLM`，直到给出答案或超步数。
这 ~50–100 行是 Harness 的心脏。**别用 LangChain 的 AgentExecutor。**
面试可讲的点：为什么要循环、为什么要限制 max_steps（防死循环）、怎么让 LLM 知道"该停了"。

### 2. 工具调用协议
- 自己定义每个工具的 **JSON Schema**（名字、参数、描述）；
- 自己写"把工具描述塞进上下文 → LLM 返回结构化 tool_call → 解析 → 执行 → 把结果回填"这套 plumbing。
- 实现方式：先用 DeepSeek/OpenAI 的 **function calling (tools) API** 跑通；理解原理后能讲"如果不靠 function calling，就是 2022 年 ReAct 的做法——用 prompt 逼 LLM 输出 JSON 再 parse"。

### 3. 检索编排
混合检索（BM25 + 向量）的**融合逻辑**（怎么加权、怎么合并排序）、rerank 的编排。
检索的"怎么组织"自己写；底层 embedding/向量库用库。

### 4. 评测脚本
指标计算、LLM-as-judge 的 prompt 设计、评测集构造。**这是简历上最能讲、也最少人做的部分。**

---

## 三、可以用库（commodity，别浪费时间自己造）

| 用库 | 选型 | 原因 |
|---|---|---|
| embedding | bge-m3（硅基流动/魔搭 API 或 sentence-transformers）| 不用自己训 embedding |
| 向量库 | Chroma → 后面换 Qdrant | 不用自己写向量检索底层 |
| LLM | DeepSeek SDK（openai 兼容）| 你已有 key |
| BM25 | `rank_bm25`（或自己写，只有几十行，手写更加分）| 简单 |
| 文档解析 | pypdf / unstructured | 不用手写 PDF 解析 |
| Web 框架 | FastAPI + Gradio | 部署用，不用手写 |

---

## 四、评测体系（简历上能讲的部分）

1. **自建 QA 集**：从你的文档里抽 50–100 个真实问题，每题标注【标准答案 + 答案出处在哪几段】。
2. **检索评测**（离线，不用调 LLM，快）：Recall@k、MRR、nDCG——衡量"答案出处有没有被检索到"。
3. **端到端评测**：忠实度（答案是否有引用支撑，防幻觉）+ 正确性（LLM-as-judge 对标准答案打分）。
4. **消融对比实验**（这个最值钱，直接证明你懂"为什么这么做"）：
   - BM25 vs 向量 vs 混合 vs 混合+rerank；
   - 单次检索 vs Agent 多步检索。
   - 结论示例："加 rerank 后 Recall@5 从 0.62 → 0.81，因为……"——面试就能讲出因果。

---

## 五、里程碑（压缩到 ~8–9 周，11 月底前可写进简历）

> 记住优先级：**成绩 > 项目 > 刷题 > 八股**。项目要能 demo、能讲清，不必完美。

1. **里程碑 1（~1 周）**：`minimal_rag.py` 升级为真实 embedding + 向量库 + DeepSeek API 的 RAG 基线。跑通数据管道，先有"能回答"的底座。
2. **里程碑 2（~2 周）**：手写检索底座——分块策略实验（块大小/重叠）、BM25 + 向量混合检索、rerank 接入。能讲"为什么混合检索 + rerank 有效"。
3. **里程碑 3（~2 周）**：手写 Agent 主循环 + 工具调用协议。把检索包成工具，Agent 学会"该不该查、查什么、查几次、答不上怎么办"。至少 3 个工具（search / rerank / final_answer）。
4. **里程碑 4（~2 周）**：评测体系。自建 QA 集 + 跑 Recall@k/MRR/忠实度 + 消融对比。
5. **里程碑 5（~1–2 周）**：FastAPI + Gradio 部署、写 README、写技术博客（把面试那 25 道题的原理沉淀成文章）。

---

## 六、面试"讲清楚"的三件套（每个里程碑都对应）

- **解决的问题**：文档问答里"单次检索答不准、答不上、答案不可信"→ 用 Agent 多步决策 + 混合检索 + 溯源解决。
- **方案权衡**：为什么手写 ReAct 而不是调 LangChain；为什么混合检索而不是纯向量；为什么加 rerank。
- **踩过的坑**：分块大小影响召回、LLM 工具调用偶发解析失败要重试、Agent 死循环要限步数……（边做边记，做成"踩坑清单"写进 README）。
