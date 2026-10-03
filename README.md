# vLLM 工程 Copilot

> 给 **vLLM 源码 + 官方文档** 做的问答 Copilot：`RAG 双索引（文档 + 代码）+ ReAct Agent`。
> 主打不是「能答」，而是**把评测口径本身做成了研究对象**——用 20 个受控实验回答一个反直觉的问题：
> *「代码检索的 0.94 命中率，有多少是检索能力，有多少只是题面里白送的字面线索？」*
> 答案（实验 20，配对差值 + McNemar 精确检验）：**+0.53 是术语溢价，不是能力**。

---

## 30 秒看懂这个项目

| 维度 | 做了什么 | 关键数字 |
|---|---|---|
| **检索** | 文档库（markdown 结构切块 + small-to-big）与代码库（AST 符号切块 + LLM 描述合成）双索引；向量 + BM25 混合召回 → RRF 融合 → bge-reranker 重排 | 代码索引 48 个 .py → 523 chunk |
| **Agent** | ReAct 循环，7 个工具（检索文档/检索代码/跑 Python/读/写/改文件），子进程隔离 | deepseek-chat 驱动 |
| **评测** | 19 个渐进实验 + 1 个两档配对测量，结论全部带统计检验 | 见下方结果表 |

**真正的卖点是评测，不是 demo。** 全套实验记录在 [`experiments.md`](experiments.md)，每个结论都能复现、每步都有脚本、每条都被数据而非直觉裁决。

---

## 架构

```
                        ┌─────────────────────────────────────────────┐
  用户问题               │                 ReAct Agent                 │
      │                 │  route: write_code / run_python / search    │
      ▼                 │  工具: search / search_code / run_python /   │
 ┌─────────────┐        │        run_file / read / write / edit        │
 │  LLM 路由    │──────▶ │  （子进程隔离，P0-2 权限门）                  │
 │ deepseek    │        └───────┬──────────────────────┬──────────────┘
 └─────────────┘                │                      │
                                ▼                      ▼
                     ┌──────────────────┐   ┌──────────────────────────┐
                     │  文档索引         │   │  代码索引                 │
                     │  vLLM docs .md   │   │  vLLM src/ (AST 符号切块) │
                     │  结构切块+父块     │   │  LLM 描述合成(锚点, 0.3)  │
                     └────────┬─────────┘   └──────────┬───────────────┘
                              │                        │
                              ▼                        ▼
                 ┌─────────────────────────────────────────────┐
                 │  混合召回: 向量(bge-base) + BM25 → RRF 融合   │
                 │      → 交叉编码器重排(bge-reranker-base)      │
                 └─────────────────────────────────────────────┘
```

- **文档检索**（`rag_baseline.py`）：markdown 按标题结构切块，small-to-big 带父上下文。
- **代码检索**（`code_chunker.py` / `code_index.py` / `docstring_synth.py`）：把源码按 AST 符号（函数/类/常量/字段）切块，**用 LLM 为每个符号合成一句英文描述作为嵌入锚点**（`OVERVIEW_WEIGHT=0.3`），让「函数名和问题字面不重合」时仍能被向量召回。
- **模型**：`BAAI/bge-base-en-v1.5`（嵌入）、`BAAI/bge-reranker-base`（重排）、`deepseek-chat`（生成/合成）。

### 服务化：Go 网关 + gRPC 分层（Python 做脑、Go 做壳）

```
浏览器 / 客户端
   │  HTTP (JSON)
   ▼
Go 网关 (Gin)  ── 融入：JWT 鉴权 / Redis 缓存 / 限流
   │  gRPC (protobuf)
   ▼
Python gRPC 服务 ── agent.run 原样包装，索引常驻内存
   │
   ▼
检索（双索引）+ ReAct + DeepSeek 生成
```

- **`proto/copilot.proto`**：Go 与 Python 共用的 gRPC 接口（`Copilot.Run`：query → answer + contexts；`Copilot.Suggestions`：返回评测集抽样的建议问题）。
- **`grpc_server/`（Python）**：`import agent → get_client → load_index` 一次性加载索引常驻内存，`server.py` 起 gRPC（`:50051`）；服务面默认只开 `search` + `search_code`，不开 run_python/run_file/write（最安全）。
- **`gateway/`（Go）**：Gin 起 HTTP（`:8080`），gRPC client 连 Python 服务；`/api/chat` 走「限流 → JWT 鉴权 → Redis 缓存(cache-aside) → gRPC Run」，`/api/suggestions` 转发建议问题，`/api/token` 签发 JWT；同时托管前端页面（`/` 与 `/static/`，复用项目根 static/，与 webapp.py 同一份文件）。
- **为什么这么分**：检索/推理是 Python 生态（PyTorch、sentence-transformers）的护城河，保持不动；网关、鉴权、缓存、限流是 Go 微服务的强项，独立成壳——两端用 protobuf 契约解耦，可分别部署/扩缩容。

---

## 核心结果：两档测量（实验 20）

问题在于：实验 18/19 把代码题改写成「高层角色描述」（去同源）后 R@10 掉到 0.44，但这一刀砍过头了——
真实用户提问时，知道类名的会说「MEDUSA 那个 proposer」，不知道的才说「那种叠几个分类头的」。
于是把「题面里有没有术语线索」拆成两档，逐题配对，量化它到底值多少分。

**设计**：同一索引、同一 reranker、同一 top-10，只换题面口径（自然 17 题 vs 抽象 17 题，同 id 同 gold）。
Δ = 自然命中率 − 抽象命中率，McNemar 精确检验 + bootstrap CI + Wilcoxon。

**L1 配对差值（主指标 R@10，主口径 S1 现行 RRF）**

| 分组 | n | 自然 R@10 | 抽象 R@10 | Δ | 95% CI | McNemar p |
|---|---|---|---|---|---|---|
| A 组 | 8 | 1.000 | 0.375 | +0.625 | [+0.250, +0.875] | 0.062 |
| B 组 | 9 | 0.889 | 0.444 | +0.444 | [+0.111, +0.778] | 0.125 |
| **合计** | **17** | **0.941** | **0.412** | **+0.529** | **[+0.294, +0.765]** | **0.004** |

**L2 补充**：ΔMRR = +0.508（Wilcoxon p=0.003），名次差中位 −10。

### 三个结论

1. **「术语线索」值 +0.53 R@10，全程 c=0**（没有一道「抽象命中而自然不中」）。自然档 0.94 里大部分是
   题面里词根/符号名的字面重合，不是检索能力。**对外能力口径修正为**：不点名词时 R@10 ≈ **0.41**，
   点名词时 0.94，差值 0.53 是术语溢价。诚实估计当前代码检索能力 = **0.41**（真实用户更常不点名）。
2. **缺口在召回，不在重排**：抽象档召回覆盖率仅 **0.53**（自然 0.94），8/17 题的 gold 压根没进 top-20。
   语义召回（不知道类名、只描述功能）才是真短板。
3. **BM25 的角色按题面翻转**——修正实验 10「混合检索被否掉」的结论：抽象档纯向量 0.53 > RRF 0.41
   （BM25 是噪声）；自然档 RRF 0.94 > 纯向量 0.82（BM25 是 +0.12 资产）。精确说法：**BM25 只在
   query 没有词面锚点时是噪声，有术语线索时是资产**——两条腿都要，但融合要按题型切（下一步的 router）。

> **口径声明**（写进任何对外表述）：① Δ 读「词根/符号线索值多少分」，不读「真实用户 vs 抽象」——
> 两臂题面都看着 gold 写出来；② 主指标 R@10 预指定；③ 截断常数 21（`RECALL_N`+1）。
> 完整 12 格消融表（2 档 × 6 融合策略）与逐题名次见 `eval_two_tier.py` 输出 / `two_tier_results.json`。

---

## 快速开始

```powershell
# 1. 依赖（Python 3.12）
py -3.12 -m pip install -r requirements.txt

# 2. 环境变量
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 国内镜像；装完模型后可 HF_HUB_OFFLINE=1
$env:DEEPSEEK_API_KEY = "<你的 key>"          # 生成 + docstring 合成必需

# 3. 语料（corpus/ 整个目录不入库，见 .gitignore）
#    源码：解压 corpus/src/vllm-0.29.0.tar.gz 到 corpus/src/vllm-0.29.0/
#           （本机快照；新机器从 vLLM GitHub tag v0.29.0 重新下载）
#    文档：corpus/docs/ 为 vLLM 官方文档 markdown（vLLM 仓库 docs/source/ 下的 .md）

# 4. 跑起来
py -3.12 agent.py          # 交互式问答 Copilot
py -3.12 eval_two_tier.py  # 复现两档测量（12 格消融 + L1/L2 报告）

py -3.12 webapp.py         # Flask 网页版（纯 Python 直连路径，作对照）

# ---- 服务化（Go 网关 + gRPC，见「架构 · 服务化」）----
# 4.1 Python gRPC 服务（先起，首次加载索引约 1~6 分钟）
#     DEEPSEEK_API_KEY 放环境变量，或写入项目根 .env（已 gitignore，服务自动读取）
cd grpc_server && python3 server.py        # 监听 :50051

# 4.2 Go 网关（另开终端；需 go 1.26 + Redis 在 6379）
cd gateway && go run .                      # 监听 :8080，浏览器打开 http://localhost:8080/ 即前端

# 4.3 走网关问答
curl -X POST localhost:8080/api/chat -H "Content-Type: application/json" -d "{\"query\":\"What is vLLM?\"}"
# → {"answer":"...","contexts":["..."]}，与 webapp 的 /api/chat 同构
curl localhost:8080/api/suggestions          # 建议问题（取自评测集，走 gRPC）
```

---

## 目录结构

```
rag-project/
├── agent.py               # ReAct Agent：路由 + 7 个工具 + 子进程隔离
├── rag_baseline.py        # 文档检索：结构切块 + small-to-big + 嵌入
├── bm25.py / rerank.py    # BM25 与交叉编码器重排
├── code_chunker.py        # AST 符号切块
├── code_index.py          # 代码索引：嵌入 + 描述锚点(0.3) + 内容哈希缓存键
├── docstring_synth.py     # LLM 描述合成 + 缓存
├── config.py              # 切块/召回/上下文预算常量
├── eval_data.py           # 评测集（文档 QA + 代码 QA 双档 + 生成题）
├── eval_qa.py / eval_code.py / eval_codegen.py
├── eval_two_tier.py       # ★ 两档配对测量（实验 20 的脚本）
├── answer_eval.py / judge.py   # 生成质量评测
├── two_tier_results.json  # 实验 20 落盘结果
├── experiments.md         # ★ 20 个实验的完整记录（先读这个）
├── agent_roadmap.md       # Agent 设计路线
├── rag_qa.md              # 面试问答准备
├── proto/                 # ★ gRPC 接口定义（Go + Python 共用）
│   └── copilot.proto
├── grpc_server/           # ★ Python gRPC 服务（索引常驻内存）
│   ├── server.py
│   ├── Dockerfile         # python:3.10-slim + CPU torch
│   └── copilot_pb2*.py    # 生成
├── gateway/               # ★ Go 网关（Gin + gRPC client + JWT + Redis + 限流）
│   ├── main.go
│   ├── Dockerfile         # 多阶段 Go 构建（golang:1.26 → alpine）
│   ├── internal/{grpcclient,auth,cache,handler}/
│   └── config/config.yml.example
├── static/                # 前端（index.html / app.js / style.css，由 Go 网关托管）
├── docker-compose.yml     # ★ 三服务编排（redis + grpc-server + gateway）
├── .dockerignore
├── .github/workflows/ci.yml   # ★ Go CI（gofmt / vet / build / test）
├── loadtest/k6-chat.js        # ★ k6 压测脚本
└── docs/review-probes/    # 复审用的一次性探针脚本与结论（归档）
```

---

## Docker 部署（docker-compose）

三服务编排：`redis` + `grpc-server`（Python）+ `gateway`（Go），一键起整套。

```bash
# 1. 起三服务（首次 build 两个镜像，Python 侧装 CPU torch 较慢，属正常）
docker compose up -d --build

# 2. 等 Python 服务加载索引（首次约 1~6 分钟，走宿主机挂载的模型/语料）
docker compose logs -f grpc-server

# 3. 验证
curl localhost:8080/healthz
curl -X POST localhost:8080/api/chat -H "Content-Type: application/json" -d "{\"query\":\"What is vLLM?\"}"
```

- 镜像只含代码 + 依赖；模型（~2.9GB）与语料（156MB）从宿主机 **volume 挂载**，不进镜像。
- `DEEPSEEK_API_KEY` 从项目根 `.env` 经 compose 的 `${...}` 注入容器，不入镜像不入库。
- 本地开发仍是「`python3 server.py` + `go run .`」那套（见「快速开始」），与容器共用同一份代码。

## CI/CD（GitHub Actions）

push / PR 到 main 自动跑 gateway 的 Go 五步：`gofmt` 格式检查 → `go vet` 静态检查 → `go build` 编译 → `go test` 单测（bufconn 内存 gRPC + miniredis，无需真实依赖）。

配置在 `.github/workflows/ci.yml`。Python 镜像 build 不进 CI（装 torch 太慢），只做 Go 侧，保证每次 push 秒级反馈。

## 压测（k6）

用 k6 压 `/api/chat` 的缓存命中路径（网关 → 鉴权 → 限流 → Redis 缓存），脚本见 `loadtest/k6-chat.js`。

```bash
# 1. 抬高限流阈值（默认 60 次/分钟会让压测全打 429；实测缓存命中吞吐 ~6600 req/s、
#    50s 约 33 万请求，阈值需 10^7 量级——10 万会被限流拦掉，第一轮压测正是卡在 10 万处）
#    在项目根 .env 加一行 RATE_LIMIT=10000000，然后
docker compose up -d --force-recreate gateway

# 2. 跑压测（setup() 先预热一次缓存，之后全命中 Redis，不烧 DeepSeek）
k6 run loadtest/k6-chat.js                              # 终端汇总：QPS / p95 / 错误率
k6 run --out json=results.json loadtest/k6-chat.js      # JSON 报告
```

---

## 已知局限与下一步

1. **语义召回是真短板**：抽象档召回覆盖率 0.53，8/17 题 gold 没进 top-20。方向：描述合成的 prompt 改进、
   更大嵌入模型（实验 10 曾因 chunk 太小被否掉，需在 chunk 修好后再测）、查询扩展。
2. **chunk 大小是主导瓶颈**（实验 10，文档侧）：文档 chunk 切散仍是检索上限主因。代码侧「18.7% 块超 512 token 被静默截断」已改按 token 切分修掉（`MAX_CHUNK_TOKENS=460`，实测 0% 超线）；RERANK 最多喂 768 字符仍是有意保留的重排上限。
3. **router 第 4 步缺失**：实验 20 证明融合策略的价值按题型翻转，但 Agent 的 `route()` 目前还没有
   「按 query 是否含术语线索选融合」这一步——这是把 0.53 溢价「捡回来」的最直接落点。
4. **一条题面标注**：抽象档 dflash 题面「merges the main model's work into the same pass」是错误概括
   （DFlash 实际跑自己的 draft model），已冻结不改、仅标注；RejectionSampler 在 vLLM 有两处定义但
   索引只含 GPU spec_decode 那处，命中无歧义。

---

*20 个实验、每个结论的原始数据与复现脚本，见 [`experiments.md`](experiments.md)。*
