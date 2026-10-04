"use strict";

const CAT_LABEL = { doc: "📄 文档配置", code: "🧩 源码实现", gen: "💻 代码需求" };
// 英文建议问题 → 中文（仅界面显示用；点击后仍用英文原文查询，因为语料与 embedding 均为英文）。
const ZH = {
  "How does vLLM dynamically load and unload LoRA adapters at runtime without restarting the server?": "vLLM 如何在运行时动态加载和卸载 LoRA 适配器而无需重启服务？",
  "What quantization schemes and calibration approaches does vLLM support for the FP8 KV cache?": "vLLM 为 FP8 KV 缓存支持哪些量化方案和校准方法？",
  "What KV transfer connectors does vLLM support for disaggregated prefilling?": "vLLM 支持哪些用于分离式预填充的 KV 传输连接器？",
  "How do you configure the n-gram prompt lookup window in vLLM's speculative decoding?": "如何在 vLLM 的投机解码中配置 n-gram prompt 查找窗口？",
  "What backends does vLLM support for generating structured outputs?": "vLLM 支持哪些用于生成结构化输出的后端？",
  "How do you limit the number of tokens a reasoning model spends producing its reasoning block?": "如何限制推理模型生成推理块所消耗的 token 数？",
  "What is the default hashing algorithm for vLLM's prefix caching, and which alternatives provide reproducible cross-language hashing?": "vLLM 前缀缓存的默认哈希算法是什么？哪些替代方案提供可复现的跨语言哈希？",
  "What is the default preemption mode in vLLM V1, and why was it chosen over the alternative?": "vLLM V1 的默认抢占模式是什么？为什么选择它而不是替代方案？",
  "What class acts as the central controller for CUDA Graphs dispatch in vLLM v1, and what structure is used as the dispatch key?": "vLLM V1 中哪个类充当 CUDA Graphs 分发的中央控制器？用什么结构作为分发键？",
  "What two mechanisms does vLLM use to recreate the Hugging Face processor's output for multimodal inputs without seeing the original text?": "vLLM 用哪两种机制在看不到原始文本的情况下重建 Hugging Face processor 的多模态输入输出？",
  "What CUDA kernel does vLLM's paged attention implement, and what warp-shuffle primitive reduces qk_max across warps?": "vLLM 的 paged attention 实现了哪个 CUDA kernel？用什么 warp-shuffle 原语跨 warp 归约 qk_max？",
  "What data class is the basic building block for prefix caching in vLLM v1, and which KV cache manager methods does the scheduler call to fetch computed blocks and allocate slots?": "vLLM V1 中前缀缓存的基本构建块是哪个数据类？调度器调用 KV 缓存管理器的哪些方法来取已计算块和分配槽位？",
  "Which coordinator does vLLM's hybrid KV cache manager choose when a model has exactly two KV cache groups, and which one handles a single group?": "当模型恰好有两个 KV 缓存组时，vLLM 的混合 KV 缓存管理器选择哪个协调器？单个组由哪个处理？",
  "What two command-line flags specify the attention backend, and which optional backend targets NVIDIA SM120/SM121 GPUs?": "指定 attention 后端的两个命令行标志是什么？哪个可选后端针对 NVIDIA SM120/SM121 GPU？",
  "What unified constructor signature does vLLM require all model classes to implement, and which async engine class serves online requests?": "vLLM 要求所有模型类实现的统一构造函数签名是什么？哪个异步引擎类服务在线请求？",
  "Which environment variable controls vLLM's Python multiprocessing start method, and which one was used in v1 to enable multiprocessing in the engine core?": "哪个环境变量控制 vLLM 的 Python 多进程启动方法？V1 中用哪个在引擎核心里启用多进程？",
  "What entry point group name does vLLM use to register general plugins, and which function loads plugins into every process?": "vLLM 用哪个入口点组名注册通用插件？哪个函数把插件加载到每个进程？",
  "Which environment variable disables vLLM's torch.compile cache, and which config class controls dynamic shapes behavior?": "哪个环境变量禁用 vLLM 的 torch.compile 缓存？哪个配置类控制动态形状行为？",
  "Which quantization fusions does vLLM's -O1 optimization level enable, and which kernel config flag does it turn on?": "vLLM 的 -O1 优化级别启用了哪些量化融合？它打开了哪个 kernel 配置标志？",
  "When Hugging Face cannot find a model_type in transformers, which config.json field does it use to locate the config class, and what token environment variable does vLLM pass to download the model?": "当 Hugging Face 在 transformers 中找不到 model_type 时，它用 config.json 的哪个字段定位配置类？vLLM 传递哪个 token 环境变量来下载模型？",
  "What are vLLM's training-free video token pruning algorithms, and which flag enables the feature?": "vLLM 的无训练视频 token 剪枝算法有哪些？哪个标志启用该特性？",
  "Which flag is mandatory to enable automatic tool choice, and which environment variable disables structural-tag enforcement for tool calling?": "启用自动工具选择必须设哪个标志？哪个环境变量禁用工具调用的结构标签强制？",
  "What speculative_config fields configure an EAGLE draft model, and which method string selects the EAGLE3 variant?": "哪些 speculative_config 字段配置 EAGLE draft 模型？哪个方法字符串选择 EAGLE3 变体？",
  "Which flags pin GPU workers to NUMA nodes, and what numactl syntax do they use for explicit CPU binding?": "哪些标志把 GPU worker 固定到 NUMA 节点？它们用什么 numactl 语法做显式 CPU 绑定？",
  "What environment variable controls the KV cache size for the CPU backend, and which processor argument reduces processed image size for Qwen2-VL models?": "哪个环境变量控制 CPU 后端的 KV 缓存大小？哪个 processor 参数减小 Qwen2-VL 模型处理后的图像尺寸？",
  "What asynchronous queue APIs does vLLM's LLM class offer for non-blocking generation, and which method resets the multimodal cache?": "vLLM 的 LLM 类提供哪些用于非阻塞生成的异步队列 API？哪个方法重置多模态缓存？",
  "Which LLM method executes a callable collectively across all workers, and which one resets the prefix cache?": "哪个 LLM 方法在所有 worker 上集体执行一个可调用对象？哪个方法重置前缀缓存？",
  "Which flag selects the MLA prefill backend, and which decode backend serves DeepSeek V4 sparse MLA by default on SM12x?": "哪个标志选择 MLA 预填充后端？哪个解码后端默认在 SM12x 上服务 DeepSeek V4 稀疏 MLA？",
  "Which utility splits long audio into chunks at quiet points for Whisper transcription, and what argument caps each chunk's length?": "哪个工具把长音频在静音点切块用于 Whisper 转录？哪个参数限制每块长度？",
  "Which tool parser handles models that emit Pythonic list-based tool calls instead of JSON, and what manager registers custom parsers?": "哪个工具解析器处理输出 Python 风格列表式工具调用的模型？哪个管理器注册自定义解析器？",
  "In vLLM V1, which scheduling budget governs how many tokens a prefill chunk may contain, and which latency metric improves because decode is prioritized?": "vLLM V1 中哪个调度预算决定预填充块可包含多少 token？由于解码被优先处理，哪个延迟指标得到改善？",
  "Which environment variable switches vLLM's BPE tokenizer to the Rust backend, and what exception does vLLM raise at tokenizer load if the package is missing?": "哪个环境变量把 vLLM 的 BPE 分词器切换到 Rust 后端？缺少该包时 vLLM 在分词器加载时抛出什么异常？",
  "How do you size vLLM's multi-modal processor cache, and which cache layout keeps data in shared memory across worker processes?": "如何设置 vLLM 多模态处理器缓存的大小？哪种缓存布局把数据保存在跨 worker 进程的共享内存中？",
  "Which environment variables control OpenMP thread placement, the reserved CPU count, and the visible memory nodes for vLLM's CPU backend?": "哪些环境变量控制 vLLM CPU 后端的 OpenMP 线程放置、保留 CPU 数和可见内存节点？",
  "Which dual-mode CUDA graph configurations switch dynamically between full and piecewise graphs at runtime, and which enum defines these modes?": "哪些双模 CUDA graph 配置在运行时动态切换完整图和分段图？哪个枚举定义这些模式？",
  "Which config option selects the FlashAttention version, and what are the defaults on Blackwell versus Hopper GPUs?": "哪个配置项选择 FlashAttention 版本？Blackwell 和 Hopper GPU 上的默认值分别是什么？",
  "Which speculative-decoding keys bound the suffix-tree depth, the number of cached requests, and the speculative length multiple?": "哪些投机解码键约束后缀树深度、缓存请求数和投机长度倍数？",
  "Which speculative-decoding key selects how draft tokens are verified, and which two synthetic-mode settings supply acceptance rates or a target mean length?": "哪个投机解码键选择如何验证 draft token？哪两个合成模式设置提供接受率或目标平均长度？",
  "Which pooling APIs does the LLM class offer for embedding, classification, and scoring models?": "LLM 类为嵌入、分类和打分模型提供哪些池化 API？",
  "Which LLM methods initialize and drive the weight-transfer cycle used for RL training?": "哪些 LLM 方法初始化和驱动用于 RL 训练的权重传输循环？",
  "Which LLM methods start and stop a profiling session, and what optional argument names the trace?": "哪些 LLM 方法启动和停止性能分析会话？哪个可选参数命名 trace？",
  "Which LLM argument caps the number of multi-modal items per prompt, and which config classes describe the per-modality size hints for images and videos?": "哪个 LLM 参数限制每个 prompt 的多模态项数量？哪些配置类描述图像和视频的每种模态大小提示？",
  "Which config flag offloads image normalization to the GPU, and which module folds the rescale factor into its weight and bias?": "哪个配置标志把图像归一化卸载到 GPU？哪个模块把缩放因子折叠进其权重和偏置？",
  "Which KV-transfer connectors support CPU offloading, a distributed KV store, and ROCm-only disaggregation?": "哪些 KV 传输连接器支持 CPU 卸载、分布式 KV 存储和仅 ROCm 的分离？",
  "Besides xgrammar and guidance, which structured-output backends use Rust-style regex, and which one uses Python's re module?": "除了 xgrammar 和 guidance，哪些结构化输出后端使用 Rust 风格正则？哪个使用 Python 的 re 模块？",
  "Which process coordinates data-parallel ranks, and over what transport and topology do API servers reach engine cores?": "哪个进程协调数据并行 rank？API 服务器通过什么传输和拓扑到达引擎核心？",
  "Which flag registers a custom tool-parser plugin, and which class do you subclass to implement one?": "哪个标志注册自定义工具解析器插件？实现一个需要继承哪个类？",
  "Which flag selects the reasoning parser, and which two CLI arguments define the boundary tokens and server-wide chat-template defaults?": "哪个标志选择推理解析器？哪两个 CLI 参数定义边界 token 和服务端全局聊天模板默认值？",
  "Which flags configure Mamba's fine-grained prefix caching, and which one sets the granularity of prefix-cache keys?": "哪些标志配置 Mamba 的细粒度前缀缓存？哪个设置前缀缓存键的粒度？",
  "In the --kv-transfer-config JSON, which three keys name the connector, its role, and the buffer device?": "在 --kv-transfer-config JSON 中，哪三个键命名连接器、其角色和缓冲设备？",
  "Which proposer implements MEDUSA-style drafting by stacking several classification heads on top of the target model's hidden states?": "哪个 proposer 通过在目标模型隐藏状态上堆叠多个分类头来实现 MEDUSA 风格的草稿生成？",
  "For the multi-module multi-token-prediction path, how does vLLM prepare each draft module's input hidden states and embeddings before running them together?": "对于多模块多 token 预测路径，vLLM 在把它们一起运行之前如何准备每个 draft 模块的输入隐藏状态和嵌入？",
  "Which speculator subclass specializes the autoregressive drafting path for EAGLE-style draft models?": "哪个 speculator 子类专门处理 EAGLE 风格 draft 模型的自回归草稿生成路径？",
  "Which proposer extracts hidden states from the target model so they can be fed to a draft model for EAGLE-style speculative decoding?": "哪个 proposer 从目标模型提取隐藏状态，以便喂给 draft 模型做 EAGLE 风格投机解码？",
  "Which class verifies draft tokens against the target model's token probabilities and accepts or rejects each one?": "哪个类对照目标模型的 token 概率验证 draft token 并逐个接受或拒绝？",
  "How does the suffix-matching proposal method look up the current token suffix against previously seen sequences to propose a continuation?": "后缀匹配的提案方法如何在之前见过的序列中查找当前 token 后缀来提议延续？",
  "How does vLLM adaptively decide how many draft tokens to verify per request, based on measured per-step cost curves?": "vLLM 如何根据测得的每步成本曲线，自适应地决定每个请求要验证多少个 draft token？",
  "How does the multi-token-prediction path reuse the target model's prefill step to seed the draft model's decode?": "多 token 预测路径如何复用目标模型的预填充步骤来为 draft 模型的解码做种子？",
  "When the draft model and the target model use different tokenizers, which component translates token IDs from one vocabulary to the other?": "当 draft 模型和目标模型使用不同分词器时，哪个组件把 token ID 从一个词表翻译到另一个？",
  "In the n-gram drafting method, how does vLLM find the longest run of tokens in the context that matches the current suffix, and turn the tokens that follow it into draft candidates?": "在 n-gram 草稿生成方法中，vLLM 如何在上下文中找到与当前后缀匹配的最长 token 串，并把紧随其后的 token 变成 draft 候选？",
  "When running n-gram drafting on GPU, how does vLLM keep its on-device token-id and token-count tables updated incrementally as accepted tokens extend the sequence?": "在 GPU 上运行 n-gram 草稿生成时，vLLM 如何在被接受的 token 扩展序列时增量更新其设备上的 token-id 和 token-count 表？",
  "Which data structure holds the per-request bookkeeping for one speculative decoding step, such as the list of candidate tokens and how many were accepted?": "哪个数据结构保存一次投机解码步骤的每请求记账，例如候选 token 列表和被接受的数量？",
  "The rejection-sampling step needs a log-sum-exp of the target model's probabilities over the whole vocabulary; how does vLLM combine the per-block results into that global value?": "拒绝采样步骤需要目标模型在整个词表上的概率的 log-sum-exp；vLLM 如何把每块的结果合并成这个全局值？",
  "Which drafting component runs a standalone draft model token-by-token in a loop, handling its loading, attention setup, and CUDA graph capture to propose candidates?": "哪个草稿生成组件在循环中逐个 token 运行独立的 draft 模型，处理其加载、attention 设置和 CUDA graph 捕获来提议候选？",
  "Which drafting path proposes all of its speculative tokens in a single forward pass of a separate draft model, in the DFlash parallel-drafting technique?": "哪个草稿生成路径在 DFlash 并行草稿技术中，通过独立 draft 模型的一次前向传播提议所有投机 token？",
  "Given the speculative decoding settings in the config, which entry point selects and constructs the right proposer for the requested method?": "给定配置中的投机解码设置，哪个入口点为所请求的方法选择并构造正确的 proposer？",
  "For the Gemma family, how does the speculative decoding path set up the multi-group KV cache and the centroid masking its lightweight prediction heads rely on?": "对于 Gemma 家族，投机解码路径如何设置多组 KV 缓存和其轻量预测头所依赖的质心掩码？",
  "Write a program that performs offline batch inference with vLLM, enabling FP8 KV-cache quantization (kv_cache_dtype=\"fp8\") and automatic prefix caching.": "编写一个程序，用 vLLM 执行离线批处理推理，启用 FP8 KV 缓存量化（kv_cache_dtype=\"fp8\"）和自动前缀缓存。",
  "Write a program that instantiates a vLLM LLM and sets max_num_batched_tokens=16384 to control the chunked-prefill scheduling budget.": "编写一个程序，实例化 vLLM LLM 并设置 max_num_batched_tokens=16384 来控制分块预填充调度预算。",
  "Write a program that loads a base model with LoRA support and issues a generation request tagged with a LoRA adapter via LoRARequest.": "编写一个程序，加载支持 LoRA 的基础模型，并通过 LoRARequest 发出带 LoRA 适配器标签的生成请求。",
  "Write a program that uses vLLM's structured outputs API to constrain generation to two sentiment choices (positive / negative).": "编写一个程序，用 vLLM 的结构化输出 API 把生成约束为两种情感选择（positive / negative）。",
  "Write a program that enables n-gram speculative decoding with a prompt lookup window bounded by prompt_lookup_min=2 and prompt_lookup_max=5.": "编写一个程序，启用 n-gram 投机解码，prompt 查找窗口由 prompt_lookup_min=2 和 prompt_lookup_max=5 界定。",
  "Write a function rrf_fusion(rankings, k=60) where rankings is a list of lists (each an ordered list of item ids, best first), returning a dict of item id → Reciprocal Rank Fusion score (sum over lists of 1/(k + rank), rank 1-based).": "编写函数 rrf_fusion(rankings, k=60)，rankings 是列表的列表（每个是有序的 item id 列表，最好在前），返回 item id → Reciprocal Rank Fusion 分数的字典（对所有列表求和 1/(k + rank)，rank 从 1 计）。",
  "Write a function kv_cache_bytes(num_tokens, num_layers, num_kv_heads, head_dim, bytes_per_elem=2) that returns the KV cache size in bytes = num_tokens * num_layers * 2 (K and V) * num_kv_heads * head_dim * bytes_per_elem.": "编写函数 kv_cache_bytes(num_tokens, num_layers, num_kv_heads, head_dim, bytes_per_elem=2)，返回 KV 缓存字节大小 = num_tokens * num_layers * 2（K 和 V）* num_kv_heads * head_dim * bytes_per_elem。",
  "Write a function next_power_of_2(n) that returns the smallest power of two greater than or equal to n, for positive integer n.": "编写函数 next_power_of_2(n)，返回大于等于 n 的最小 2 的幂，n 为正整数。",
  "Write the SpecDecodeMetadata dataclass to workspace/spec_metadata.py, matching vLLM's vllm/v1/spec_decode/metadata.py exactly: same field names, field types, and __post_init__ behavior.": "编写 SpecDecodeMetadata 数据类到 workspace/spec_metadata.py，与 vLLM 的 metadata.py 完全一致：字段名、字段类型和 __post_init__ 行为。",
  "Write a function accept_draft_token(target_log_prob, draft_log_prob, u, target_argmax, draft_token, is_greedy) to workspace/accept.py that replicates vLLM's rejection-sampling accept decision from vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py: greedy mode and the non-greedy probability-ratio test.": "编写函数 accept_draft_token(...) 到 workspace/accept.py，复现 vLLM 的拒绝采样接受决策：贪心模式和非贪心概率比检验。",
  "Write a function get_max_chunk_logits(vocab_size) to workspace/max_chunk.py that replicates vLLM's get_max_chunk_logits from vllm/v1/worker/gpu/spec_decode/rejection_sampler.py: the largest number of logits rows one verification chunk may hold, capped by a fixed FP32 target-logits buffer budget.": "编写函数 get_max_chunk_logits(vocab_size) 到 workspace/max_chunk.py，复现 vLLM 的 get_max_chunk_logits：一个验证块可容纳的最大 logits 行数，受固定 FP32 目标 logits 缓冲预算约束。",
  "The function next_power_of_2 in workspace/spec_helper.py is buggy: it returns 2 for n=1 (and is wrong for every exact power of two). Read the file with read_file, then write the corrected function back with edit_file.": "workspace/spec_helper.py 中的 next_power_of_2 函数有 bug：它对 n=1 返回 2（且对每个精确的 2 的幂都是错的）。用 read_file 读取文件，再用 edit_file 写回修正后的函数。",
};
const state = { suggestions: [], loading: false, hasAsked: false };

const $ = (sel) => document.querySelector(sel);
const chatEl = $("#chat");
const inputEl = $("#input");
const suggestGroupsEl = $("#suggestGroups");
const suggestPanel = $("#suggestPanel");
const collapseBtn = $("#collapseBtn");
const emptyHintEl = $("#emptyHint");

// ---------- 鉴权 ----------
// 网关默认开启 JWT 鉴权：页面加载时用 demo 凭据取 token，缓存到 localStorage 复用，
// 避免每次刷新都打 /api/token（该端点有 5 次/分钟限流）。生产环境应换成真实登录，
// 凭据与 token 都不应出现在前端。
const AUTH_TOKEN_KEY = "copilot_token";
let authToken = safeGet(AUTH_TOKEN_KEY);

function safeGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function safeSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
function safeDel(k) { try { localStorage.removeItem(k); } catch (e) {} }

async function ensureToken() {
  if (authToken) return authToken;
  const res = await fetch("/api/token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: "admin", password: "admin123" }),
  });
  if (!res.ok) throw new Error("获取访问令牌失败（HTTP " + res.status + "）");
  const data = await res.json();
  authToken = data.token;
  safeSet(AUTH_TOKEN_KEY, authToken);
  return authToken;
}

// authFetch：自动带 Bearer token；遇 401 清缓存重新取 token 再重试一次。
async function authFetch(url, opts = {}) {
  opts.headers = Object.assign({}, opts.headers, { Authorization: "Bearer " + (await ensureToken()) });
  let res = await fetch(url, opts);
  if (res.status === 401) {
    authToken = null;
    safeDel(AUTH_TOKEN_KEY);
    opts.headers.Authorization = "Bearer " + (await ensureToken());
    res = await fetch(url, opts);
  }
  return res;
}

// ---------- 转义 & 极简 markdown ----------
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inlineMd(s) {
  return escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

function renderMd(src) {
  const lines = String(src || "").split("\n");
  let html = "", inUl = false, inOl = false, inCode = false, codeBuf = [], codeLang = "";
  const flush = () => {
    if (inUl) { html += "</ul>"; inUl = false; }
    if (inOl) { html += "</ol>"; inOl = false; }
  };
  const flushCode = () => {
    html += `<pre><code class="lang-${codeLang}">${escapeHtml(codeBuf.join("\n"))}</code></pre>`;
    codeBuf = []; codeLang = ""; inCode = false;
  };
  for (const line of lines) {
    const fence = /^```(\w*)\s*$/.exec(line);
    if (fence) {
      if (inCode) flushCode();
      else { flush(); codeLang = fence[1]; inCode = true; }
      continue;
    }
    if (inCode) { codeBuf.push(line); continue; }
    if (!line.trim()) { flush(); continue; }
    let m;
    if ((m = /^(#{1,6})\s+(.*)$/.exec(line))) { flush(); html += `<h${m[1].length}>${inlineMd(m[2])}</h${m[1].length}>`; }
    else if ((m = /^\s*[-*+]\s+(.*)$/.exec(line))) { if (!inUl) { html += "<ul>"; inUl = true; } html += `<li>${inlineMd(m[1])}</li>`; }
    else if ((m = /^\s*(\d+)\.\s+(.*)$/.exec(line))) { if (!inOl) { html += "<ol>"; inOl = true; } html += `<li>${inlineMd(m[2])}</li>`; }
    else { flush(); html += `<p>${inlineMd(line)}</p>`; }
  }
  if (inCode) flushCode();
  flush();
  return html;
}

// ---------- 建议问题 ----------
async function loadSuggestions() {
  const btn = $("#refreshBtn");
  btn.disabled = true;
  try {
    const res = await authFetch("/api/suggestions");
    const data = await res.json();
    state.suggestions = data.suggestions || [];
    renderSuggestions();
  } catch (e) {
    suggestGroupsEl.innerHTML = `<div class="empty-hint">加载建议失败：${escapeHtml(String(e))}</div>`;
  } finally {
    btn.disabled = false;
  }
}

function renderSuggestions() {
  const groups = { doc: [], code: [], gen: [] };
  state.suggestions.forEach((s, i) => {
    s._i = i;
    (groups[s.category] || groups.doc).push(s);
  });
  let html = "";
  for (const cat of ["doc", "code", "gen"]) {
    const items = groups[cat];
    if (!items.length) continue;
    html += `<div class="sug-group"><div class="sug-cat">${CAT_LABEL[cat]}</div>`;
    for (const it of items) {
      html += `<button class="sug-chip sug-${cat}" data-i="${it._i}">${escapeHtml(ZH[it.text] || it.text)}</button>`;
    }
    html += "</div>";
  }
  suggestGroupsEl.innerHTML = html;
  suggestGroupsEl.querySelectorAll(".sug-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const it = state.suggestions[+chip.dataset.i];
      if (it) ask(it.text);
    });
  });
}

function setCollapsed(collapsed) {
  suggestPanel.classList.toggle("collapsed", collapsed);
  collapseBtn.textContent = collapsed ? "展开 ▾" : "收起 ▴";
  collapseBtn.title = collapsed ? "展开建议" : "收起建议";
}

// ---------- 对话 ----------
function addMessage(role, text, contexts) {
  if (emptyHintEl) emptyHintEl.remove();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML =
    `<div class="role">${role === "user" ? "你" : "Copilot"}</div>` +
    `<div class="bubble">${role === "user" ? escapeHtml(text) : renderMd(text)}</div>`;
  if (role === "assistant" && contexts && contexts.length) {
    const cites = contexts.map((c) => `<div class="ctx">${escapeHtml(c)}</div>`).join("");
    wrap.innerHTML += `<details class="citations"><summary>📎 检索依据（${contexts.length} 条）</summary>${cites}</details>`;
  }
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return wrap;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.innerHTML = `<div class="role">Copilot</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return wrap;
}

let pendingQuery = null; // 点建议时暂存的英文原文（界面显示中文、实际查询仍发英文）

function ask(text) {
  inputEl.value = ZH[text] || text; // 输入框显示中文
  pendingQuery = text;              // 但查询发英文原文（语料/embedding 是英文）
  submit();
}

async function submit() {
  const q = inputEl.value.trim();
  if (!q || state.loading) return;
  const query = pendingQuery || q; // 点建议 → 英文原文；手动输入 → 输入内容
  pendingQuery = null;
  if (!state.hasAsked) { state.hasAsked = true; setCollapsed(true); }
  inputEl.value = "";
  inputEl.style.height = "auto";
  addMessage("user", q);
  const typing = addTyping();
  state.loading = true;
  $("#sendBtn").disabled = true;
  try {
    const res = await authFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: query }),
    });
    const data = await res.json();
    typing.remove();
    if (data.error) addMessage("assistant", "⚠️ " + data.error);
    else addMessage("assistant", data.answer, data.contexts);
  } catch (e) {
    typing.remove();
    addMessage("assistant", "⚠️ 请求失败：" + e);
  } finally {
    state.loading = false;
    $("#sendBtn").disabled = false;
  }
}

// ---------- 事件 ----------
$("#refreshBtn").addEventListener("click", loadSuggestions);
$("#collapseBtn").addEventListener("click", () => setCollapsed(!suggestPanel.classList.contains("collapsed")));
$("#sendBtn").addEventListener("click", submit);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submit(); }
});
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});

loadSuggestions();
