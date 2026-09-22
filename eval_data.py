# -*- coding: utf-8 -*-
"""
评测数据单一来源：题目 + gold 关键词集中在这里，eval_qa.py 和 judge.py 都从这里取。

之前题目散落在 eval_qa.py（QA_SET / QA_SEMANTIC）和 judge.py（EVAL_QUESTIONS）两处，
题面重复手写，改一处另一处漂移；gold 一旦两处不一致，_check_gold 的泄漏校验就失效。
这里收拢成一份，杜绝双份维护。

gold 是「答案段落里独有、别处几乎不出现」的精确技术词（缩写/参数名/类名），
已用 check_gold.py 校验：每题 3 个 gold 词能唯一定位到 1 个答案文档，避免泄漏造成假命中。

出题规范（check_gold 的 harm 模型）：
  - 硬要求：3 个 gold 词能同时出现在同一文档（唯一答案文档）；不能无解、不能多文档可解。
  - 优选：gold 词尽量只在答案文档出现；单词跨文档泄漏对「全 3 词」判定无害（其余词钉死答案），可接受。
  - chunk 大小是杠杆：切太碎（500）会把答案段落切散，3 词散到多个 chunk、Recall@1 恒 0；
    1000 字符后多数题 3 词已共现于单一 chunk。出题时看 check_gold 的 single_chunk 字段判断。
"""

# 50 道检索评测题，每题 3 个 gold 关键词。id 用于跨脚本稳定引用（judge 选题、扩充评测集时定位）。
QA_SET = [
    {"id": "lora", "q": "How does vLLM dynamically load and unload LoRA adapters at runtime without restarting the server?", "gold": ["VLLM_ALLOW_RUNTIME_LORA_UPDATING", "load_lora_adapter", "LoRAResolver"]},
    {"id": "fp8", "q": "What quantization schemes and calibration approaches does vLLM support for the FP8 KV cache?", "gold": ["llm-compressor", "fp8_e5m2", "per-attention-head"]},
    {"id": "kv-transfer", "q": "What KV transfer connectors does vLLM support for disaggregated prefilling?", "gold": ["NixlConnector", "MooncakeConnector", "LMCacheConnectorV1"]},
    {"id": "ngram", "q": "How do you configure the n-gram prompt lookup window in vLLM's speculative decoding?", "gold": ["prompt_lookup_max", "prompt_lookup_min", "ngram"]},
    {"id": "structured", "q": "What backends does vLLM support for generating structured outputs?", "gold": ["xgrammar", "llguidance", "StructuredOutputsParams"]},
    {"id": "reasoning", "q": "How do you limit the number of tokens a reasoning model spends producing its reasoning block?", "gold": ["thinking_token_budget", "reasoning_start_str", "reasoning_end_str"]},
    {"id": "prefix-hash", "q": "What is the default hashing algorithm for vLLM's prefix caching, and which alternatives provide reproducible cross-language hashing?", "gold": ["sha256", "sha256_cbor", "xxhash"]},
    {"id": "preemption", "q": "What is the default preemption mode in vLLM V1, and why was it chosen over the alternative?", "gold": ["RECOMPUTE", "PreemptionMode", "SWAP"]},
    {"id": "cudagraph", "q": "What class acts as the central controller for CUDA Graphs dispatch in vLLM v1, and what structure is used as the dispatch key?", "gold": ["CudagraphDispatcher", "BatchDescriptor", "CUDAGraphWrapper"]},
    {"id": "multimodal", "q": "What two mechanisms does vLLM use to recreate the Hugging Face processor's output for multimodal inputs without seeing the original text?", "gold": ["Dummy Input Text", "Prompt Update Detection", "PromptUpdate"]},
    {"id": "paged-attn", "q": "What CUDA kernel does vLLM's paged attention implement, and what warp-shuffle primitive reduces qk_max across warps?", "gold": ["paged_attention_kernel", "VLLM_SHFL_XOR_SYNC", "Qk_dot"]},
    {"id": "apc-block", "q": "What data class is the basic building block for prefix caching in vLLM v1, and which KV cache manager methods does the scheduler call to fetch computed blocks and allocate slots?", "gold": ["KVCacheBlock", "get_computed_blocks", "allocate_slots"]},
    {"id": "hybrid-kvcache", "q": "Which coordinator does vLLM's hybrid KV cache manager choose when a model has exactly two KV cache groups, and which one handles a single group?", "gold": ["HybridKVCacheCoordinator", "UnitaryKVCacheCoordinator", "KVCacheCoordinatorNoPrefixCache"]},
    {"id": "attention-backend", "q": "What two command-line flags specify the attention backend, and which optional backend targets NVIDIA SM120/SM121 GPUs?", "gold": ["--attention-backend", "--attention-config.backend", "b12x"]},
    {"id": "arch-overview", "q": "What unified constructor signature does vLLM require all model classes to implement, and which async engine class serves online requests?", "gold": ["VllmConfig", "AsyncLLMEngine", "keyword-only"]},
    {"id": "multiprocessing", "q": "Which environment variable controls vLLM's Python multiprocessing start method, and which one was used in v1 to enable multiprocessing in the engine core?", "gold": ["VLLM_WORKER_MULTIPROC_METHOD", "VLLM_ENABLE_V1_MULTIPROCESSING", "forkserver"]},
    {"id": "plugin", "q": "What entry point group name does vLLM use to register general plugins, and which function loads plugins into every process?", "gold": ["vllm.general_plugins", "vllm.platform_plugins", "load_plugins_by_group"]},
    {"id": "torch-compile", "q": "Which environment variable disables vLLM's torch.compile cache, and which config class controls dynamic shapes behavior?", "gold": ["VLLM_DISABLE_COMPILE_CACHE", "DynamicShapesConfig", "BACKED_SIZE_OBLIVIOUS"]},
    {"id": "opt-level", "q": "Which quantization fusions does vLLM's -O1 optimization level enable, and which kernel config flag does it turn on?", "gold": ["fuse_norm_quant", "fuse_act_quant", "enable_flashinfer_autotune"]},
    {"id": "hf-integration", "q": "When Hugging Face cannot find a model_type in transformers, which config.json field does it use to locate the config class, and what token environment variable does vLLM pass to download the model?", "gold": ["auto_map", "HF_TOKEN", "safetensors"]},
    {"id": "video-pruning", "q": "What are vLLM's training-free video token pruning algorithms, and which flag enables the feature?", "gold": ["--video-pruning-rate", "vidcom2", "Efficient Video Sampling"]},
    {"id": "tool-calling", "q": "Which flag is mandatory to enable automatic tool choice, and which environment variable disables structural-tag enforcement for tool calling?", "gold": ["--enable-auto-tool-choice", "VLLM_ENFORCE_STRICT_TOOL_CALLING", "llama3_json"]},
    {"id": "eagle", "q": "What speculative_config fields configure an EAGLE draft model, and which method string selects the EAGLE3 variant?", "gold": ["draft_tensor_parallel_size", "eagle3", "speculator-models"]},
    {"id": "numa", "q": "Which flags pin GPU workers to NUMA nodes, and what numactl syntax do they use for explicit CPU binding?", "gold": ["--numa-bind", "--numa-bind-nodes", "--numa-bind-cpus"]},
    {"id": "conserve-mem", "q": "What environment variable controls the KV cache size for the CPU backend, and which processor argument reduces processed image size for Qwen2-VL models?", "gold": ["VLLM_CPU_KVCACHE_SPACE", "mm_processor_kwargs", "max_dynamic_patch"]},
    {"id": "offline-api", "q": "What asynchronous queue APIs does vLLM's LLM class offer for non-blocking generation, and which method resets the multimodal cache?", "gold": ["LLM.enqueue", "LLM.wait_for_completion", "LLM.reset_mm_cache"]},
    {"id": "offline-rpc", "q": "Which LLM method executes a callable collectively across all workers, and which one resets the prefix cache?", "gold": ["collective_rpc", "apply_model", "reset_prefix_cache"]},
    {"id": "mla-backend", "q": "Which flag selects the MLA prefill backend, and which decode backend serves DeepSeek V4 sparse MLA by default on SM12x?", "gold": ["-ac.mla_prefill_backend", "FLASHMLA_SPARSE_DSV4", "FLASHINFER_MLA_SPARSE_DSV4"]},
    {"id": "split-audio", "q": "Which utility splits long audio into chunks at quiet points for Whisper transcription, and what argument caps each chunk's length?", "gold": ["split_audio", "max_clip_duration_s", "min_energy_window_size"]},
    {"id": "pythonic-tool", "q": "Which tool parser handles models that emit Pythonic list-based tool calls instead of JSON, and what manager registers custom parsers?", "gold": ["pythonic", "extract_tool_calls", "ToolParserManager"]},
    {"id": "chunked-prefill", "q": "In vLLM V1, which scheduling budget governs how many tokens a prefill chunk may contain, and which latency metric improves because decode is prioritized?", "gold": ["max_num_batched_tokens", "chunked prefill", "TTFT"]},
    {"id": "fastokens", "q": "Which environment variable switches vLLM's BPE tokenizer to the Rust backend, and what exception does vLLM raise at tokenizer load if the package is missing?", "gold": ["VLLM_USE_FASTOKENS", "fastokens", "ImportError"]},
    {"id": "mm-caching", "q": "How do you size vLLM's multi-modal processor cache, and which cache layout keeps data in shared memory across worker processes?", "gold": ["mm_processor_cache_gb", "mm_processor_cache_type", "key-replicated"]},
    {"id": "cpu-affinity", "q": "Which environment variables control OpenMP thread placement, the reserved CPU count, and the visible memory nodes for vLLM's CPU backend?", "gold": ["VLLM_CPU_OMP_THREADS_BIND", "VLLM_CPU_NUM_OF_RESERVED_CPU", "CPU_VISIBLE_MEMORY_NODES"]},
    {"id": "cudagraph-dual", "q": "Which dual-mode CUDA graph configurations switch dynamically between full and piecewise graphs at runtime, and which enum defines these modes?", "gold": ["FULL_AND_PIECEWISE", "FULL_DECODE_ONLY", "CudagraphModes"]},
    {"id": "flash-attn-version", "q": "Which config option selects the FlashAttention version, and what are the defaults on Blackwell versus Hopper GPUs?", "gold": ["flash_attn_version", "FA4", "FA3"]},
    {"id": "suffix-decoding", "q": "Which speculative-decoding keys bound the suffix-tree depth, the number of cached requests, and the speculative length multiple?", "gold": ["suffix_decoding_max_tree_depth", "suffix_decoding_max_cached_requests", "suffix_decoding_max_spec_factor"]},
    {"id": "rejection-sampling", "q": "Which speculative-decoding key selects how draft tokens are verified, and which two synthetic-mode settings supply acceptance rates or a target mean length?", "gold": ["rejection_sample_method", "synthetic_acceptance_rates", "synthetic_acceptance_length"]},
    {"id": "pooling", "q": "Which pooling APIs does the LLM class offer for embedding, classification, and scoring models?", "gold": ["LLM.embed", "LLM.classify", "LLM.score"]},
    {"id": "weight-transfer", "q": "Which LLM methods initialize and drive the weight-transfer cycle used for RL training?", "gold": ["LLM.init_weight_transfer_engine", "LLM.start_weight_update", "LLM.update_weights"]},
    {"id": "profiling", "q": "Which LLM methods start and stop a profiling session, and what optional argument names the trace?", "gold": ["LLM.start_profile", "LLM.stop_profile", "trace prefix"]},
    {"id": "limit-mm", "q": "Which LLM argument caps the number of multi-modal items per prompt, and which config classes describe the per-modality size hints for images and videos?", "gold": ["limit_mm_per_prompt", "ImageDummyOptions", "VideoDummyOptions"]},
    {"id": "mm-device-norm", "q": "Which config flag offloads image normalization to the GPU, and which module folds the rescale factor into its weight and bias?", "gold": ["mm_device_do_normalize", "FusedInputNorm", "Qwen2_5_VLForConditionalGeneration"]},
    {"id": "disagg-connectors", "q": "Which KV-transfer connectors support CPU offloading, a distributed KV store, and ROCm-only disaggregation?", "gold": ["OffloadingConnector", "FlexKVConnectorV1", "MoRIIOConnector"]},
    {"id": "structured-backends", "q": "Besides xgrammar and guidance, which structured-output backends use Rust-style regex, and which one uses Python's re module?", "gold": ["outlines", "lm-format-enforcer", "structured-outputs-config"]},
    {"id": "arch-processes", "q": "Which process coordinates data-parallel ranks, and over what transport and topology do API servers reach engine cores?", "gold": ["DP Coordinator Process", "ZMQ", "many-to-many"]},
    {"id": "tool-parser-plugin", "q": "Which flag registers a custom tool-parser plugin, and which class do you subclass to implement one?", "gold": ["--tool-parser-plugin", "--tool-call-parser", "Hermes2ProToolParser"]},
    {"id": "reasoning-config", "q": "Which flag selects the reasoning parser, and which two CLI arguments define the boundary tokens and server-wide chat-template defaults?", "gold": ["--reasoning-parser", "--reasoning-config", "--default-chat-template-kwargs"]},
    {"id": "mamba-prefix-cache", "q": "Which flags configure Mamba's fine-grained prefix caching, and which one sets the granularity of prefix-cache keys?", "gold": ["--mamba-cache-mode", "--enable-mamba-fine-grained-prefix-cache", "--prefix-match-unit"]},
    {"id": "kv-config", "q": "In the --kv-transfer-config JSON, which three keys name the connector, its role, and the buffer device?", "gold": ["kv_connector", "kv_role", "kv_buffer_device"]},
]

# 语义改写题：和精确题「同答案、不同问法」，query 刻意避开 gold 精确词（gold 词不作为子串出现）。
# 考「靠意思找到答案 chunk」，专治 BM25 的字面匹配主场——这正是 router「语义题走向量」要赢的那一半。
# 镜像精确题 id：lora/reasoning/cudagraph/prefix-hash + fp8/preemption/multimodal/paged-attn/
#   apc-block/multiprocessing/torch-compile/hf-integration/tool-calling/offline-api/mla-backend。
# 出题约束：query 不能含任一 gold 词子串，否则退化成字面题、测不出语义检索（check_gold 会标 ⚠）。
QA_SEMANTIC = [
    {"q": "Can vLLM swap in or remove fine-tuned LoRA adapters on a live server without shutting it down, and how does it wire up a custom adapter directory?", "gold": ["VLLM_ALLOW_RUNTIME_LORA_UPDATING", "load_lora_adapter", "LoRAResolver"]},
    {"q": "How do you cap how long a reasoning model spends thinking before it must start producing its final answer, and what string markers enclose that thinking?", "gold": ["thinking_token_budget", "reasoning_start_str", "reasoning_end_str"]},
    {"q": "Which component orchestrates the selection and replay of vLLM's static execution graphs, and what key data structure tells it which graph to run?", "gold": ["CudagraphDispatcher", "BatchDescriptor", "CUDAGraphWrapper"]},
    {"q": "What fingerprint does vLLM compute to recognize a reused prompt prefix, and which alternative deterministic checksums can you switch to for cross-language reproducibility?", "gold": ["sha256", "sha256_cbor", "xxhash"]},
    {"q": "What are the ways vLLM can quantize the KV cache to 8-bit floats, and how are the scaling factors calibrated for each layer or tensor?", "gold": ["llm-compressor", "fp8_e5m2", "per-attention-head"]},
    {"q": "When vLLM must evict a running request to free up KV cache memory, does it re-run the prefill or try to offload the cached state to a slower store?", "gold": ["RECOMPUTE", "PreemptionMode", "SWAP"]},
    {"q": "Since vLLM re-runs the Hugging Face processor without the original text, how does it reconstruct the exact output the processor would have produced?", "gold": ["Dummy Input Text", "Prompt Update Detection", "PromptUpdate"]},
    {"q": "In the GPU kernel for paged attention, how do the threads within a warp share the running maximum of the query-key dot products during reduction?", "gold": ["paged_attention_kernel", "VLLM_SHFL_XOR_SYNC", "Qk_dot"]},
    {"q": "What is the smallest unit of cached key-value memory that the scheduler tracks for prefix reuse, and which manager methods retrieve the reusable ones and reserve fresh ones?", "gold": ["KVCacheBlock", "get_computed_blocks", "allocate_slots"]},
    {"q": "Which environment setting controls whether vLLM starts worker processes by forking or spawning, and which older variable did the v1 engine use for the same purpose?", "gold": ["VLLM_WORKER_MULTIPROC_METHOD", "VLLM_ENABLE_V1_MULTIPROCESSING", "forkserver"]},
    {"q": "Which setting stops vLLM from reusing previously compiled computation graphs, and which configuration object controls how variable input shapes are handled?", "gold": ["VLLM_DISABLE_COMPILE_CACHE", "DynamicShapesConfig", "BACKED_SIZE_OBLIVIOUS"]},
    {"q": "When transformers cannot match a model to a known architecture, which config field tells it where the custom classes live, and which credential does vLLM forward to download the weights?", "gold": ["auto_map", "HF_TOKEN", "safetensors"]},
    {"q": "Which flag lets the model choose a tool on its own without the client pre-selecting it, and which toggle relaxes the enforcement of special framing tags around tool calls?", "gold": ["--enable-auto-tool-choice", "VLLM_ENFORCE_STRICT_TOOL_CALLING", "llama3_json"]},
    {"q": "With the offline engine's LLM class, how do you submit prompts without blocking the caller and later collect the finished outputs, and how do you clear the cached multimodal inputs?", "gold": ["LLM.enqueue", "LLM.wait_for_completion", "LLM.reset_mm_cache"]},
    {"q": "For DeepSeek's sparse multi-head latent attention, how do you pick the prefill implementation and which decode implementation is used by default on SM12x GPUs?", "gold": ["-ac.mla_prefill_backend", "FLASHMLA_SPARSE_DSV4", "FLASHINFER_MLA_SPARSE_DSV4"]},
]

# judge 忠实度评测（里程碑 4）的题目子集：4 道常规题 + 1 道「真·陷阱题」。
# 陷阱题要「听起来合理、但语料里确实没有」，看 agent 是「拒绝编造」还是「凭空答」。
# 之前的 MTP 陷阱已失效：README.md 已文档化 MTP 配置、step3p5.py 有源码实现，agent 能查到
# 真材料（忠实度 0.94 却是「答对了」而非「没编造」）——换成编造的名称陷阱。
_by_id = {qa["id"]: qa["q"] for qa in QA_SET}
JUDGE_QUESTIONS = [
    _by_id["lora"],
    _by_id["fp8"],
    _by_id["preemption"],
    # 陷阱：adaptive per-layer KV 量化 + --kv-cache-quantization-level 都是编造的，
    # 真 vLLM 用 kv_cache_dtype 做统一 fp8 量化。诚实答法=「无此 flag/机制」，编造答法=编个默认值。
    "How does vLLM's adaptive KV cache quantization choose per-layer bit-widths, "
    "and what is the default value of the --kv-cache-quantization-level flag?",
    _by_id["structured"],
]

# 答案正确性评测（answer_eval.py）的题目子集：每题除 gold 外，再手写「必须事实」（required facts）。
# gold 只测「检索到没」，测不出「答对没」——agent 可能检索到正确 chunk、忠实复述它，却漏说关键事实
# 或答非所问。这里把标准答案拆成 2~3 条「必须正确陈述的事实」，answer_eval.py 拿 agent 答案逐条核对。
# 关键纪律：facts 必须是「从文档读出的机制/因果/约束，用自己的话复述」，绝不能是「gold 关键词换句话」
# ——否则正确性指标就和 keyword Recall 重复了（检索到关键词 + 忠实复述 ≈ 必然命中，测不出新东西）。
# flag/类名题答案本身就含标识符，就把标识符和它的作用/条件/默认值绑在一起，让 judge 测「讲对没」而非「出现没」。
# facts 是 ground truth，改前先 grep 对应文档核对（数值/默认值/机制对齐文档）。
ANSWER_QUESTIONS = [
    {"q": _by_id["lora"], "facts": [
        "Dynamic adapter updating is opt-in: it only works after an environment variable is explicitly set, and the docs warn it carries security risk and should not be used in production outside a trusted environment.",
        "Adapters are loaded and unloaded through HTTP API endpoints while the server keeps running, with no restart required.",
        "A resolver plugin can fetch adapters on demand from local or remote sources (local directory, S3, or Hugging Face Hub); with multiple resolvers registered, vLLM uses the first adapter it finds.",
    ]},
    {"q": _by_id["fp8"], "facts": [
        "FP8 KV-cache quantization comes in two granularities: per-tensor (one scale per tensor) and per-attention-head (one scale per head).",
        "The recommended calibration is dataset-based through llm-compressor, which estimates scales from a curated calibration dataset for higher accuracy.",
        "Per-attention-head quantization only works with the Flash Attention backend and relies on llm-compressor's calibration path.",
    ]},
    {"q": _by_id["kv-transfer"], "facts": [
        "KV transfer between prefill and decode instances is pluggable through connectors selected in the kv_connector config.",
        "Supported connectors include NixlConnector (which moves KV over NIXL backends such as UCX or GDS), MooncakeConnector, and LMCacheConnectorV1.",
        "Multiple connectors can be composed into an ordered list via MultiConnector.",
    ]},
    {"q": _by_id["ngram"], "facts": [
        "The n-gram prompt lookup window is bounded by a minimum and a maximum n-gram size, each set through its own config key.",
        "If only one bound is set the other mirrors it, and if neither is set both default to 5.",
    ]},
    {"q": _by_id["structured"], "facts": [
        "vLLM supports several pluggable structured-output backends, with xgrammar and guidance as the two primary ones.",
        "Additional backends include outlines and lm-format-enforcer, which accept a different regex dialect (Rust-style vs Python's re).",
    ]},
    {"q": _by_id["reasoning"], "facts": [
        "A reasoning model's thinking budget caps how many tokens it may spend inside its reasoning block, set as a per-request sampling parameter.",
        "The block is delimited by start and end marker strings; once the token count reaches the budget, vLLM forces the model to emit the end marker, terminating the block.",
    ]},
    {"q": _by_id["prefix-hash"], "facts": [
        "Prefix-cache hashing defaults to sha256.",
        "For reproducible cross-language hashing, variants like sha256_cbor and xxhash_cbor serialize keys with canonical CBOR instead of Python pickle, so hashes don't drift across Python/vLLM versions.",
    ]},
    {"q": _by_id["preemption"], "facts": [
        "V1's default preemption mode is RECOMPUTE (recompute the preempted request) rather than SWAP (offload its KV cache).",
        "RECOMPUTE is chosen because recomputation has lower overhead than swapping in the V1 architecture.",
    ]},
    {"q": _by_id["cudagraph"], "facts": [
        "CUDA Graphs dispatch in v1 is centralized in a dispatcher that holds the single source of truth about available graphs.",
        "The dispatch key is a batch descriptor carrying the batch shape: token count, request count, and whether all requests share the same query length (uniform).",
    ]},
    {"q": _by_id["multimodal"], "facts": [
        "To reproduce the HF processor's output without seeing the original text, vLLM feeds a synthetic placeholder (dummy) text into the processor.",
        "It then detects how the prompt's placeholder tokens should be updated to match the processor's output.",
    ]},
    {"q": _by_id["paged-attn"], "facts": [
        "Paged attention is implemented by a dedicated CUDA kernel for the paged KV layout.",
        "It uses an XOR-shuffle primitive to reduce the running max of query-key scores across warps.",
    ]},
    {"q": _by_id["torch-compile"], "facts": [
        "The torch.compile cache can be disabled via an environment variable, intended for debugging compilation or cache issues.",
        "Dynamic-shapes behavior is controlled by a config class whose type is BACKED (default), UNBACKED, or BACKED_SIZE_OBLIVIOUS.",
    ]},
    {"q": _by_id["multiprocessing"], "facts": [
        "The Python multiprocessing start method used by vLLM workers is selected via an environment variable, defaulting to fork.",
        "If CUDA is detected as already initialized, vLLM forces the spawn method and emits a warning, because fork would break.",
    ]},
    {"q": _by_id["mla-backend"], "facts": [
        "The MLA prefill backend is selected by a dedicated attention-config flag; if unset it is chosen automatically at runtime based on hardware and config.",
        "DeepSeek V4 sparse MLA uses dedicated decode backends, defaulting on NVIDIA to FLASHINFER_MLA_SPARSE_DSV4 on SM12x and FLASHMLA_SPARSE_DSV4 on other CUDA architectures.",
    ]},
    {"q": _by_id["chunked-prefill"], "facts": [
        "max_num_batched_tokens is the scheduling budget that caps how many tokens a single prefill chunk may contain; a prefill that doesn't fit is automatically split into chunks.",
        "Chunked prefill prioritizes decode requests, which improves inter-token latency (ITL).",
    ]},
    # 两道「多跳」难题（新题，不在 QA_SET，无 gold）：答案必须跨两个文档拼起来，一次检索答不全。
    # 用来检验 agent 的多步能力，也让正确性这层在「单跳 1.00 基线」之外真正起分辨作用。
    {"q": "vLLM V1's scheduler prioritizes decode requests and then uses the remaining token budget for prefill. "
          "What kind of batch does this scheduling create, and how does that determine which CUDA graph mode vLLM uses to execute it?", "facts": [
        "Chunked prefill batches decode requests together with prefill chunks into one batch, producing a mixed prefill-decode (non-uniform) batch.",
        "CUDA graphs treat pure-decode (uniform) batches separately from prefill/mixed (non-uniform) batches and capture them separately.",
        "Mixed (non-uniform) batches are served by PIECEWISE CUDA graphs: the default FULL_AND_PIECEWISE mode captures a full graph for uniform decode and piecewise graphs for everything else.",
    ]},
    {"q": "How does vLLM's prefix caching keep entries distinct when different requests share the same text but "
          "use different images, and what does the multimodal processing pipeline have to do with making that caching possible?", "facts": [
        "The multimodal processor recreates the HF processor's output via Dummy Input Text and Prompt Update Detection, without seeing the original text.",
        "This establishes the correspondence between placeholder feature tokens (like <image>) and the multimodal inputs, which is what enables optimizations like chunked prefill and prefix caching.",
        "Prefix-cache keys add extra hashes beyond the token prefix — among them multi-modality input hashes (alongside LoRA IDs and cache salts) — so different multimodal inputs map to distinct cache blocks.",
    ]},
    # 实验 18 补的「难题」：打破 33/33 零信息量，让正确性这层真正有分辨力。
    #   - 陷阱题：答案是「不存在」，戳 agent 的幻觉倾向（它常编一个 env 变量名出来）。
    #   - 源码题：答案只写在源码里（rejection_sampler.py 的常量），文档搜不到，逼它走 search_code
    #     （实验 13 已证明源码检索是弱项），答不对才说明这层真的在测东西。
    {"q": "Which single environment variable turns on speculative decoding in vLLM for any model, and what does it default to?", "facts": [
        "There is no environment variable that turns on speculative decoding; it is enabled through the speculative_config argument passed to LLM / AsyncLLM.",
        "speculative_config takes a SpeculativeConfig (or a dict) naming the method and its parameters; there is no env-var default.",
    ]},
    {"q": "In vLLM's GPU rejection sampler, a fixed buffer caps how many logits rows a single verification chunk may hold. What is that buffer's size, and how is the per-chunk row cap computed from the vocabulary size?", "facts": [
        "The fixed target-logits buffer is MAX_CHUNK_BYTES = 2**30 bytes (1 GiB).",
        "The per-chunk row cap is get_max_chunk_logits(vocab_size) = max(1, 2**30 // (vocab_size * 4)), because each FP32 logits row is 4 bytes; a larger vocabulary means fewer rows per chunk.",
    ]},
]

# 代码检索评测题（eval_code.py）：gold = 实现该答案的「符号名」（类/函数/方法），不再是文档关键词。
# 语料 = 代码索引（vLLM v1 投机解码 48 个 .py / 523 个符号 chunk，见 agent.CODE_DIRS）。
# 出题纪律（check_code_gold 的 harm 模型，同 check_gold）：
#   - 可解：gold 符号必须在代码索引里（出现在 ≥1 chunk），否则题目无解。
#   - 稀疏：gold 符号只能出现在极少数 chunk（类名会出现在「类概览 + 每个方法」chunk，是正常放大，
#     仍远小于 523）；像 propose/load_model 这种几十个 chunk 都有的通用名禁用——测不出检索能力。
#   - 不泄漏：query 不能含 gold 符号名字面，否则 BM25 字面命中，退化成「查名字」而非「查实现」。
#   - 不同源（实验 18 收紧）：query 不能含「答案 chunk 里出现的同源词」——不只是符号名字面，还包括
#     docstring/源码里的实现措辞（如 "prompt-lookup"、"hidden states"、"log-sum-exp"、"autoregressive"）。
#     旧版 18 题几乎每道都用 docstring 的近义改写出题，R@1 其实在测「题面和合成描述字面重合」，不是
#     「检索能理解代码」。去同源词后 R@1 掉到 0，只有 R@10（能不能捞回来）剩 0.44 是真本事。
#     所以下面的 q 一律用「高层角色描述」措辞，避开 gold 符号名与源码实现词。
CODE_QA_SET = [
    {"id": "rejection-sampler", "q": "After the drafting model has already proposed a batch of candidate tokens, which class compares each proposed token's probability under the drafting model against its probability under the main model, and keeps or drops the token accordingly?", "gold": ["RejectionSampler"]},
    {"id": "vocab-mapping", "q": "When a small drafting model and the main model number their tokens differently, which helper translates between the two numbering schemes?", "gold": ["VocabMapping"]},
    {"id": "ngram-lookup", "q": "For the drafting approach that searches the prompt prefix within a bounded-size window, how does it find the longest match inside that window and continue from it?", "gold": ["_find_longest_matched_ngram_and_propose_tokens"]},
    {"id": "ngram-gpu-update", "q": "In the GPU version of that backward-scanning drafting approach, how are its bookkeeping tables refreshed between successive decoding steps as suggestions get approved?", "gold": ["update_ngram_gpu_tensors_incremental"]},
    {"id": "sd-metadata", "q": "Which data container holds all the per-request bookkeeping for a speculation run, such as the candidate token list and how many were kept?", "gold": ["SpecDecodeMetadata"]},
    {"id": "hidden-states-extract", "q": "For the scheme where a second, smaller model derives its candidates from the main model's internal activations, which component hands those activations over?", "gold": ["ExtractHiddenStatesProposer"]},
    {"id": "logsumexp", "q": "How does the acceptance stage fold the probabilities over the whole vocabulary into summary quantities used to make each keep-or-drop call?", "gold": ["_compute_global_logsumexp"]},
    {"id": "adaptive-verify", "q": "How does vLLM decide how many candidate tokens to inspect for each request by reasoning about the observed cost of inspecting different counts?", "gold": ["AdaptiveVerificationManager"]},
    {"id": "medusa", "q": "Which candidate-generation scheme stacks several small prediction layers on top of the main model's internal representation?", "gold": ["MedusaProposer"]},
    {"id": "suffix-decode", "q": "How does the drafting strategy that keeps a running cache of the sequences it has already generated match the current sequence's ending against that cache and continue?", "gold": ["SuffixDecodingProposer"]},
    {"id": "autoregressive", "q": "Which candidate generator runs a standalone draft model one token at a time in a loop?", "gold": ["AutoRegressiveSpeculator"]},
    {"id": "mtp", "q": "How does the several-tokens-at-once path reuse the main model's first pass to seed the second model's generation?", "gold": ["MTPSpeculator"]},
    {"id": "multi-mtp", "q": "How does the several-tokens-at-once variant spanning multiple modules set up each module's internal workspace before running them together?", "gold": ["prepare_input_hidden_states_and_embeddings"]},
    {"id": "dflash", "q": "How does the drafting path that merges the main model's work into the same pass run the second model?", "gold": ["DFlashSpeculator"]},
    {"id": "speculator-factory", "q": "Given a complete runtime configuration, which entry point selects and constructs the right candidate-generation object for the requested speculation mode?", "gold": ["init_speculator"]},
    {"id": "eagle-spec", "q": "Which candidate generator augments a draft model's input with the main model's hidden states to improve its proposals?", "gold": ["EagleSpeculator"]},
    {"id": "gemma4", "q": "How does the proposal path for one specific model family let the second model share the main model's key-value cache and set up pre-captured graphs for its lightweight prediction heads?", "gold": ["Gemma4Proposer"]},
    {"id": "dspark", "q": "How does one candidate-generation path draw its suggestions one at a time for each request from the main model's internal state?", "gold": ["DSparkSpeculator"]},
]

# 档① 自然提问（配对差值的「自然」一臂）：与 CODE_QA_SET（档②抽象）按 id 对齐、同 gold。
# 两臂同 id 逐题配对，delta = 题面口径净效应（术语+措辞风格）。17 题：A 组 8 道从 orig 复原
# （措辞归一：target model 统一、main model 消除），B 组 9 道新写。
# 冻结记录（2026-09-22 review 后）：
#   - dflash 原题「merges the draft model's computation into the target model's forward pass」已删：
#     DFlash 是 DraftModelSpeculator，跑自己的 draft model 做并行草案（1+N layout、一次 forward
#     pass 出全部候选），并不并入 target 的 forward pass。已改为「separate draft model + single
#     forward pass」。注：档② dflash 题面「merges the main model's work into the same pass」同是
#     这个错误概括，但档②已冻结不改，只在报表口径声明里标注。
#   - dspark 暂摘：原题「loads a draft model and samples each request's candidate tokens one at a
#     time」描述的是 AutoRegressiveSpeculator（逐 token 推进），不是 DSparkSpeculator（DFlash 子类，
#     semi-autoregressive 并行块草案）。待 ②a 定案后单独重写；现在它既答不对又占配对名额、只添噪声。
#   - rejection-sampler 的 RejectionSampler 在 vLLM 源码有两处定义（v1/sample/ 与 v1/worker/gpu/
#     spec_decode/），但 v1/sample/ 不在 CODE_DIRS，索引里只有后者 → 命中判定无歧义。这是源码层
#     命名过载，报表口径声明里标注，不重写题面。
# 方法学提醒（报表三行出）：两组题面都看着 gold 写出来，delta 只能读成「词根/符号线索值多少分」，
# 不是「真实用户 vs 抽象」。ngram-lookup/logsumexp/ngram-gpu-update 三道近乎「标识符去下划线」，
# 与其余题不同分布 → L1/L2 按 A 组(8)/B 组(9)/合计(17) 三行报，不合成一个数。
NATURAL_QA = {
    "medusa": "Which proposer implements MEDUSA-style drafting by stacking several classification heads on top of the target model's hidden states?",
    "multi-mtp": "For the multi-module multi-token-prediction path, how does vLLM prepare each draft module's input hidden states and embeddings before running them together?",
    "eagle-spec": "Which speculator subclass specializes the autoregressive drafting path for EAGLE-style draft models?",
    "hidden-states-extract": "Which proposer extracts hidden states from the target model so they can be fed to a draft model for EAGLE-style speculative decoding?",
    "rejection-sampler": "Which class verifies draft tokens against the target model's token probabilities and accepts or rejects each one?",
    "suffix-decode": "How does the suffix-matching proposal method look up the current token suffix against previously seen sequences to propose a continuation?",
    "adaptive-verify": "How does vLLM adaptively decide how many draft tokens to verify per request, based on measured per-step cost curves?",
    "mtp": "How does the multi-token-prediction path reuse the target model's prefill step to seed the draft model's decode?",
    "vocab-mapping": "When the draft model and the target model use different tokenizers, which component translates token IDs from one vocabulary to the other?",
    "ngram-lookup": "In the n-gram drafting method, how does vLLM find the longest run of tokens in the context that matches the current suffix, and turn the tokens that follow it into draft candidates?",
    "ngram-gpu-update": "When running n-gram drafting on GPU, how does vLLM keep its on-device token-id and token-count tables updated incrementally as accepted tokens extend the sequence?",
    "sd-metadata": "Which data structure holds the per-request bookkeeping for one speculative decoding step, such as the list of candidate tokens and how many were accepted?",
    "logsumexp": "The rejection-sampling step needs a log-sum-exp of the target model's probabilities over the whole vocabulary; how does vLLM combine the per-block results into that global value?",
    "autoregressive": "Which drafting component runs a standalone draft model token-by-token in a loop, handling its loading, attention setup, and CUDA graph capture to propose candidates?",
    "dflash": "Which drafting path proposes all of its speculative tokens in a single forward pass of a separate draft model, in the DFlash parallel-drafting technique?",
    "speculator-factory": "Given the speculative decoding settings in the config, which entry point selects and constructs the right proposer for the requested method?",
    "gemma4": "For the Gemma family, how does the speculative decoding path set up the multi-group KV cache and the centroid masking its lightweight prediction heads rely on?",
}

# 代码生成评测题（eval_codegen.py）：task 是「写代码」的指令，facts 是静态「代码事实」。
# 和答案正确性（answer_eval）的分工：那里 facts 判「说没说对」，这里 facts 判「代码里有没有用对 API」。
# facts 是 ground truth，写前 grep 文档核对 API 名/参数值（同 answer_eval 纪律）——API 记错 = 参考答案错。
#   checks = 对 AST 做结构断言（见 code_facts.py）：import/call/kwarg/dict_entry/field/func/class/assign
#            /name/compare，支持 "not": true 取反（抓废弃 API 如 guided_json、抓 bug 形如 1<<n.bit_length()）。
#   runnable  = True 的纯 Python 题额外跑 run_check（真执行断言），是比静态结构更强的信号。
#   seed      = 可选 {相对路径: 内容}，跑题前预写进 workspace（edit_file 改 bug 题用它先铺一个有 bug 的文件）。
# 分两档：前 8 题「跟文档走」的 API 用法（照 grep 的文档 API 写）；后 4 题「读源码才能写对」（facts/run_check
# 钉的是只写在源码里的细节：dataclass 字段名、1 GiB 常量、log-ratio 公式、off-by-one 修 bug）。
# task 用「Write a program/function」这类自然措辞即可：route() 已拆出 write_code 一路（实验 14 修掉
# 「Write a Python script 误路由到 run_python」的 bug），写代码题会正确走 write_file/edit_file。
CODE_GEN_QUESTIONS = [
    {"id": "fp8-apc", "path": "fp8_apc.py", "runnable": False,
     "task": ("Write a program that performs offline batch inference with vLLM, enabling FP8 KV-cache "
              "quantization (kv_cache_dtype=\"fp8\") and automatic prefix caching."),
     "facts": [
         {"desc": "imports LLM and SamplingParams from vllm", "checks": [{"kind": "import", "module": "vllm", "names": ["LLM", "SamplingParams"]}]},
         {"desc": "instantiates the engine", "checks": [{"kind": "call", "name": "LLM"}]},
         {"desc": "enables FP8 KV cache via kv_cache_dtype=\"fp8\"", "checks": [{"kind": "kwarg", "name": "kv_cache_dtype", "value": "fp8"}]},
         {"desc": "enables automatic prefix caching via enable_prefix_caching=True", "checks": [{"kind": "kwarg", "name": "enable_prefix_caching", "value": True}]},
     ]},
    {"id": "chunked-prefill", "path": "chunked_prefill.py", "runnable": False,
     "task": ("Write a program that instantiates a vLLM LLM and sets max_num_batched_tokens=16384 to "
              "control the chunked-prefill scheduling budget."),
     "facts": [
         {"desc": "imports LLM from vllm", "checks": [{"kind": "import", "module": "vllm", "names": ["LLM"]}]},
         {"desc": "instantiates the engine", "checks": [{"kind": "call", "name": "LLM"}]},
         {"desc": "sets the batch token budget to 16384", "checks": [{"kind": "kwarg", "name": "max_num_batched_tokens", "value": 16384}]},
     ]},
    {"id": "lora", "path": "lora_offline.py", "runnable": False,
     "task": ("Write a program that loads a base model with LoRA support and issues a generation request "
              "tagged with a LoRA adapter via LoRARequest."),
     "facts": [
         {"desc": "imports LoRARequest from vllm.lora.request", "checks": [{"kind": "import", "module": "vllm.lora.request", "names": ["LoRARequest"]}]},
         {"desc": "enables LoRA via enable_lora=True", "checks": [{"kind": "kwarg", "name": "enable_lora", "value": True}]},
         {"desc": "constructs a LoRARequest and passes it via lora_request=", "checks": [
             {"kind": "call", "name": "LoRARequest"},
             {"kind": "kwarg", "name": "lora_request"},
         ]},
     ]},
    {"id": "structured", "path": "structured_output.py", "runnable": False,
     "task": ("Write a program that uses vLLM's structured outputs API to constrain generation to two "
              "sentiment choices (positive / negative)."),
     "facts": [
         {"desc": "imports StructuredOutputsParams and SamplingParams", "checks": [{"kind": "imports", "names": ["StructuredOutputsParams", "SamplingParams"]}]},
         {"desc": "passes structured_outputs= to SamplingParams", "checks": [{"kind": "kwarg", "name": "structured_outputs"}]},
         {"desc": "constrains to choices via choice=", "checks": [{"kind": "kwarg", "name": "choice"}]},
         {"desc": "does NOT use the deprecated guided_json / guided_decoding_backend API", "checks": [
             {"kind": "name", "name": "guided_json", "not": True},
             {"kind": "name", "name": "guided_decoding_backend", "not": True},
         ]},
     ]},
    {"id": "ngram-sd", "path": "ngram_sd.py", "runnable": False,
     "task": ("Write a program that enables n-gram speculative decoding with a prompt lookup window bounded "
              "by prompt_lookup_min=2 and prompt_lookup_max=5."),
     "facts": [
         {"desc": "passes a speculative_config to the engine", "checks": [{"kind": "kwarg", "name": "speculative_config"}]},
         {"desc": "selects the ngram method", "checks": [{"kind": "dict_entry", "key": "method", "value": "ngram"}]},
         {"desc": "sets the lookup window bounds 2 and 5", "checks": [
             {"kind": "pair", "key": "prompt_lookup_min", "value": 2},
             {"kind": "pair", "key": "prompt_lookup_max", "value": 5},
         ]},
         {"desc": "sets num_speculative_tokens", "checks": [{"kind": "pair", "key": "num_speculative_tokens"}]},
     ]},
    # 纯 Python 可跑题：静态 facts 只钉签名，真伪靠 run_check 真执行。
    {"id": "rrf", "path": "rrf.py", "runnable": True,
     "task": ("Write a function rrf_fusion(rankings, k=60) where rankings is a list of lists (each an ordered "
              "list of item ids, best first), returning a dict of item id → Reciprocal Rank Fusion score "
              "(sum over lists of 1/(k + rank), rank 1-based)."),
     "facts": [{"desc": "defines rrf_fusion with a k parameter", "checks": [{"kind": "func", "name": "rrf_fusion", "args": ["k"]}]}],
     "run_check": ("d = rrf_fusion([[1, 2], [2, 1]], k=60)\n"
                   "assert abs(d[1] - (1/61 + 1/62)) < 1e-9, d\n"
                   "assert abs(d[2] - (1/62 + 1/61)) < 1e-9, d\n")},
    {"id": "kv-cache-size", "path": "kv_cache.py", "runnable": True,
     "task": ("Write a function kv_cache_bytes(num_tokens, num_layers, num_kv_heads, head_dim, "
              "bytes_per_elem=2) that returns the KV cache size in bytes = num_tokens * num_layers * 2 "
              "(K and V) * num_kv_heads * head_dim * bytes_per_elem."),
     "facts": [{"desc": "defines kv_cache_bytes with the required parameters",
                "checks": [{"kind": "func", "name": "kv_cache_bytes", "args": ["num_tokens", "num_layers", "num_kv_heads", "head_dim"]}]}],
     "run_check": "assert kv_cache_bytes(100, 32, 8, 128, 2) == 100*32*2*8*128*2\n"},
    {"id": "next-pow2", "path": "next_pow2.py", "runnable": True,
     "task": ("Write a function next_power_of_2(n) that returns the smallest power of two greater than or "
              "equal to n, for positive integer n."),
     "facts": [{"desc": "defines next_power_of_2", "checks": [{"kind": "func", "name": "next_power_of_2"}]}],
     "run_check": ("assert next_power_of_2(1) == 1\n"
                   "assert next_power_of_2(3) == 4\n"
                   "assert next_power_of_2(5) == 8\n"
                   "assert next_power_of_2(8) == 8\n")},

    # ---- 硬题：读源码才能写对（facts/run_check 钉的是只写在源码里的细节）----
    {"id": "spec-metadata", "path": "spec_metadata.py", "runnable": False,
     "task": ("Write the SpecDecodeMetadata dataclass to workspace/spec_metadata.py, matching vLLM's "
              "vllm/v1/spec_decode/metadata.py exactly: same field names, field types, and __post_init__ "
              "behavior."),
     "facts": [
         {"desc": "declares a dataclass named SpecDecodeMetadata", "checks": [{"kind": "class", "name": "SpecDecodeMetadata", "decorator": "dataclass"}]},
         {"desc": "has the draft-token fields", "checks": [{"kind": "field", "class": "SpecDecodeMetadata", "names": ["draft_token_ids", "num_draft_tokens"]}]},
         {"desc": "has the cumulative-count fields", "checks": [{"kind": "field", "class": "SpecDecodeMetadata", "names": ["cu_num_draft_tokens", "cu_num_sampled_tokens"]}]},
         {"desc": "has the logits-index fields", "checks": [{"kind": "field", "class": "SpecDecodeMetadata", "names": ["target_logits_indices", "bonus_logits_indices", "logits_indices"]}]},
         {"desc": "computes max_spec_len in __post_init__", "checks": [{"kind": "assign", "target": "max_spec_len", "method": "__post_init__"}]},
     ]},
    {"id": "accept-decision", "path": "accept.py", "runnable": True,
     "task": ("Write a function accept_draft_token(target_log_prob, draft_log_prob, u, target_argmax, "
              "draft_token, is_greedy) to workspace/accept.py that replicates vLLM's rejection-sampling "
              "accept decision from vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py: greedy mode "
              "and the non-greedy probability-ratio test."),
     "facts": [
         {"desc": "defines accept_draft_token with a greedy flag", "checks": [{"kind": "func", "name": "accept_draft_token", "args": ["is_greedy"]}]},
         {"desc": "greedy branch compares the draft to the target argmax", "checks": [{"kind": "compare", "left": "target_argmax", "right": "draft_token"}]},
     ],
     "run_check": ("assert accept_draft_token(0.0, 0.0, 0.5, 3, 3, True)\n"
                   "assert not accept_draft_token(0.0, 0.0, 0.5, 3, 4, True)\n"
                   "assert accept_draft_token(0.0, 0.0, 0.5, -1, 7, False)\n"
                   "assert not accept_draft_token(0.0, 5.0, 0.5, -1, 7, False)\n")},
    {"id": "max-chunk-logits", "path": "max_chunk.py", "runnable": True,
     "task": ("Write a function get_max_chunk_logits(vocab_size) to workspace/max_chunk.py that replicates "
              "vLLM's get_max_chunk_logits from vllm/v1/worker/gpu/spec_decode/rejection_sampler.py: the "
              "largest number of logits rows one verification chunk may hold, capped by a fixed FP32 "
              "target-logits buffer budget."),
     "facts": [{"desc": "defines get_max_chunk_logits", "checks": [{"kind": "func", "name": "get_max_chunk_logits"}]}],
     "run_check": ("assert get_max_chunk_logits(100000) == 2684\n"
                   "assert get_max_chunk_logits(2**20) == 256\n"
                   "assert get_max_chunk_logits(3 * 10**8) == 1\n")},
    {"id": "fix-next-pow2", "path": "spec_helper.py", "runnable": True,
     "seed": {"spec_helper.py": "def next_power_of_2(n: int) -> int:\n"
                                "    \"\"\"Return the smallest power of two >= n.\"\"\"\n"
                                "    return 1 << n.bit_length()\n"},
     "task": ("The function next_power_of_2 in workspace/spec_helper.py is buggy: it returns 2 for n=1 "
              "(and is wrong for every exact power of two). Read the file with read_file, then write the "
              "corrected function back with edit_file."),
     "facts": [
         {"desc": "keeps the next_power_of_2 definition", "checks": [{"kind": "func", "name": "next_power_of_2"}]},
         {"desc": "removes the buggy 1 << n.bit_length() form", "checks": [{"kind": "pow2_buggy", "not": True}]},
     ],
     "run_check": ("assert next_power_of_2(1) == 1\n"
                   "assert next_power_of_2(2) == 2\n"
                   "assert next_power_of_2(3) == 4\n"
                   "assert next_power_of_2(5) == 8\n"
                   "assert next_power_of_2(8) == 8\n")},
]
