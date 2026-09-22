# -*- coding: utf-8 -*-
"""
RAG 基线（里程碑 1）：真实文档 → 分块 → 真 embedding → 检索 → DeepSeek 生成

这是从 minimal_rag.py 的"假数据 demo"升级来的"真 RAG 底座"。
里程碑 1 只求跑通完整管道、先有"能回答"的东西；分块策略、混合检索、
rerank、Agent、评测这些细节，留给里程碑 2~4。

依赖 + 跑法：
    py -3.12 -m pip install sentence-transformers openai
    $env:DEEPSEEK_API_KEY = "sk-你的key"      # PowerShell 里设置一次
    py -3.12 rag_baseline.py

RAG 四步（对照 minimal_rag.py 的注释）：
    1. 加载+分块(chunk)  2. 向量化(embed)  3. 检索(retrieve)  4. 生成(generate)
"""

import os

# 国内直连 huggingface.co 会超时（WinError 10060），走 hf-mirror.com 镜像。
# 必须在 import sentence_transformers 之前设置（下载模型时才读这个变量）。
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import glob
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHUNK_SIZE  # 结构切块大小单一来源：默认值不再这里硬编码一份会漂移的 500

# bge-v1.5 系列建议：query 加指令前缀、文档不加（和 eval_qa.py 保持一致）。
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ---------- 1. 加载真实文档 + 分块 ----------

def load_documents(dirs, extensions=(".md", ".py", ".txt")):
    """把目录下指定后缀的文件读成 (来源路径, 文本)。

    里程碑 1 拿本项目自己的文件当"知识库"，不用联网下载；
    里程碑 2 再把 dirs 换成真实开源库的文档目录。
    """
    docs = []
    for d in dirs:
        for path in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if os.path.splitext(path)[1] not in extensions:
                continue
            if "__pycache__" in path or os.path.basename(path) == "input.txt":
                continue  # 跳过缓存目录和超大无关文件（莎士比亚文本）
            try:
                with open(path, encoding="utf-8") as f:
                    docs.append((path, f.read()))
            except (UnicodeDecodeError, OSError):
                pass
    return docs


def chunk_text(text, chunk_size=500, overlap=50):
    """把一篇长文本切成若干重叠的小块（chunk）。

    为什么切块：太长塞不进 embedding 模型（有长度上限），且检索粒度会变粗。
    为什么重叠：切太碎会切断上下文，重叠一部分能保住边界处的语义。
    里程碑 2 会专门实验 chunk_size / overlap 对检索效果的影响。
    """
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def _split_by_heading_level(text, max_level=6):
    """按 markdown 标题行（# 开头）切分成 (标题, 正文) 列表，只把 level<=max_level 的标题当切点。

    max_level 控制「切多细」：6 = 所有标题（# 到 ######）都切（chunk_markdown 的粒度）；
    2 = 只按 ## 切（把 ###/#### 子节归并进父节）；1 = 只按 # 切（父节 = 整篇文档）。

    追踪代码块状态（``` ... ```）：代码块里常有 `#` 注释行（Python/bash/yaml），
    若不判断会被误当成 markdown 标题——把代码块切碎，并凭空多出"假标题前缀"。
    fence 行本身保留在 body 里，保证代码块在 chunk 里仍是完整闭合的。
    """
    sections = []
    heading, body = "", []
    in_code_block = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block   # 翻转：进入 / 退出代码块
            body.append(line)
            continue
        m = re.match(r"^(#{1,6})\s", line)
        if not in_code_block and m and len(m.group(1)) <= max_level:
            if heading or body:
                sections.append((heading, "\n".join(body)))
            heading, body = line, []
        else:
            body.append(line)
    if heading or body:
        sections.append((heading, "\n".join(body)))
    return sections


def _chunk_section(heading, body, max_chunk):
    """把一个 (标题, 正文) 节按 max_chunk 切块，标题前缀挂在每块上（chunk_markdown 的切法）。

    抽出独立函数：chunk_markdown 和 chunk_markdown_with_parent 共用同一套切法，
    保证「父段落标签」只加信息、不改切块。
    """
    if not body.strip():
        # 空段：只有标题没有正文的节（## 后紧跟下一个 ##）。旧版直接返回 []，标题文本凭空蒸发——
        # 标题本身是可检索信息，单独成 chunk，别让空段整段丢失。
        return [heading.strip()] if heading.strip() else []
    prefix = (heading + "\n") if heading else ""
    chunks, cur = [], ""
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue  # 空行只是视觉分隔，不占 chunk
        if cur and len(prefix) + len(cur) + len(stripped) + 1 > max_chunk:
            chunks.append((prefix + cur).strip())
            cur = stripped
        else:
            cur = (cur + "\n" + stripped) if cur else stripped
    if cur.strip():
        chunks.append((prefix + cur).strip())
    return chunks


def chunk_markdown(text, max_chunk=CHUNK_SIZE):
    """按结构切块：先按标题切成"节"，节内按行累积，把标题前缀挂在每一块上。

    对比 chunk_text（固定字数硬切）：
    chunk_text 会在 500 字处硬切，可能把"标题"和"答案正文"劈开，且长答案被切成的
    后续块里【没有标题】——检索时这些"孤儿块"不知道自己在回答哪个问题，容易漏检。

    chunk_markdown 的关键点：
      1. 按标题/空行边界切，不劈开一行文字；
      2. 每个 chunk 都带上"标题前缀"——长答案被切成 N 块，这 N 块都自报家门，
         检索"为什么用 AdamW"时，答案的第 2、3 块（含"解耦/二阶矩"）也能被命中。
    """
    chunks = []
    for heading, body in _split_by_heading_level(text, 6):
        chunks.extend(_chunk_section(heading, body, max_chunk))
    return chunks


def chunk_markdown_with_parent(text, max_chunk=CHUNK_SIZE, parent_level=2):
    """按结构切块，并给每块打上「父段落」标签 —— 小-to-大（small-to-big）检索用。

    和 chunk_markdown 切出【完全相同的 chunk 文本】（共用 _chunk_section），
    额外记录每个 chunk 归属的父段落全文。父段落粒度由 parent_level 定：
      2 = 最近的 ## 节（把 ###/#### 子节归并成一个大节，如「FP8 KV Cache Overview」）
      1 = 整篇文档（父段落 = 全文，适合「答案贯穿全篇」的超长文档题）
    返回 [(父段落全文, chunk 文本), ...]，与 chunk_markdown 的 chunk 顺序一一对应。

    用途：检索时用小块（精确召回），命中后把整段父段落还给 LLM——既拿到完整答案，
    又不牺牲小块 embedding 的检索精度。这是「chunk 越大喂 LLM 噪声越大」的正规解。
    """
    entries = []
    for p_heading, p_body in _split_by_heading_level(text, parent_level):
        parent_full = ((p_heading + "\n") if p_heading else "") + p_body
        subsections = _split_by_heading_level(p_body, 6)
        # 父节正文开头那段（第一个更深标题之前）在原文里挂的是父标题，这里补回去，
        # 否则它和 chunk_markdown 里「父标题开头的 intro chunk」对不上。
        if subsections and not subsections[0][0] and p_heading:
            subsections[0] = (p_heading, subsections[0][1])
        for heading, body in subsections:
            for c in _chunk_section(heading, body, max_chunk):
                entries.append((parent_full.strip(), c))
    return entries


# ---------- 2. 向量化（embedding） ----------

def embed(texts, model):
    """把文本映射成语义向量。和 minimal_rag.py 的词袋本质一样：
    「文本 → 数值向量」，只是 bge 懂语义（"自动求导"和"autograd"会靠得近）。
    normalize_embeddings=True 表示向量归一化，这样点积就 == 余弦相似度。
    """
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ---------- 3. 检索 ----------

def retrieve(query_vec, chunk_vecs, top_k=10):
    """归一化向量的点积 == 余弦相似度，取最像的前 top_k 个。"""
    scores = chunk_vecs @ query_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores[top_idx]


# ---------- 4. 生成 ----------

def build_prompt(context_chunks, query):
    context = "\n\n".join(f"[来源 {src}]\n{txt}" for src, txt in context_chunks)
    return f"""你是一个只依据资料回答的助手。请根据下面的资料回答问题：
- 如果资料里有答案，就用资料里的内容回答，并尽量说明来自哪个文件。
- 如果资料里没有答案，直接回答"资料不足，无法回答"，不要编造。

资料：
{context}

问题：{query}
"""


def main():
    # 加载模型（bge-base-en：英文语义模型，和 eval_qa.py 的语料/模型对齐）
    print("加载 embedding 模型 ...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # 1) 加载 + 分块：只加载真实 vLLM 文档（corpus/docs），
    #    不收本项目自己的 .py/.md —— 否则 eval_qa.py 里的题目会泄漏进检索结果。
    here = os.path.dirname(os.path.abspath(__file__))
    docs = load_documents([os.path.join(here, "corpus", "docs")], extensions=(".md",))
    chunks = []  # 每个元素 (来源路径, 块文本)
    for src, text in docs:
        # .md 用结构切块（按标题/段落，避免劈开"问题+答案"）；其他（.py 等）用固定字数
        fn = chunk_markdown if src.endswith(".md") else chunk_text
        for c in fn(text):
            chunks.append((src, c))
    print(f"知识库：{len(docs)} 个文件 → {len(chunks)} 个 chunk")

    # 2) 向量化所有 chunk（一次性算好，检索时直接复用，不用重复 embed）
    chunk_vecs = embed([c for _, c in chunks], model)

    # 3) 生成用 client
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "未找到 DEEPSEEK_API_KEY。请先在 PowerShell 里设置：\n"
            '  $env:DEEPSEEK_API_KEY = "sk-你的key"\n'
            "然后再运行：py -3.12 rag_baseline.py"
        )
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # 4) 交互问答
    print("\n开始提问（输入 q 或直接回车退出）\n")
    while True:
        query = input("你：").strip()
        if query.lower() in {"q", "quit", "exit", ""}:
            break
        q_vec = embed([QUERY_INSTRUCTION + query], model)[0]
        top_idx, scores = retrieve(q_vec, chunk_vecs, top_k=3)
        context_chunks = [(chunks[i][0], chunks[i][1]) for i in top_idx]
        print("\n检索到的 chunk（相似度）：")
        for (src, _), s in zip(context_chunks, scores):
            print(f"  [{s:.3f}] {os.path.basename(src)}")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": build_prompt(context_chunks, query)}],
        )
        print(f"回答：{resp.choices[0].message.content}\n")


if __name__ == "__main__":
    main()
