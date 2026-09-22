# -*- coding: utf-8 -*-
"""
代码感知切块：用 AST 把 Python 源码按「符号」（函数 / 类 / 方法）切成检索单元。

和 chunk_markdown 的区别（为什么不能复用）：
  - markdown 靠 # 标题切——结构在「标题」里
  - 代码靠 AST 切——结构在「语法树」里
  固定长度切会把一个函数拦腰截断，检索出来是半截函数，答不了「怎么实现」。

每个 chunk = 一个自足的符号，拼两样东西：
  1. 出处头（相对路径:起始-结束行号 | 类型 符号名）——生成时逐句 cite 用
  2. 该符号的完整源码段（签名 + docstring + 函数体）——embedding 读得懂 + LLM 看得全

切块规则（第一版）：
  - 每个顶层 FunctionDef / AsyncFunctionDef → 一个 chunk
  - 每个 ClassDef → 一个「类概览」chunk（类 docstring + 字段声明 + 方法签名一览）
  - 类里的每个方法 → 一个 chunk（header 带 类名.方法名）
  - 模块级常量（全大写名，如 MAX_CHUNK_BYTES = 2**30）→ 一个「常量」chunk
  - 装饰器（@override / @register_* 等）归入它修饰的符号 chunk——行号在 def 之前，是语义关键标记
  - 模块 docstring → 一个「文件概览」chunk（这文件是干嘛的）

补丁（实验 15）：符号粒度索引曾把「模块常量」和「类字段」整类漏掉——函数体里只有
常量名没有值（`max(1, MAX_CHUNK_BYTES // ...)` 拿不到 `= 2**30`），数据类的字段注解
也不在索引里。已补：模块级 Assign/AnnAssign（全大写名）各成一个常量 chunk；类字段
（Assign/AnnAssign）收进类概览 chunk。

用法：
  from code_chunker import chunk_python
  for text, start_line in chunk_python("vllm/.../mtp.py", source_str):
      ...

注：单个函数若极长（罕见），第一版保持完整不切（检索单元=符号，宁完整不截断），
    后续可加「超长函数按逻辑块再切」的降级策略。
"""

import ast


def _node_start(node):
    """节点真实起始行：带装饰器时取第一个装饰器行（decorator_list 在 def 之前）。"""
    start = node.lineno
    for d in getattr(node, "decorator_list", None) or ():
        start = min(start, d.lineno)
    return start


def _decorator_text(source, node):
    """节点的装饰器源码（@xxx 行，含多行装饰器）；无装饰器返回 ""。"""
    decos = getattr(node, "decorator_list", None) or []
    if not decos:
        return ""
    lines = source.splitlines()
    out = []
    for d in decos:
        d_end = getattr(d, "end_lineno", None) or d.lineno
        for ln in range(d.lineno, d_end + 1):
            out.append(lines[ln - 1].strip())
    return "\n".join(out)


def _node_source(source, node):
    """取节点对应的原始源码段（装饰器 + 签名/docstring/函数体）。

    坑：ast.get_source_segment 只从 node.lineno（def 那一行）起取，而装饰器在
    decorator_list 里、行号在 def 之前——不补的话 @override/@register_* 这类
    「覆写哪个父类 / 注册到哪个后端」的语义标记会整段丢失，语义检索直接失效。
    """
    body = ast.get_source_segment(source, node) or ""
    deco = _decorator_text(source, node)
    return f"{deco}\n{body}" if deco else body


def _header(path, node, kind, qualname):
    """出处头：[source 相对路径:起始-结束行 | 类型 符号名]"""
    start = _node_start(node)
    end = getattr(node, "end_lineno", None) or node.lineno
    return f"[source {path}:{start}-{end} | {kind} {qualname}]"


def _func_chunk(path, source, node, kind, qualname=None):
    """把一个函数/方法打包成一个 chunk。qualname = 全限定名（如 Outer.Inner.method），默认节点名。"""
    text = _header(path, node, kind, qualname or node.name) + "\n" + _node_source(source, node)
    return (text, _node_start(node))


def _assign_names(node):
    """赋值目标的 Name 列表（Assign 有 targets，AnnAssign 只有单个 target）。"""
    targets = getattr(node, "targets", None)
    if targets is None:  # AnnAssign
        targets = [node.target]
    return [t.id for t in targets if isinstance(t, ast.Name)]


def _is_const_name(name):
    """PEP8 常量命名：全大写（允许下划线/数字/前导下划线）。只索引常量，避免把
    scheduler = Scheduler() 这类模块级初始化也切进来。"""
    return name.isupper()


def _const_chunk(path, source, node, names):
    """模块级常量 chunk：一个赋值一个 chunk。常量是「实现细节」的关键锚点——
    函数体只引用常量名（`MAX_CHUNK_BYTES // ...`），值（`= 2**30`）得靠这一块才能检到。"""
    qualname = ",".join(names)
    text = _header(path, node, "constant", qualname) + "\n" + _node_source(source, node)
    return (text, _node_start(node))


def _class_overview(path, source, node, qualname=None):
    """类的「目录」chunk：装饰器 + docstring + 字段声明 + 方法签名一览，回答「这类的职责/形状」。

    qualname = 全限定名（嵌套类 Outer.Inner），默认节点名。"""
    doc = ast.get_docstring(node) or ""
    deco = _decorator_text(source, node)
    fields = []
    for f in node.body:
        if isinstance(f, (ast.Assign, ast.AnnAssign)):
            # 字段/类属性声明（dataclass 的 `x: torch.Tensor` 或普通类属性 `DEFAULT = 5`）。
            # 用完整源码段（不是 def 单行）——字段可能带默认值跨行。
            seg = ast.get_source_segment(source, f)
            if seg:
                fields.append("    " + seg.strip())
    method_sigs = []
    for m in node.body:
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # 只取 def 那一行作为签名（不展开函数体，避免和 method chunk 重复且臃肿）
            first_line = source.splitlines()[m.lineno - 1].strip()
            method_sigs.append("    " + first_line)
    if not doc and not deco and not fields and not method_sigs:
        return None
    header = _header(path, node, "class", qualname or node.name)
    parts = [header]
    if deco:
        parts.append(deco)
    if doc:
        parts.append(doc)
    if fields:
        parts.append("# 字段")
        parts.extend(fields)
    if method_sigs:
        parts.append("# 方法签名")
        parts.extend(method_sigs)
    return ("\n".join(parts), _node_start(node))


def _module_overview(path, source, tree):
    """文件概览 chunk：模块 docstring，回答「这文件是干嘛的」。无 docstring 则跳过。"""
    doc = ast.get_docstring(tree)
    if not doc:
        return None
    header = f"[source {path}:1 | module overview]"
    return (header + "\n" + doc, 1)


def _chunk_nodes(path, source, nodes, qual, in_class, out):
    """递归切节点列表（顶层 body 或类/函数 body）：函数/类/常量各成 chunk。

    补丁（实验 17）：旧版只遍历 tree.body 顶层，嵌套类（class 定义在别的 class/function 里）
    整类丢失。这里递归下钻，嵌套类/方法带全限定名（Outer.Inner / Outer.Inner.method）。
    qual = 父符号全名（'' 表示模块级），in_class = 当前是否在类体内（决定方法 vs 函数 kind）。
    """
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            full = f"{qual}.{node.name}" if qual else node.name
            kind = "method" if in_class else "function"
            out.append(_func_chunk(path, source, node, kind=kind, qualname=full))
            _chunk_nodes(path, source, node.body, qual=full, in_class=False, out=out)
        elif isinstance(node, ast.ClassDef):
            full = f"{qual}.{node.name}" if qual else node.name
            co = _class_overview(path, source, node, qualname=full)
            if co:
                out.append(co)
            _chunk_nodes(path, source, node.body, qual=full, in_class=True, out=out)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and not in_class:
            # 模块级常量（全大写名）单独成 chunk；类字段已收进类概览 chunk，不再单独切
            names = _assign_names(node)
            consts = [n for n in names if _is_const_name(n)]
            if consts:
                out.append(_const_chunk(path, source, node, consts))


def chunk_python(path, source):
    """按 AST 切一个 .py 文件，返回 [(text, start_line), ...]。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # 语法错误 / 非纯 Python：退化成「整个文件一个 chunk」，不阻塞 ingest
        return [(f"[source {path}:1 | whole-file (parse failed)]\n{source}", 1)]

    chunks = []
    ov = _module_overview(path, source, tree)
    if ov:
        chunks.append(ov)

    _chunk_nodes(path, source, tree.body, qual="", in_class=False, out=chunks)
    return chunks


if __name__ == "__main__":
    # 自检：不依赖真实 vLLM 文件，用一段内联样例验证切块边界。
    sample = '''\
"""A tiny proposal worker to demonstrate code-aware chunking."""

import torch

MAX_STEPS = 4
_FP32_BYTES = 4


class ProposerWorker:
    """Runs the draft model and returns proposals."""

    draft_batch_size: int = 32

    class DraftCache:
        """Caches draft tokens for the worker."""
        capacity: int = 8

    def __init__(self, draft_model):
        self.draft_model = draft_model

    @override
    def propose(self, hidden_states, num_steps):
        """Run num_steps forward passes and collect draft tokens."""
        logits = []
        for _ in range(num_steps):
            logits.append(self.draft_model(hidden_states))
        return logits


@register_model("proposal")
def verify(proposals, target):
    """Accept the first matching token (mock)."""
    return proposals[0] == target
'''

    chunks = chunk_python("vllm/mtp.py", sample)
    print(f"源码 {len(sample.splitlines())} 行，切出 {len(chunks)} 个 chunk：\n")
    for text, line in chunks:
        print("=" * 70)
        print(text)
        print()

    # 回归断言：装饰器必须进 chunk（@override 语义标记丢失会毁掉「谁覆写了 forward」检索）
    joined = "\n".join(t for t, _ in chunks)
    assert "@override" in joined, "装饰器 @override 丢失"
    assert "@register_model" in joined, "装饰器 @register_model 丢失"
    # 实验 15 补丁：模块常量 + 类字段必须进 chunk（否则函数体只引用常量名、拿不到值）
    assert "constant MAX_STEPS" in joined, "模块常量 MAX_STEPS 未成 chunk"
    assert "MAX_STEPS = 4" in joined, "常量值 MAX_STEPS = 4 丢失"
    assert "_FP32_BYTES = 4" in joined, "模块常量 _FP32_BYTES 未成 chunk"
    assert "draft_batch_size: int = 32" in joined, "类字段 draft_batch_size 未进类概览"
    # 实验 17 补丁：嵌套类不能整类丢失（旧版只遍历顶层，class 套 class 就漏了）
    assert "class ProposerWorker.DraftCache" in joined, "嵌套类 DraftCache 未成 chunk"
    assert "capacity: int = 8" in joined, "嵌套类字段 capacity 未进类概览"
    print("OK：装饰器 / 模块常量 / 类字段 / 嵌套类均已进 chunk")
