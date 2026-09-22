# -*- coding: utf-8 -*-
"""
代码结构核对（code_facts）：把「代码写对没」从「查字符串」升级成「解析 AST」。

为什么旧的 substring 判分是假的：
  - 假货能过：把正确词写进注释、甚至整个文件全是注释，`"kv_cache_dtype" in code` 照样命中，
    语法检查也拦不住（纯注释文件是合法 Python）。
  - 真货被冤：正确代码 + 一行提到废弃 API 的注释，`"guided_json" not in code` 直接判挂。
根因：substring 看的是「文本里有没有这串」，不区分「注释/字符串」和「真正的代码结构」。
修法：ast.parse 之后注释根本进不了语法树；fact 改成对 AST 节点做结构断言（函数/类/调用/
关键字参数/字典键/字段/赋值/名字引用），一个改动同时堵死假阳性和假阴性。

fact 结构：{"desc": "...", "checks": [ {kind, ...}, ... ]}
  - 一个 fact 有 1..n 个 check，全部通过才算过。
  - check 支持 "not": true 取反（如「不得出现 guided_json」「不得有 1<<n.bit_length() 的 bug 形」）。
  - value 认字面常量（str/int/bool/float），且会把「NAME = 常量」的模块/函数级名字解析回常量值
    （`max_num_batched_tokens=MAX_NUM` 且 `MAX_NUM = 16384` 也算命中）。局限：不折叠算术（2**14 不解析）。

check 类型（kind）：
  import      从指定模块 from ... import 指定名字    {module, names}
  imports     任意模块 import 进指定名字            {names}
  call        调用某个名字（函数/类）               {name}
  kwarg       某次调用带关键字参数                  {name, value?}
  dict_entry  字典字面量带 key                     {key, value?}
  pair        kwarg 或 dict_entry 任一命中         {key, value?}
  func        定义某个函数                         {name, args?}
  class       定义某个类（可限定装饰器）            {name, decorator?}
  field       类里有这些字段声明                    {class, names}
  assign      给 target 赋值（可限定在 method 内）   {target, method?}
  name        标识符出现（Name/属性/import/关键字参数/字典 key/定义名）  {name}
  compare     一次比较里同时出现左右两个名字         {left, right}
  pow2_buggy  出现 1 << n.bit_length() 的 bug 形（bit_length 作用在裸 n 上）
"""
import ast


def _value_matches(node, expected, const_map):
    """value 匹配：期望是字面常量时，比 ast.Constant 的值；Name 则解析回常量值再比。"""
    if expected is None:
        return True
    if isinstance(node, ast.Constant):
        return node.value == expected
    if isinstance(node, ast.Name) and node.id in const_map:
        return const_map[node.id] == expected
    return False


def _const_map(tree):
    """收集「名字 = 字面常量」的映射（模块级 + 函数内），供 value 解析。"""
    m = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            if isinstance(n.value, ast.Constant):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        m[t.id] = n.value.value
        elif isinstance(n, ast.AnnAssign):
            if isinstance(n.target, ast.Name) and isinstance(n.value, ast.Constant):
                m[n.target.id] = n.value.value
    return m


def _call_func_name(node):
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _target_name(t):
    """赋值目标名：Name→id，Attribute(self.x)→attr。"""
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Attribute):
        return t.attr
    return None


def _decorator_names(node):
    out = []
    for d in node.decorator_list:
        if isinstance(d, ast.Name):
            out.append(d.id)
        elif isinstance(d, ast.Attribute):
            out.append(d.attr)
        elif isinstance(d, ast.Call):
            out.append(_call_func_name(d))
    return out


def _func_arg_names(fn):
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return names


def _identifier_uses(tree):
    """所有「结构性标识符」出现处：Name/属性/import/kwarg 名/字典 key/定义名/参数名。
    故意不含普通字符串字面量（Constant 值）——docstring/注释字符串提到某词不算「用」。"""
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.keyword) and n.arg is not None:
            names.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, ast.alias):
            names.add(n.name)
        elif isinstance(n, ast.Dict):
            for k in n.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    names.add(k.value)
    return names


def _check_pos(tree, c, const_map):
    kind = c["kind"]
    if kind == "import":
        want = set(c["names"])
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module == c.get("module"):
                if want <= {a.name for a in n.names}:
                    return True
        return False
    if kind == "imports":
        want = set(c["names"])
        got = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.ImportFrom, ast.Import)):
                got.update(a.name for a in n.names)
        return want <= got
    if kind == "call":
        return any(_call_func_name(n) == c["name"]
                   for n in ast.walk(tree) if isinstance(n, ast.Call))
    if kind == "kwarg":
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                for k in n.keywords:
                    if k.arg == c["name"] and _value_matches(k.value, c.get("value"), const_map):
                        return True
        return False
    if kind == "dict_entry":
        for n in ast.walk(tree):
            if isinstance(n, ast.Dict):
                for k, v in zip(n.keys, n.values):
                    if (isinstance(k, ast.Constant) and k.value == c["key"]
                            and _value_matches(v, c.get("value"), const_map)):
                        return True
        return False
    if kind == "pair":
        return _check_pos(tree, {**c, "kind": "kwarg", "name": c["key"]}, const_map) or \
               _check_pos(tree, {**c, "kind": "dict_entry"}, const_map)
    if kind == "func":
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == c["name"]:
                if all(a in _func_arg_names(n) for a in c.get("args", [])):
                    return True
        return False
    if kind == "class":
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef) and n.name == c["name"]:
                d = c.get("decorator")
                if d is None or d in _decorator_names(n):
                    return True
        return False
    if kind == "field":
        want = set(c["names"])
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef) and n.name == c["class"]:
                fields = set()
                for stmt in n.body:
                    if isinstance(stmt, ast.AnnAssign):
                        nm = _target_name(stmt.target)
                        if nm:
                            fields.add(nm)
                    elif isinstance(stmt, ast.Assign):
                        for t in stmt.targets:
                            nm = _target_name(t)
                            if nm:
                                fields.add(nm)
                if want <= fields:
                    return True
        return False
    if kind == "assign":
        target, method = c["target"], c.get("method")
        scopes = [tree] if method is None else [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method]
        for scope in scopes:
            for stmt in ast.walk(scope):
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    ts = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                    if any(_target_name(t) == target for t in ts):
                        return True
        return False
    if kind == "name":
        return c["name"] in _identifier_uses(tree)
    if kind == "compare":
        left, right = c["left"], c["right"]
        for n in ast.walk(tree):
            if isinstance(n, ast.Compare):
                ids = set()
                if isinstance(n.left, ast.Name):
                    ids.add(n.left.id)
                for comp in n.comparators:
                    if isinstance(comp, ast.Name):
                        ids.add(comp.id)
                if left in ids and right in ids:
                    return True
        return False
    if kind == "pow2_buggy":
        for n in ast.walk(tree):
            if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.LShift)
                    and isinstance(n.left, ast.Constant) and n.left.value == 1):
                right = n.right
                if (isinstance(right, ast.Call) and isinstance(right.func, ast.Attribute)
                        and right.func.attr == "bit_length" and not right.args
                        and isinstance(right.func.value, ast.Name) and right.func.value.id == "n"):
                    return True
        return False
    raise ValueError(f"未知 check 类型: {kind}")


def check_fact(tree, fact, const_map):
    """fact 的所有 check 全过才 True（支持 not 取反）。"""
    for c in fact["checks"]:
        ok = _check_pos(tree, c, const_map)
        if c.get("not"):
            ok = not ok
        if not ok:
            return False
    return True


def check_facts(code, facts):
    """把 code 解析成 AST 后逐 fact 核对，返回 [(desc, ok), ...]。语法错误则全 False。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [(f["desc"], False) for f in facts]
    cm = _const_map(tree)
    return [(f["desc"], check_fact(tree, f, cm)) for f in facts]


if __name__ == "__main__":
    # 自检：把用户实测的三种「假判分」场景钉成回归断言。
    GOOD = '''\
from vllm import LLM, SamplingParams

def main():
    llm = LLM(model="x", kv_cache_dtype="fp8", enable_prefix_caching=True)
    p = SamplingParams(temperature=0.7)
    return llm.generate(["hi"], p)
'''
    FACTS = [
        {"desc": "imports LLM and SamplingParams from vllm",
         "checks": [{"kind": "import", "module": "vllm", "names": ["LLM", "SamplingParams"]}]},
        {"desc": "instantiates the engine", "checks": [{"kind": "call", "name": "LLM"}]},
        {"desc": "enables FP8 KV cache", "checks": [{"kind": "kwarg", "name": "kv_cache_dtype", "value": "fp8"}]},
        {"desc": "enables APC", "checks": [{"kind": "kwarg", "name": "enable_prefix_caching", "value": True}]},
    ]
    assert all(ok for _, ok in check_facts(GOOD, FACTS)), "正确代码应该全过"

    # 假阳性 1：整个文件全是注释（正确词都写在注释里）→ 一项都不得过
    COMMENT_SHELL = '''\
# from vllm import LLM, SamplingParams
# llm = LLM(kv_cache_dtype="fp8", enable_prefix_caching=True)
'''
    assert not any(ok for _, ok in check_facts(COMMENT_SHELL, FACTS)), "纯注释空壳不该拿分"

    # 假阳性 2：参数值改错、正确值藏在注释里 → 值那项必须挂
    WRONG_VALUE = '''\
from vllm import LLM
# kv_cache_dtype="fp8"
llm = LLM(model="x", kv_cache_dtype="fp16", enable_prefix_caching=True)
'''
    res = dict(check_facts(WRONG_VALUE, FACTS))
    assert res["enables FP8 KV cache"] is False, "注释里的正确值不该救回写错的值"

    # 假阴性：正确代码 + 一行提到老 API 的注释 → not 那项必须过（注释进不了 AST）
    GUIDED_OK = '''\
from vllm import SamplingParams
# 旧版用 guided_json，现已废弃
p = SamplingParams(structured_outputs=foo)
'''
    not_guided = [{"desc": "no guided_json", "checks": [{"kind": "name", "name": "guided_json", "not": True}]}]
    assert check_facts(GUIDED_OK, not_guided)[0][1] is True, "提到老 API 的注释不该冤枉正确代码"

    # 常量解析：max_num_batched_tokens=MAX_NUM 且 MAX_NUM=16384 要命中
    CONST = '''\
from vllm import LLM
MAX_NUM = 16384
llm = LLM(max_num_batched_tokens=MAX_NUM)
'''
    budget = [{"desc": "budget 16384", "checks": [{"kind": "kwarg", "name": "max_num_batched_tokens", "value": 16384}]}]
    assert check_facts(CONST, budget)[0][1] is True, "名字常量应该解析回值"

    # pow2 bug 形：buggy 要被 not 判挂，fixed 要过
    buggy = [{"desc": "removes buggy", "checks": [{"kind": "pow2_buggy", "not": True}]}]
    assert check_facts("def next_power_of_2(n):\n    return 1 << n.bit_length()\n", buggy)[0][1] is False
    assert check_facts("def next_power_of_2(n):\n    return 1 << (n - 1).bit_length()\n", buggy)[0][1] is True

    print("OK：代码结构核对通过（注释空壳被拦、注释提旧 API 不冤、名字常量可解析、pow2 bug 形可判）")
