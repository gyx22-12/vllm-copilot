# -*- coding: utf-8 -*-
"""
route() 单元测试：三类意图（write_code / run_python / search）+ 65 道评测题全 search。

历史（为什么锁这些断言）：
  1. 用户 review 发现旧 route() 用 implement/kernel/compute 分「文档/源码/代码执行」，65 题里 7 道
     被误判（6 道含 implement/kernel 被拍成 search_code、1 道 compute 问句被拍成 run_python）。
     这个测试把「65 题全 search」固化——route() 一旦又变复杂/重引入关键词分派，先在这里炸。
  2. 实验 14 发现旧 route() 把 write 和 run 混在 run_python 的触发词里：「Write a Python script」
     被误路由到 run_python（hint 让 agent 跑代码而非写文件），router 根本没有「生成代码」意图。
     现拆成 write_code（写/生成 → write_file）和 run_python（跑/执行 → run_python）两路。

两个开关：VLLM_COPILOT_ALLOW_WRITE=1 开 write 工具、VLLM_COPILOT_ALLOW_RUN_PYTHON=1 开 run_python，
都默认关。route 只在对应开关打开时才分派到 write_code / run_python，否则降级 search——避免把 agent
指到不存在的工具。

用法：py -3.12 test_route.py
"""

from eval_data import QA_SET, QA_SEMANTIC
import agent


def test_all_eval_questions_route_to_search():
    """65 题 gold 全在文档 → 期望路由全 "search"（默认 write/run 开关都关）。"""
    bad = [(qa.get("id", "semantic"), agent.route(qa["q"])) for qa in QA_SET + QA_SEMANTIC
           if agent.route(qa["q"]) != "search"]
    assert not bad, f"{len(bad)} 道题被误判（期望全 search）: {bad}"


def test_default_all_search():
    """默认（开关都关）：写/跑代码的题也降级 search（对应工具没暴露，别指到不存在的工具）。"""
    assert agent.route("write a python script to compute the KV cache size") == "search"
    assert agent.route("run this python code to compute 1+1") == "search"


def test_write_code_when_write_enabled():
    """write 开、run 关：写/生成代码 → write_code；跑/执行 → 降级 search。"""
    old = agent._ALLOW_WRITE
    try:
        agent._ALLOW_WRITE = True
        # 实验 14 的教训：write 是「生成代码」，不再混进 run_python
        assert agent.route("Write a Python script that runs offline batch inference with vLLM") == "write_code"
        assert agent.route("Write a function rrf_fusion(rankings, k=60)") == "write_code"
        assert agent.route("Write a program that enables FP8 KV cache") == "write_code"
        assert agent.route("implement a class to wrap the sampler") == "write_code"
        # run/execute 不是 write 意图；run 开关没开 → search
        assert agent.route("Run this python code to compute 1+1") == "search"
    finally:
        agent._ALLOW_WRITE = old


def test_run_python_only_explicit_when_enabled():
    """run 开、write 关：只认显式「跑/执行」；write/compute 自然语言词不误判。"""
    old = agent._ALLOW_RUN_PYTHON
    try:
        agent._ALLOW_RUN_PYTHON = True
        assert agent.route("run this python code to compute 1+1") == "run_python"
        assert agent.route("execute the snippet below") == "run_python"
        # write 是生成意图，不再是 run_python 触发词（实验 14 修正）
        assert agent.route("write a python script to compute the KV cache size") == "search"
        # 用户 review 点名的误判样本：compute 问句（文档题）不拍成 run_python
        assert agent.route(
            "What fingerprint does vLLM compute to recognize a reused prompt prefix?") == "search"
    finally:
        agent._ALLOW_RUN_PYTHON = old


def test_write_and_run_together():
    """两个开关都开：write → write_code，run/execute → run_python，各归各、互不抢。"""
    old_w, old_r = agent._ALLOW_WRITE, agent._ALLOW_RUN_PYTHON
    try:
        agent._ALLOW_WRITE = True
        agent._ALLOW_RUN_PYTHON = True
        assert agent.route("write a python script to compute the KV cache size") == "write_code"
        assert agent.route("run this python code to compute 1+1") == "run_python"
        assert agent.route("How does vLLM implement prefix caching?") == "search"
    finally:
        agent._ALLOW_WRITE, agent._ALLOW_RUN_PYTHON = old_w, old_r


if __name__ == "__main__":
    test_all_eval_questions_route_to_search()
    test_default_all_search()
    test_write_code_when_write_enabled()
    test_run_python_only_explicit_when_enabled()
    test_write_and_run_together()
    print("✓ route() 单元测试通过：write_code / run_python / search 三类意图各归各，65 题全 search")
