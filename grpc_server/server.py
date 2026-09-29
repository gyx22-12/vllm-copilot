# -*- coding: utf-8 -*-
"""vLLM Copilot 的 gRPC 服务：把 agent.run 包成 gRPC 接口，索引常驻内存。

跑法：
    python3 server.py    # 需先有 DEEPSEEK_API_KEY（环境变量，或项目根 .env）

与 webapp.py 的差异：这里是「后端服务面」，不面向浏览器，所以不开 run_python / run_file /
write_file（gRPC 只暴露 search + search_code，最安全）。检索 / ReAct 逻辑原样复用 agent.run。
建议问题（Suggestions RPC）也复用 eval_data 的评测集，与 webapp.py 同一来源。
"""
import os
import random
import sys
from concurrent import futures

# agent.py 在项目根（本文件的上一级目录），从 grpc_server/ 启动服务时需把项目根加进 sys.path。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从项目根 .env 读 DEEPSEEK_API_KEY（若环境变量未设）。.env 已 gitignore，只进本地不入库。
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# 安全边界：服务面不暴露任意代码执行 / 写文件（在 import agent 之前固化，同 webapp.py 的做法）。
os.environ.setdefault("VLLM_COPILOT_ALLOW_WRITE", "0")
os.environ.setdefault("VLLM_COPILOT_ALLOW_RUN_PYTHON", "0")

import grpc  # noqa: E402
import agent  # noqa: E402
import copilot_pb2  # noqa: E402
import copilot_pb2_grpc  # noqa: E402
from eval_data import QA_SET, NATURAL_QA, CODE_GEN_QUESTIONS  # noqa: E402

_client = None

# ---- 建议问题池（= 评测集里「用户可能问的问题 / 可能提的代码需求」，同 webapp.py）----
_DOC = [{"category": "doc", "text": qa["q"]} for qa in QA_SET]
_CODE = [{"category": "code", "text": q} for q in NATURAL_QA.values()]
_GEN = [{"category": "gen", "text": g["task"]} for g in CODE_GEN_QUESTIONS]


class CopilotServicer(copilot_pb2_grpc.CopilotServicer):
    def Run(self, request, context):
        q = (request.query or "").strip()
        if not q:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("query 不能为空")
            return copilot_pb2.AnswerReply()
        try:
            answer, contexts = agent.run(q, _client)
        except Exception as e:  # noqa: BLE001 —— 错误回给调用方，别让服务崩
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"处理失败：{e}")
            return copilot_pb2.AnswerReply()
        return copilot_pb2.AnswerReply(answer=answer, contexts=list(contexts))

    def Suggestions(self, request, context):
        """每次请求重新抽签：4 文档 + 3 源码 + 3 代码需求，同 webapp.py 的口径。"""
        doc = random.sample(_DOC, min(4, len(_DOC)))
        code = random.sample(_CODE, min(3, len(_CODE)))
        gen = random.sample(_GEN, min(3, len(_GEN)))
        out = []
        for s in doc + code + gen:
            out.append(copilot_pb2.Suggestion(category=s["category"], text=s["text"]))
        return copilot_pb2.SuggestionsReply(suggestions=out)


def serve():
    global _client
    _client = agent.get_client()
    agent.load_index(_client)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    copilot_pb2_grpc.add_CopilotServicer_to_server(CopilotServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC 服务已启动：0.0.0.0:50051  （Ctrl+C 退出）")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
