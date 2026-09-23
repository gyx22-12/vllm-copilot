# -*- coding: utf-8 -*-
"""vLLM Copilot 网页前端（Flask）：深色聊天界面 + 建议问题 + 真实问答。

跑法：
    $env:DEEPSEEK_API_KEY = "sk-你的key"
    py -3.12 -m pip install flask      # 首次
    py -3.12 webapp.py
    浏览器打开 http://127.0.0.1:8000

设计：
- 启动时加载文档索引 + 代码索引（同 agent.load_index，一次性、只读）。
- 只开 search / search_code / write_file / edit_file / read_file（写进 workspace/ 白名单，
  不碰 corpus/ 与 vLLM 源码树）；不开 run_python / run_file —— 网页侧无任意代码执行面。
- 建议问题直接取自评测集（eval_data）：语料限定了可答范围，所以页面一打开就弹出
  「用户可能问的问题 / 可能提的代码需求」，点「换一批」重新抽签。
"""
import os
import random

# 在 import agent 之前打开写文件工具：网页 demo 要能接住「代码需求」类建议（写代码到 workspace/）。
# 仍不暴露 run_python/run_file —— 不执行任意代码。设 VLLM_COPILOT_ALLOW_WRITE=0 可关掉写文件。
os.environ.setdefault("VLLM_COPILOT_ALLOW_WRITE", "1")

from flask import Flask, jsonify, request  # noqa: E402
import agent  # noqa: E402
from eval_data import QA_SET, NATURAL_QA, CODE_GEN_QUESTIONS  # noqa: E402

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---- 建议问题池（= 评测集里「用户可能问的问题 / 可能提的代码需求」）----
_DOC = [{"category": "doc", "text": qa["q"]} for qa in QA_SET]            # 文档配置问答（50）
_CODE = [{"category": "code", "text": q} for q in NATURAL_QA.values()]   # 源码实现问答（17）
_GEN = [{"category": "gen", "text": g["task"]} for g in CODE_GEN_QUESTIONS]  # 代码需求（12）

_client = None


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/suggestions")
def suggestions():
    """每次请求都重新抽签：页面一打开、或点「换一批」，都拿到一撮新建议。"""
    doc = random.sample(_DOC, min(4, len(_DOC)))
    code = random.sample(_CODE, min(3, len(_CODE)))
    gen = random.sample(_GEN, min(3, len(_GEN)))
    return jsonify({"suggestions": doc + code + gen})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "问题不能为空。"}), 400
    if _client is None:
        return jsonify({"error": "索引尚未加载完成，请稍候再试。"}), 503
    try:
        answer, contexts = agent.run(q, _client)
    except Exception as e:  # noqa: BLE001 —— 把任何运行时错误回给前端，别让服务器崩
        return jsonify({"error": f"处理失败：{e}"}), 500
    return jsonify({"answer": answer, "contexts": contexts})


if __name__ == "__main__":
    _client = agent.get_client()
    agent.load_index(_client)
    print("前端已启动：http://127.0.0.1:8000  （Ctrl+C 退出）")
    app.run(host="127.0.0.1", port=8000, threaded=True)
