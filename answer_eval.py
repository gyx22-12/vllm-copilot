# -*- coding: utf-8 -*-
"""里程碑 6：答案正确性评测（reference-answer 对比）。

补齐 RAG 评测三件套的最后一条腿：
  - Recall@k（eval_qa.py）＝ 检索到没（过程指标）
  - 忠实度（judge.py）＝ 有没有编造（过程指标）
  - 正确性（本脚本）＝ 答对了没（结果指标）——拿手写「必须事实」逐条核对

动机：关键词命中 ≠ 回答正确。一个 agent 可能检索到正确 chunk、忠实复述它，却漏说关键事实
或答非所问。这里把参考答案拆成 2~4 条「必须正确陈述的事实」（见 eval_data.ANSWER_QUESTIONS，
事实从 corpus/docs 核着写、是 ground truth），跑 agent 拿答案 → LLM judge 逐条判断「事实是否
被正确陈述」→ 正确率 = 命中事实 / 总事实。

与 judge.py 的区别：judge.py 判断「回答里的每个声明有没有上下文支撑」（忠实度，防编造）；
本脚本判断「标准答案里的每条必须事实有没有被回答说出来」（正确性，防漏/防错）。方向相反、正交。

用法：$env:DEEPSEEK_API_KEY = "sk-..."; py -3.12 answer_eval.py
"""

import json

from agent import load_index, get_client, run
from eval_data import ANSWER_QUESTIONS
from judge import _json_chat  # 复用 judge 的 JSON 裁判调用（同一模型、同一 response_format，单一来源）


def judge_facts(client, answer, facts):
    """逐条判断「必须事实」是否被回答正确陈述，返回 {fact, stated} 列表。"""
    system = (
        "你是答案正确性评测的裁判。给定一组「标准答案要点」（每条是一个必须正确陈述的事实）"
        "和一份模型生成的回答，逐条判断该事实是否被回答正确陈述。"
        "判断规则："
        "1) 回答明确写出该事实、或写出语义等价的表述（数值/名称/机制正确即可，不要求逐字），判 stated=true。"
        "2) 回答遗漏该事实、或写出的内容与该事实矛盾/数值错误，判 stated=false。"
        '只输出 JSON：{"verdicts": [{"fact": "...", "stated": true/false}, ...]}'
    )
    user = f"标准答案要点：\n{json.dumps(facts, ensure_ascii=False)}\n\n模型回答：\n{answer}"
    data = _json_chat(client, system, user)
    return data.get("verdicts", [])


def main():
    client = get_client()
    load_index(client)  # 和 judge.py 一致：加载完整索引（含代码），测的就是线上跑的那个 agent

    rows = []
    for i, item in enumerate(ANSWER_QUESTIONS, 1):
        print(f"\n===== [{i}/{len(ANSWER_QUESTIONS)}] {item['q'][:70]} =====")
        answer, _ = run(item["q"], client, temperature=0)
        verdicts = judge_facts(client, answer, item["facts"])
        if len(verdicts) != len(item["facts"]):
            print(f"  ⚠ 标准答案 {len(item['facts'])} 条事实，judge 只返回 {len(verdicts)} 条判定，可能有丢失")
        n_hit = sum(1 for v in verdicts if v.get("stated"))
        acc = n_hit / len(item["facts"]) if item["facts"] else 0.0
        rows.append({"q": item["q"], "answer": answer, "facts": item["facts"],
                     "verdicts": verdicts, "accuracy": acc})
        print(f"  正确率 = {acc:.2f}（{n_hit}/{len(item['facts'])} 条事实被正确陈述）")
        for v in verdicts:
            if not v.get("stated"):
                print(f"    ✗ 漏/错: {v.get('fact', '')[:100]}")

    total = sum(len(r["facts"]) for r in rows)
    n_hit = sum(sum(1 for v in r["verdicts"] if v.get("stated")) for r in rows)
    avg = sum(r["accuracy"] for r in rows) / len(rows)
    print(f"\n===== 平均正确率 = {avg:.2f}（{n_hit}/{total} 条必须事实被正确陈述，{total - n_hit} 条漏/错）=====")
    print("（正确率 = 必须事实覆盖率；与 Recall@k（检索到没）、忠实度（编没编）正交，三层合起来才完整）")

    with open("answer_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("完整结果已写入 answer_eval_results.json")


if __name__ == "__main__":
    main()
