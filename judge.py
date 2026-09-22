# -*- coding: utf-8 -*-
"""
里程碑 4：LLM-as-judge 评估忠实度（faithfulness）。

里程碑 2 的 Recall@k / MRR / nDCG 只能测「检索到没有」，测不出「生成的答案是否忠实于检索到的原文」。
这里用另一个 LLM 当裁判，对 agent 的最终回答逐条核验：

    忠实度 = 能被检索上下文支持的原子断言数 / 原子断言总数

每题流程：
  1. 跑 agent 拿到 (answer, contexts)          —— 复用 agent.run
  2. 拆解：把 answer 拆成一组「原子断言」         —— LLM 调用①
  3. 验证：逐条判断断言是否被 contexts 支持        —— LLM 调用②
  4. 忠实度 = 被支持数 / 总数，1.0 = 零编造

关键设计：
  - 拆「原子断言」而不是整体打 1~5 分：整体打分太糊（judge 容易给和稀泥的分），
    拆到「一句话一个事实」再逐条 yes/no，才能定位到底哪句在编。
  - 忠实度只衡量「有没有编造」，不衡量「答没答全」——拒绝回答（资料不足）是忠实的（1.0）
    但不完整。完整度/相关性是另外的维度，本里程碑先不做。
  - 局限：judge 和 agent 用同一个模型（deepseek-chat）会有自偏好偏差；正式评测可换更强 judge。

用法：$env:DEEPSEEK_API_KEY = "sk-..."; py -3.12 judge.py
"""

import json

from agent import load_index, get_client, run, build_search, LLM_MODEL
from eval_data import JUDGE_QUESTIONS

JUDGE_MODEL = LLM_MODEL  # 裁判模型；想更稳可换成 deepseek-reasoner 或更强模型


def _json_chat(client, system, user):
    """一次裁判调用，强制 JSON 输出，返回解析后的 dict（解析失败返回 {}）。"""
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def decompose(client, answer):
    """LLM 调用①：把 answer 拆成一组原子断言列表。"""
    system = (
        "你是忠实度评测的「拆解器」。把回答拆成一组原子断言（atomic claims）："
        "每条只陈述一个可独立判断真伪的事实。忽略寒暄、标题和代码排版，只拆对文档事实的陈述。"
        "同一事实的不同表述/同义复述只出一条，不要为凑数把一句话拆碎——"
        "一条断言对应一个「新事实」，重复重述同一个上下文事实不算新断言。"
        "不要拆纯出处标注（文件/行号引用）、代码块本身或「详见某节」这类导航语。"
        "若回答只是说「资料不足/查不到」而没断言任何事实，返回空列表。"
        '只输出 JSON：{"claims": ["...", ...]}'
    )
    data = _json_chat(client, system, f"回答：\n{answer}")
    return data.get("claims", [])


def verify(client, claims, context):
    """LLM 调用②：逐条判断断言能否被上下文支持。"""
    system = (
        "你是忠实度评测的「验证器」。给定检索上下文和一组原子断言，逐条判断断言是否被上下文支持。"
        "判断规则："
        "1) 事实断言：上下文明确写明、或只是对上下文已有信息的同义改写/直接重述，才算 supported=true；"
        "需要补充上下文里没有的新事实才能成立的（即使你自己知道它是对的），一律 supported=false。"
        "2) 元断言（描述「上下文包含/不包含什么」「文档说/没说什么」）：按它对上下文的描述是否准确判断——"
        "描述准确=supported，描述失实=false；不要因为「上下文没直接写出这句话」就判 false。"
        '只输出 JSON：{"verdicts": [{"claim": "...", "supported": true/false}, ...]}'
    )
    user = f"检索上下文：\n{context}\n\n原子断言：\n" + json.dumps(claims, ensure_ascii=False)
    data = _json_chat(client, system, user)
    return data.get("verdicts", [])


def judge_one(client, question):
    """对一题跑完整评测，返回 question/answer/claims/verdicts/faithfulness。"""
    answer, contexts = run(question, client, temperature=0)  # 评测固定温度，消除采样随机性
    context = "\n\n".join(contexts)
    claims = decompose(client, answer)
    if not claims:
        # 没拆出断言（纯拒绝/空答）：无断言 = 无编造，忠实度 1.0（但 ≠ 完整）
        return {"question": question, "answer": answer, "claims": [],
                "verdicts": [], "faithfulness": 1.0, "note": "（无原子断言：拒绝/空答）"}
    # 反啰嗦（#5）：去掉空白/逐字重复的断言——LLM 可能把同一事实换个说法多拆几条，
    # 让忠实度分母虚高、把真编造稀释掉。同义复述交给 decompose 的 prompt 合并，这里只挡逐字重复。
    claims = list(dict.fromkeys(c.strip() for c in claims if c.strip()))
    verdicts = verify(client, claims, context)
    if len(verdicts) != len(claims):
        print(f"  ⚠ 拆出 {len(claims)} 条断言，但 judge 只回了 {len(verdicts)} 条判定，可能有丢失")
    n_sup = sum(1 for v in verdicts if v.get("supported"))
    faith = n_sup / len(verdicts) if verdicts else 0.0
    return {"question": question, "answer": answer, "claims": claims,
            "verdicts": verdicts, "faithfulness": faith, "note": ""}


def calibrate(client):
    """自检：喂已知「忠实 / 编造 / 啰嗦+一句谎」样本，看 judge 能否分辨。

    若 judge 连这个都判错，说明 prompt 或裁判模型有问题，后面的分数不可信（judge 漂移）。
    第三个样本专测「啰嗦复述别把一句谎稀释掉」——反啰嗦改动（#5）的关键回归点。
    """
    print("===== 裁判自检（calibration）=====")
    ctx = "vLLM 支持 FP8 KV cache，通过 kv_cache_dtype='fp8' 开启。"
    faithful = "vLLM 通过 kv_cache_dtype='fp8' 设置支持 FP8 KV cache。"
    hallucinated = "vLLM 的 FP8 KV cache 要求显卡至少 1TB 显存。"
    for label, ans in (("忠实样本", faithful), ("编造样本", hallucinated)):
        claims = decompose(client, ans)
        verdicts = verify(client, claims, ctx) if claims else []
        n_sup = sum(1 for v in verdicts if v.get("supported"))
        print(f"  [{label}] {n_sup}/{len(verdicts)} 条被支持（期望：忠实=全支持，编造=0）")
    verbose_lie = ("vLLM 支持 FP8 KV cache，通过 kv_cache_dtype='fp8' 开启。"
                   "把 kv_cache_dtype 设成 'fp8'，引擎就会用 8-bit 浮点存 KV cache。"
                   "开启后要求显卡至少 1TB 显存。")
    claims = decompose(client, verbose_lie)
    verdicts = verify(client, claims, ctx) if claims else []
    n_sup = sum(1 for v in verdicts if v.get("supported"))
    n_lie = len(verdicts) - n_sup
    print(f"  [啰嗦+一句谎] {n_sup}/{len(verdicts)} 被支持，揪出 {n_lie} 处编造（期望：≥1 处「1TB 显存」）")
    print()


def main():
    client = get_client()
    # 和 agent.py 线上一致：带 client 加载代码索引（docstring 合成有缓存）。
    # 之前不传 client 导致 _CODE 未加载、search_code 不暴露，router 的降级行把误判静默修好，
    # 评测口径 ≠ 线上口径——这里统一成「测的就是线上跑的那个 agent」。
    load_index(client)
    calibrate(client)
    rows = []
    for i, q in enumerate(JUDGE_QUESTIONS, 1):
        print(f"\n===== [{i}/{len(JUDGE_QUESTIONS)}] {q[:70]} =====")
        r = judge_one(client, q)
        rows.append(r)
        n_sup = sum(1 for v in r["verdicts"] if v.get("supported"))
        print(f"  忠实度 = {r['faithfulness']:.2f}（{n_sup}/{len(r['verdicts'])} 条断言被支持）{r['note']}")
        for v in r["verdicts"]:
            if not v.get("supported"):
                claim = v.get("claim", "")
                print(f"    ✗ 无依据: {claim[:90]}")
                # 交叉核对：拿这条断言去全库再搜一次，看最像的 chunk 来自哪个文件（给人工抽查指路）
                top = build_search(claim, top_k=1).split("\n", 1)[0]
                print(f"       ↳ 全库重搜 top-1: {top}")
    avg = sum(r["faithfulness"] for r in rows) / len(rows)
    n_hall = sum(1 for r in rows for v in r["verdicts"] if not v.get("supported"))
    print(f"\n===== 平均忠实度 = {avg:.2f} | 共 {n_hall} 处无依据（编造）=====")
    print("（比率会被啰嗦复述稀释——「无依据条数」才是编造的关键信号，比率只作参照）")

    # 完整结果落盘，方便人工抽查——「supported」的判定同样可能错（false negative），不能只看 ✗
    with open("judge_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("完整结果（含每条断言及其判定）已写入 judge_results.json")


if __name__ == "__main__":
    main()
