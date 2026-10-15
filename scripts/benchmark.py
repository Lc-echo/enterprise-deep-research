"""
可复现的离线基准脚本 (Reproducible Offline Benchmarks)
=======================================================
度量两个核心设计点的真实数据，替代文档中此前缺乏依据的"拍脑袋数字"：

1. Send 动态扇出的并发加速比（真实 LangGraph 拓扑，模拟固定耗时的 Worker）；
2. 状态剪枝（State Pruning）的字符压缩率（固定 fixture + 离线确定性模型，零网络、零 API 费用）。

运行：
    python scripts/benchmark.py
产出：
    覆盖写入 docs/benchmark.md，并在 stdout 打印汇总。

说明：
- 并发基准用 time.sleep 模拟 I/O 耗时，衡量的是"调度架构的加速上限"，
  真实网络 I/O 下的收益会因请求延迟分布、并发上限而波动；
- 压缩率基准使用离线仿真模型（确定性输出），衡量"剪枝机制的量级"，
  真实 LLM 产出的卡片长度不同，压缩率会随之变化。
"""

import sys
import json
import time
import platform
from pathlib import Path
from importlib.metadata import version as pkg_version
from typing import List, TypedDict, Annotated
import operator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage

from core.llm import OfflineSimulationChatModel
from agent.prompts import STATE_PRUNING_PROMPT

LANGGRAPH_VERSION = pkg_version("langgraph")
LANGCHAIN_CORE_VERSION = pkg_version("langchain-core")

# ============================ 基准参数 ============================
N_SECTIONS = 4          # 模拟章节数（默认 MAX_SECTIONS_PER_REPORT=4）
TASK_SECONDS = 1.5      # 每个 Worker 的模拟耗时（检索+抓取+撰写，真实环境通常 10~30 秒）
SEARCH_PER_SECTION = 2  # 每章检索子问题数（worker 代码硬编码 search_queries[:2]）
SOURCES_PER_QUERY = 2   # 每问题采纳信源数（worker 代码硬编码 valid_items_count >= 2）
FETCH_MAX_CHARS = 2500  # 单篇正文截断上限（worker 调用 fetch_web_content 的 max_length）
PRUNE_INPUT_CHARS = 1500  # 剪枝喂给模型的原文上限（worker 中 raw_content[:1500]）

# 固定的网页正文 fixture（模拟含导航/广告/正文噪声的中文技术文章，约 1500 字）
RAW_FIXTURE = (
    "首页 产品 解决方案 定价 联系我们 登录 注册 热门推荐 相关文章 广告 扫码关注领资料 "
    "长程自主智能体是大模型从单轮问答走向复杂任务自动化的关键形态。"
    "传统单智能体 ReAct 循环在跨度数小时、包含数十个工具调用的任务中，普遍出现目标漂移、上下文污染与错误累积。"
    "业界主流方案转向 Supervisor-Worker 分层拓扑：由规划模型将任务拆解为章节或子任务，"
    "通过编排框架的动态扇出原语并行派发给工作节点，每个节点持有独立子上下文，"
    "完成后仅把结构化结果回传全局状态，从而把上下文增长控制在可预期范围内。"
    "在工具接入层，Model Context Protocol（MCP）规范了工具的发现、Schema 描述与调用协议，"
    "使工具集可以作为独立进程运行，主进程通过标准输入输出或流协议通信，工具崩溃不影响编排层。"
    "状态剪枝是控制成本的核心环节：工作节点抓取的网页正文通常在数千到数万字，"
    "若直接汇入全局状态，会导致后续合成节点的上下文迅速膨胀并推高费用；"
    "通过对每篇正文做一次低温度的信息提纯，仅保留 2 到 4 条可溯源的事实陈述及其来源链接，"
    "可将进入全局状态的文本量压缩一个数量级以上。"
    "质量控制方面，反思节点在所有工作节点收敛后评估覆盖度，"
    "发现盲区时携带反馈回流规划节点做定向补充，同时用最大轮次作为硬熔断条件，避免无限循环。"
    "工程实现还需要考虑：同步工具调用放入线程池以免阻塞事件循环、"
    "降级路径必须显式标记而不是静默伪造信源、参考文献链接在合成后进行程序化校验。"
    "据公开技术博客报告，采用扇出并行后，章节调研阶段的墙钟时间接近单个最慢节点的耗时，"
    "而不是所有节点耗时之和；剪枝后的事实卡片也显著降低了合成节点的输入 token 数量。"
) * 3  # 重复至超过 1500 字，模拟真实网页长度


def benchmark_parallelism() -> dict:
    """基准 1：真实 LangGraph Send 扇出的并发加速。"""

    class S(TypedDict):
        done: Annotated[List[int], operator.add]

    def slow_worker(state):
        time.sleep(TASK_SECONDS)
        return {"done": [state["i"]]}

    def fanout(state):
        return [Send("slow_worker", {"i": i}) for i in range(N_SECTIONS)]

    b = StateGraph(S)
    b.add_node("slow_worker", slow_worker)
    b.add_conditional_edges(START, fanout, ["slow_worker"])
    b.add_edge("slow_worker", END)
    graph = b.compile()

    # 串行基线
    t0 = time.perf_counter()
    for i in range(N_SECTIONS):
        slow_worker({"i": i})
    serial_seconds = time.perf_counter() - t0

    # 并发（Send 扇出）
    t0 = time.perf_counter()
    graph.invoke({"done": []})
    parallel_seconds = time.perf_counter() - t0

    speedup = serial_seconds / parallel_seconds if parallel_seconds else 0
    reduction = (1 - parallel_seconds / serial_seconds) * 100 if serial_seconds else 0
    return {
        "n": N_SECTIONS,
        "task_seconds": TASK_SECONDS,
        "serial_seconds": round(serial_seconds, 2),
        "parallel_seconds": round(parallel_seconds, 2),
        "speedup": round(speedup, 2),
        "reduction_pct": round(reduction, 1),
    }


def benchmark_pruning() -> dict:
    """基准 2：状态剪枝压缩率（离线确定性模型 + 固定 fixture）。"""
    raw = RAW_FIXTURE[:PRUNE_INPUT_CHARS]
    sim = OfflineSimulationChatModel(api_key="mock", base_url="mock")
    prompt = STATE_PRUNING_PROMPT.format(
        section_title="长程智能体工程化",
        source_title="固定基准测试信源",
        raw_content=raw,
    )
    res = sim.invoke([HumanMessage(content=prompt)])
    facts = json.loads(res.content)["key_facts"]
    compressed = sum(len(f) for f in facts)
    reduction = (1 - compressed / len(raw)) * 100

    # 架构级理论测算：默认配置下抓取原文总量 vs 剪枝卡片总量
    notes = N_SECTIONS * SEARCH_PER_SECTION * SOURCES_PER_QUERY
    raw_total = notes * FETCH_MAX_CHARS
    # 按本次实测的卡片长度推算：每篇正文产出 1 张 note，每张约 3 条事实
    avg_fact_len = compressed / len(facts)
    pruned_total = notes * avg_fact_len * 3
    theoretical_reduction = (1 - pruned_total / raw_total) * 100

    return {
        "raw_chars": len(raw),
        "facts": len(facts),
        "pruned_chars": compressed,
        "reduction_pct": round(reduction, 1),
        "notes_per_report": notes,
        "raw_total_chars": raw_total,
        "pruned_total_chars": int(pruned_total),
        "theoretical_reduction_pct": round(theoretical_reduction, 1),
    }


def main():
    par = benchmark_parallelism()
    pru = benchmark_pruning()

    print("=== 基准 1：Send 扇出并发加速 ===")
    print(f"串行 {par['serial_seconds']}s  vs  并发 {par['parallel_seconds']}s  "
          f"加速比 {par['speedup']}x  耗时降低 {par['reduction_pct']}%")
    print("=== 基准 2：状态剪枝压缩率 ===")
    print(f"单篇 {pru['raw_chars']} 字 → {pru['pruned_chars']} 字（{pru['facts']} 条事实），"
          f"压缩率 {pru['reduction_pct']}%")
    print(f"全报告测算：{pru['raw_total_chars']} 字 → 约 {pru['pruned_total_chars']} 字，"
          f"理论压缩率 {pru['theoretical_reduction_pct']}%")

    md = f"""# 基准数据 (Benchmarks)

> 由 `scripts/benchmark.py` 自动生成，全程离线、可复现，不产生 API 费用。
> 运行方式：`python scripts/benchmark.py`（约 10 秒）

## 环境

- Python {platform.python_version()}
- langgraph {LANGGRAPH_VERSION}
- langchain-core {LANGCHAIN_CORE_VERSION}

## 1. Send 动态扇出的并发加速

模拟 {par['n']} 个章节、每个 Worker 固定耗时 {par['task_seconds']} 秒（检索 + 抓取 + 撰写的 I/O 型负载）：

| 调度方式 | 端到端耗时 |
|:---|---:|
| 串行（逐章节） | {par['serial_seconds']} 秒 |
| 并发（LangGraph Send 扇出） | {par['parallel_seconds']} 秒 |
| **加速比 / 耗时降低** | **{par['speedup']}x / {par['reduction_pct']}%** |

口径说明：该数据衡量调度架构的加速上限（墙钟时间趋近最慢节点而非各节点之和）；真实网络 I/O 下受延迟分布、并发线程数影响会有波动。

## 2. 状态剪枝（State Pruning）压缩率

单篇网页正文（固定 {pru['raw_chars']} 字 fixture）经离线确定性模型提纯为 {pru['facts']} 条事实卡片：

| 项目 | 字符数 |
|:---|---:|
| 剪枝前（进入全局状态的原始正文） | {pru['raw_chars']} |
| 剪枝后（事实卡片） | {pru['pruned_chars']} |
| **单篇压缩率** | **{pru['reduction_pct']}%** |

默认配置（{N_SECTIONS} 章 × {SEARCH_PER_SECTION} 问 × {SOURCES_PER_QUERY} 信源 = {pru['notes_per_report']} 篇正文）的架构级测算：

| 项目 | 字符数 |
|:---|---:|
| 剪枝前总量（每篇按 {FETCH_MAX_CHARS} 字上限） | {pru['raw_total_chars']} |
| 剪枝后总量（按本次实测卡片长度推算） | {pru['pruned_total_chars']} |
| **理论压缩率** | **{pru['theoretical_reduction_pct']}%** |

口径说明：剪枝使用离线仿真模型（确定性输出），衡量机制量级；真实 LLM 产出的卡片长度不同，压缩率会波动。

## 3. 关于"溯源准确率"

本项目不使用"100% 可验证溯源"这类绝对表述。实际机制为：
- 仅保留真实抓取到的 URL，伪造 / 占位域名一律剔除；
- 合成后程序化校验参考文献链接，缺失则用真实信源清单替换；
- 当全链路无任何真实信源时，报告顶部插入降级横幅、文末替换为降级声明，而非伪造文献。
"""
    out = ROOT / "docs" / "benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
