"""
反思闭环端到端测试（离线 LLM + 假搜索）：
验证首轮判不足时，回流只增量补充新章节，不重复全量扇出、不产生重复 section_id。
"""

import pytest
from langgraph.graph import StateGraph, START, END

from agent.state import ResearchState
from agent.supervisor import plan_outline_node, route_to_workers
from agent.worker import research_worker_node
from agent.reflector import route_after_reflection
from agent.synthesizer import synthesizer_node


def _build_graph(gap_reflector):
    """编译一张 reflector 行为可定制的测试图。"""
    builder = StateGraph(ResearchState)
    builder.add_node("supervisor", plan_outline_node)
    builder.add_node("research_worker", research_worker_node)
    builder.add_node("reflector", gap_reflector)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_to_workers, ["research_worker"])
    builder.add_edge("research_worker", "reflector")
    builder.add_conditional_edges(
        "reflector",
        route_after_reflection,
        {"synthesizer": "synthesizer", "supervisor": "supervisor"},
    )
    builder.add_edge("synthesizer", END)
    return builder.compile()


def _initial_state(topic):
    return {
        "topic": topic,
        "summary_brief": "",
        "outline_sections": [],
        "completed_sections": [],
        "global_notes": [],
        "research_loop_count": 0,
        "max_loops": 2,
        "is_sufficient": False,
        "feedback_notes": [],
        "final_report": None,
        "total_sources_count": 0,
        "degradation_events": [],
    }


def test_refinement_adds_new_sections_without_duplicates(offline_llm, fake_search):
    """首轮不足 → 回流补章：只新增 sec_4，sec_1~3 不重复调研。"""
    calls = {"n": 0}

    def gap_then_pass(state):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "is_sufficient": False,
                "feedback_notes": ["缺少成本与商业化维度的量化分析"],
            }
        return {"is_sufficient": True, "feedback_notes": ["补充后覆盖完整"]}

    graph = _build_graph(gap_then_pass)
    final_state = graph.invoke(_initial_state("pytest_loop_refinement"))

    ids = [s.section_id for s in final_state["completed_sections"]]

    # 首轮 3 个 + 补充 1 个，共 4 个，且无重复
    assert len(ids) == 4
    assert len(set(ids)) == 4
    assert "sec_4" in ids
    # 补充规划确实被执行了两次 reflector（一次判不足，一次放行）
    assert calls["n"] == 2
    # 轮次计数最终为 2（首轮 + 一次回流）
    assert final_state["research_loop_count"] == 2


def test_single_round_has_no_refinement(offline_llm, fake_search):
    """首轮即判定充分：不发生补章，只产出首轮 3 个章节。"""
    def always_pass(state):
        return {"is_sufficient": True, "feedback_notes": ["覆盖完整"]}

    graph = _build_graph(always_pass)
    final_state = graph.invoke(_initial_state("pytest_loop_single"))

    ids = [s.section_id for s in final_state["completed_sections"]]
    assert len(ids) == 3
    assert set(ids) == {"sec_1", "sec_2", "sec_3"}
    assert final_state["research_loop_count"] == 1


def test_real_sources_flow_into_report_with_no_fake_domains(offline_llm, fake_search):
    """假搜索提供的真实 URL 必须进入报告；任何仿真域名不得出现。"""
    def always_pass(state):
        return {"is_sufficient": True, "feedback_notes": ["覆盖完整"]}

    graph = _build_graph(always_pass)
    final_state = graph.invoke(_initial_state("pytest_loop_sources"))

    report = final_state["final_report"]
    assert "arxiv.org" in report
    assert "developer.aliyun.com" in report
    for fake_domain in ["research.benchmark.org", "agent-systems.org", "mcp-protocol.org"]:
        assert fake_domain not in report
    assert final_state["total_sources_count"] >= 1


def test_refinement_plan_schema_without_section_id():
    """验证 RefinementPlan 支持不包含 section_id 的模型输出（符合 prompt 规范）。"""
    from core.parsing import parse_structured
    from agent.state import RefinementPlan

    raw_json = """
    {
      "sections": [
        {
          "title": "市场规模测算与结构性变化",
          "description": "探讨中国 AI Agent 市场的潜在总空间与渗透节奏",
          "research_questions": ["未来3年市场规模如何测算？", "头部集中度如何变化？"]
        }
      ]
    }
    """
    plan = parse_structured(raw_json, RefinementPlan)
    assert len(plan.sections) == 1
    assert plan.sections[0].title == "市场规模测算与结构性变化"
    assert len(plan.sections[0].research_questions) == 2
