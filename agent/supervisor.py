"""
Supervisor 规划与编排节点 (Supervisor Planning & Routing Node)
=================================================================
在分层多智能体拓扑 (Hierarchical Multi-Agent Architecture) 中充当“研究总监”。

【核心职能】：
1. 宏观课题解构：面对用户提出的非结构化课题，利用大模型递归拆解为相互独立、逻辑递进的子章节体系；
2. 动态并行映射 (Dynamic Map-Reduce via Send API)：
   利用 LangGraph 原生的 `Send` API，运行时根据大纲动态生成 N 个分支，
   且只派发"尚未完成"的章节，避免反思回流后全量重跑造成重复调研；
3. 反思回流的定向补充 (Refinement Loop)：
   当 Reflector 判定存在盲区并回流时，本节点消费 feedback_notes，
   仅针对盲区规划 1~2 个增量补充章节（分配新的 section_id），而非盲目的全量重规划；
4. 异常容灾防击穿：当 LLM JSON 输出格式偶发损坏或网络不可达时，降级到离线适配器并显式登记降级事件。
"""

import re
import logging
from typing import List, Dict, Any
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm import get_chat_model, OfflineSimulationChatModel
from core.config import settings
from core.observability import degradation_tracker
from core.parsing import structured_invoke, parse_structured, StructuredOutputError
from agent.state import ResearchState, Outline, RefinementPlan, Section, WorkerInput
from agent.prompts import SUPERVISOR_PROMPT, SUPERVISOR_REFINE_PROMPT

logger = logging.getLogger("Agent-Supervisor")

# 单次反思回流最多补充的章节数，防止反馈驱动的扇出无限膨胀
MAX_REFINE_SECTIONS = 2


def _plan_full_outline(topic: str) -> Outline:
    """首轮全量大纲规划；模型输出不合法时经重试，仍失败则降级到离线仿真适配器。"""
    prompt = SUPERVISOR_PROMPT.format(
        topic=topic,
        max_sections=settings.MAX_SECTIONS_PER_REPORT
    )
    try:
        llm = get_chat_model(settings.SUPERVISOR_MODEL, temperature=0.3)
        return structured_invoke(
            llm,
            [
                SystemMessage(content="你是一名严谨的研究规划总监，必须严格输出纯 JSON 格式数据。"),
                HumanMessage(content=prompt),
            ],
            Outline,
        )
    except StructuredOutputError as e:
        logger.warning(f"[Supervisor] 全量规划失败 ({e})，切换至离线规划适配器")
        degradation_tracker.record(f"Supervisor 全量大纲规划失败（{str(e)[:100]}），已使用离线仿真大纲")
        mock_llm = OfflineSimulationChatModel(api_key="mock", base_url="mock")
        res = mock_llm.invoke([HumanMessage(content=f"Outline 大纲生成: {topic}")])
        return parse_structured(res.content, Outline)


def _plan_refinement(topic: str, existing_sections: List[Section], feedback: List[str]) -> List[Section]:
    """
    反思回流后的定向补充规划：只产出针对盲区的新章节。
    保证至少返回 1 个补充章节（否则 Send 扇出为空会导致工作流挂起）。
    """
    existing_titles = "\n".join([f"- {s.title}" for s in existing_sections])
    feedback_text = "\n".join([f"- {f}" for f in feedback]) if feedback else "- 质检认为信息覆盖不足，请补充关键维度。"

    prompt = SUPERVISOR_REFINE_PROMPT.format(
        topic=topic,
        existing_titles=existing_titles,
        feedback=feedback_text
    )
    try:
        llm = get_chat_model(settings.SUPERVISOR_MODEL, temperature=0.3)
        plan = structured_invoke(
            llm,
            [
                SystemMessage(content="你是一名严谨的研究规划总监，必须严格输出纯 JSON 格式数据。"),
                HumanMessage(content=prompt),
            ],
            RefinementPlan,
        )
        new_sections = [
            Section(
                section_id="pending",  # 由调用方统一分配新编号
                title=s.title or "补充维度分析",
                description=s.description or "针对质检反馈盲区的补充调研",
                research_questions=s.research_questions or ["该维度有哪些关键事实与数据？"],
                content=None,
                citations=[],
                is_completed=False
            )
            for s in plan.sections
        ]
    except StructuredOutputError as e:
        logger.warning(f"[Supervisor] 补充规划失败 ({e})，切换至离线补充适配器")
        degradation_tracker.record(f"Supervisor 补充规划失败（{str(e)[:100]}），已使用离线仿真补充章节")
        mock_llm = OfflineSimulationChatModel(api_key="mock", base_url="mock")
        res = mock_llm.invoke([HumanMessage(content=f"REFINE 增量补充规划: {topic}；盲区: {feedback_text}")])
        try:
            plan = parse_structured(res.content, RefinementPlan)
        except StructuredOutputError:
            plan = RefinementPlan(sections=[])
        new_sections = [
            Section(
                section_id="pending",
                title=s.title or "补充维度分析",
                description=s.description or "针对质检反馈盲区的补充调研",
                research_questions=s.research_questions or ["该维度有哪些关键事实与数据？"],
                content=None,
                citations=[],
                is_completed=False
            )
            for s in plan.sections
        ]

    # 最终保底：若模型未给出任何有效补充章节，用反馈文本构造一个，保证扇出非空
    if not new_sections:
        new_sections = [
            Section(
                section_id="pending",
                title="补充：质检盲区定向分析",
                description="；".join(feedback) if feedback else "对首轮未覆盖维度的补充调研",
                research_questions=["首轮遗漏的核心论据有哪些？", "是否存在相互矛盾或需要交叉验证的结论？"],
                content=None,
                citations=[],
                is_completed=False
            )
        ]
    return new_sections[:MAX_REFINE_SECTIONS]


def plan_outline_node(state: ResearchState) -> Dict[str, Any]:
    """
    Supervisor 规划节点：首轮全量拆解；反思回流时消费反馈、定向补章。

    轮次计数（research_loop_count）统一在本节点维护：
    - 首轮进入：0 -> 1；
    - 每次回流：N -> N+1；Reflector 以 count >= max_loops 作为熔断条件。
    """
    topic = state.get("topic", "未知研究课题")
    existing_sections = state.get("outline_sections", [])
    current_loop = state.get("research_loop_count", 0)

    # 首轮（新任务入口）重置降级账本，确保上一次任务的事件不串扰
    if current_loop == 0:
        degradation_tracker.reset()

    if not existing_sections:
        logger.info(f"[Supervisor] 首轮顶层规划拆解: '{topic}'")
        outline = _plan_full_outline(topic)
        logger.info(f"[Supervisor] 大纲规划完成，共拆解出 {len(outline.sections)} 个子章节")
        return {
            "summary_brief": outline.summary_brief,
            "outline_sections": outline.sections,
            "research_loop_count": 1
        }

    # 反思回流：只针对反馈盲区做增量补充
    feedback = state.get("feedback_notes", [])
    logger.info(f"[Supervisor] 收到反思回流，反馈 {len(feedback)} 条，开始定向补充规划")
    new_sections = _plan_refinement(topic, existing_sections, feedback)

    # 为补充章节分配全局递增的新 section_id
    max_idx = 0
    for s in existing_sections:
        m = re.match(r"sec_(\d+)$", s.section_id or "")
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    for offset, s in enumerate(new_sections, start=1):
        s.section_id = f"sec_{max_idx + offset}"

    merged = list(existing_sections) + new_sections
    logger.info(f"[Supervisor] 补充规划完成，新增 {len(new_sections)} 个章节: "
                f"{[s.section_id for s in new_sections]}")
    return {
        "outline_sections": merged,
        "research_loop_count": current_loop + 1
    }


def route_to_workers(state: ResearchState) -> List[Send]:
    """
    基于 LangGraph Send API 的条件分发函数。

    只派发"未完成"的章节：
    - 首轮：outline_sections 全部未完成，全量扇出；
    - 回流后：已完成章节保留在 completed_sections 中，仅扇出新增补充章节，
      避免重复调研、重复计费与结果重复累积。
    """
    sections = state.get("outline_sections", [])
    topic = state.get("topic", "")
    completed_ids = {s.section_id for s in state.get("completed_sections", [])}

    pending = [sec for sec in sections if sec.section_id not in completed_ids]
    # 理论兜底：若所有章节都已完成（不应发生，reflector 会熔断），派发全部防止空扇出挂起
    if not pending:
        pending = sections

    logger.info(f"[Supervisor] 动态 Send 路由：待派发 {len(pending)} 个 Worker（已完成 {len(completed_ids)} 个）")

    return [
        Send("research_worker", {"section": sec, "topic": topic})
        for sec in pending
    ]
