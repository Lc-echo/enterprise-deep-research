"""
Reflector 质检反思与安全熔断节点 (Reflector & Circuit Breaker)
===================================================================
在多智能体拓扑中扮演“独立质量审查与安全把关委员会”。

【核心架构机制】：
1. 质量盲区评估 (Gap & Saturation Analysis)：
   在所有并发 Worker 完成正文初稿后，汇聚各章节的核心论点，
   评审信息覆盖度是否已对用户课题形成闭环，是否存在核心矛盾被遗漏；
2. 安全熔断器模式 (Circuit Breaker Pattern)：
   自主智能体最危险的缺陷是死循环 (Infinite Loop)。一旦模型反复认为信息不充分，
   就会不断回流触发新的检索，导致 API 账单指数级激增。
   本节点引入硬性最大迭代阈值 (`max_loops`)：无论判定模型给出何种反馈，
   一旦达到轮次上限无条件强制熔断，进入最终研报合成；
3. 自适应条件路由 (Conditional Loop Routing)：
   通过 `route_after_reflection` 条件分支，动态决定是“结束调研流转至 Synthesizer”，
   还是“带着反馈意见回流至 Supervisor 进行二次定向补充”。
"""

import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm import get_chat_model
from core.config import settings
from core.observability import degradation_tracker
from core.parsing import structured_invoke
from agent.state import ResearchState, ReflectionVerdict
from agent.prompts import REFLECTOR_PROMPT

logger = logging.getLogger("Agent-Reflector")


def reflect_node(state: ResearchState) -> Dict[str, Any]:
    """
    Reflector 节点核心质检函数。
    
    【执行流水线】：
    1. 读取当前研究轮次 `current_loop` 与熔断阈值 `max_loops`；
    2. [安全门禁检查]：若达到熔断上限，立即短路返回 `is_sufficient=True`，终止循环；
    3. 汇总当前已完成的章节标题与摘要片段；
    4. 驱动大模型审查论据覆盖度与逻辑严密性；
    5. 返回更新字段，将轮次计数加 1 并更新 `is_sufficient` 判定结论。
    
    Args:
        state: 全局状态机字典
        
    Returns:
        包含 is_sufficient、feedback_notes 与递增 loop 计数的更新字典
    """
    topic = state.get("topic", "")
    current_loop = state.get("research_loop_count", 1)
    max_loops = state.get("max_loops", settings.MAX_RESEARCH_LOOPS)
    completed_sections = state.get("completed_sections", [])
    
    logger.info(f"[Reflector] 开始第 {current_loop}/{max_loops} 轮反思评审，当前已聚合 {len(completed_sections)} 个章节")
    
    # -------------------------------------------------------------------------
    # 1. 安全熔断器检查 (Circuit Breaker Enforcement)
    # -------------------------------------------------------------------------
    # 达到预设上限（默认 2 轮）时无条件熔断，确保系统在生产环境具备确定性退出保证
    if current_loop >= max_loops:
        logger.info(f"[Reflector] 已达最大反思轮次限制 ({max_loops})，触发安全熔断，流转至最终研报合成")
        return {
            "is_sufficient": True,
            "feedback_notes": ["已达设定的最大安全迭代轮数，自动流转至合成阶段。"]
        }
        
    # -------------------------------------------------------------------------
    # 2. 构建章节正文摘要视图 (Context Compression for Evaluation)
    # -------------------------------------------------------------------------
    summary_lines = []
    for s in completed_sections:
        content_preview = (s.content[:150] + "...") if s.content else "无正文"
        summary_lines.append(f"【{s.title}】: {content_preview}")
        
    completed_summary = "\n".join(summary_lines)
    prompt = REFLECTOR_PROMPT.format(
        topic=topic,
        completed_summary=completed_summary,
        current_loop=current_loop,
        max_loops=max_loops
    )
    
    # -------------------------------------------------------------------------
    # 3. 驱动大模型进行严谨的质量与盲区评估
    # -------------------------------------------------------------------------
    try:
        llm = get_chat_model(settings.SUPERVISOR_MODEL, temperature=0.1)
        verdict = structured_invoke(
            llm,
            [
                SystemMessage(content="你是一名严格的研报质检专家，必须输出纯 JSON 格式。"),
                HumanMessage(content=prompt),
            ],
            ReflectionVerdict,
        )
        is_sufficient = verdict.is_sufficient
        feedback = verdict.feedback_notes or ["评估完成"]
    except Exception as e:
        # 异常情况下保守放行以保障交付不卡死，但必须显式登记降级，不伪装成正常质检
        logger.warning(f"[Reflector] 质检解析失败，保守放行: {e}")
        degradation_tracker.record(f"Reflector 质检模型输出解析失败（{str(e)[:100]}），本轮保守判定为充分")
        is_sufficient = True
        feedback = ["质检模型输出解析失败，系统保守判定为充分，建议在稳定网络环境复跑。"]
        
    logger.info(f"[Reflector] 评审判定结论: is_sufficient={is_sufficient}, feedback={feedback}")

    # 轮次计数由 Supervisor 节点统一维护（回流时 +1），此处不再递增，避免双重计数
    return {
        "is_sufficient": is_sufficient,
        "feedback_notes": feedback
    }


def route_after_reflection(state: ResearchState) -> str:
    """
    反思节点后的条件路由判断函数 (Conditional Loop / Finish Router)。
    
    【LangGraph 边路由机制】：
    此函数作为条件边的决策函数，其返回值必须映射到图中已注册的目标节点名称：
    - 返回 "synthesizer"：说明信息已充分或熔断触发，流转至长文研报终审合成节点；
    - 返回 "supervisor"：说明评审认为信息存在关键缺口，流水线闭环回流至 Supervisor，
      重新规划定向补充章节，形成经典的自主迭代学习环 (Self-Correction Loop)。
    
    Args:
        state: 当前全局状态机
        
    Returns:
        str: 下一阶段目标节点标识字符串 ("synthesizer" 或 "supervisor")
    """
    if state.get("is_sufficient", True):
        return "synthesizer"
    return "supervisor"

