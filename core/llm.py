"""
大模型工厂模块 (LLM Factory & Resilience Module)
===================================================
1. 统一接口抽象：基于 LangChain-OpenAI 标准规范，兼容任意支持 OpenAI API 协议的模型端点
   （如阿里云通义千问 DashScope、DeepSeek、OpenAI 官方服务、Ollama 本地模型等）。
2. 主动容灾降级 (Active Failover Pattern)：实现 `ResilientFailoverChatModel` 包装器。
   当遇到 403 额度不足 (AllocationQuota)、网络抖动、DNS 故障或服务端超时错误时，
   系统会自动、无缝地平滑降级至 `OfflineSimulationChatModel`，避免整条流水线崩溃。
3. 确定性仿真机制：离线仿真器能动态感知当前 Prompt 所处的研究阶段（大纲拆解、事实剪枝、
   质量反思、研报合成），生成结构严密的仿真数据，支持离线演示、自动化单元测试与应急保活。
"""

import json
import logging
from typing import Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from core.config import settings
from core.observability import degradation_tracker

logger = logging.getLogger("LLM-Factory")


class OfflineSimulationChatModel(ChatOpenAI):
    """
    离线高保真仿真模型适配器 (Offline Simulation Adapter)。
    
    【设计背景与考量】：
    在企业生产环境或学术答辩场景下，若第三方大模型 API 遭遇欠费 (HTTP 403 / Insufficient Quota)、
    服务宕机或公司内网隔离无法访问公网，传统系统会直接抛出未捕获异常导致长程任务中断。
    本适配器继承自 `ChatOpenAI`，重写底层 `_generate` 逻辑，能够解析 Prompt 的上下文语义与
    目标任务指令，动态生成符合业务 Schema 的确定性 JSON 或学术 Markdown 研报。
    """
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any
    ) -> ChatResult:
        """
        重写 LangChain 的核心消息生成方法，根据输入 Prompt 判定调用意图并返回结构化仿真响应。
        
        Args:
            messages: 输入给大模型的历史与当前对话消息列表 (包含 SystemMessage, HumanMessage)
            stop: 停止词列表
            run_manager: LangChain 运行时回调追踪管理器
            
        Returns:
            ChatResult: 封装了 AIMessage 内容的标准 LangChain 响应对象
        """
        # 拼接全部输入消息文本，便于提取课题意图与判断当前节点类型
        full_prompt = " ".join([m.content for m in messages if isinstance(m.content, str)])
        
        # -------------------------------------------------------------------------
        # 步骤 1：动态提取用户实际输入的课题名称 (Topic Extraction)
        # -------------------------------------------------------------------------
        topic_name = "深度研究课题"
        if "【研究总课题】:" in full_prompt:
            try:
                raw_part = full_prompt.split("【研究总课题】:")[1].split("【")[0].strip()
                topic_name = raw_part.split("\n")[0].strip() or raw_part
            except Exception:
                pass
        elif "【总研究课题】:" in full_prompt:
            try:
                raw_part = full_prompt.split("【总研究课题】:")[1].split("【")[0].strip()
                topic_name = raw_part.split("\n")[0].strip() or raw_part
            except Exception:
                pass
                
        # -------------------------------------------------------------------------
        # 步骤 2：意图路由匹配与高保真数据生成 (Intent-Based Mock Generation)
        # -------------------------------------------------------------------------
        
        # 场景 A：最终研报长文合成 (Synthesizer 阶段)
        # 注意：优先匹配合成场景，防止 Prompt 中携带的历史事实笔记关键词误命中其他分支
        if "顶级科技报告主编" in full_prompt or "最终研报" in full_prompt or "SYNTHESIZER_PROMPT" in full_prompt or "综合章节" in full_prompt or "长篇深度研报" in full_prompt:
            content_str = (
                f"# {topic_name} 产业全景与商业化深度研究报告\n\n"
                f"## 摘要 (Executive Summary)\n"
                f"本报告针对【{topic_name}】展开了深度全景调研。通过多维度信息聚合与事实溯源，"
                f"系统剖析了当前市场格局演进、商业变现形态、主流厂商落地实践以及未来核心突破点。\n\n"
                f"## 一、 {topic_name} 核心现状与格局演进\n"
                f"调研表明，当前 {topic_name} 正步入从粗放增长向精细化深耕的转型阶段 [1]。"
                f"头部玩家在技术驱动、工业化研发管线与出海全球化布局方面建立起坚固壁垒 [2]。\n\n"
                f"## 二、 {topic_name} 典型落地实践与代表案例\n"
                f"主流厂商普遍采用精品化与长线运营策略，在跨端生态与沉浸式交互上持续加码 [3]。"
                f"依托自动化工具链与数据飞轮，显著提升了研发效能与内容迭代效率 [4]。\n\n"
                f"## 三、 {topic_name} 核心痛点与未来趋势展望\n"
                f"当前面临的核心挑战集中在买量获客成本高企与同质化竞争加剧。前瞻性布局表明，"
                f"技术赋能管线降本增效、全球多地域合规化运营将是未来两年的核心制胜抓手 [1][3]。\n\n"
                f"## 参考文献\n"
                f"- [1] 全球产业与消费趋势权威洞察报告 (2025-2026)\n"
                f"- [2] 行业龙头年度战略财报与研发白皮书 (2025)\n"
                f"- [3] 商业化变现与用户增长关键基准度量 (2026)\n"
                f"- [4] 创新工程实践与下一代技术演进白皮书 (2026)\n"
            )

        # 场景 B0：反思回流后的定向补充规划（必须在通用大纲分支之前匹配，补充 Prompt 同样含 sections）
        elif "REFINE" in full_prompt or "质检反馈的盲区" in full_prompt or "已经完成的章节标题" in full_prompt:
            mock_refine = {
                "sections": [
                    {
                        "section_id": "pending",
                        "title": "补充：首轮盲区的关键维度分析",
                        "description": "针对质检反馈指出的覆盖缺口进行定向补充论证",
                        "research_questions": ["首轮遗漏的核心论据与数据有哪些？", "是否存在需要交叉验证的结论？"]
                    }
                ]
            }
            content_str = json.dumps(mock_refine, ensure_ascii=False)

        # 场景 D：状态剪枝提纯。
        # 注意：必须排在反思/大纲分支之前，且只用指令特征词匹配——
        # 被抓取的网页正文里可能出现"反思""大纲"等普通词汇，不能用正文内容做路由。
        elif "key_facts" in full_prompt or "事实核查与信息压缩专家" in full_prompt or "原始正文内容" in full_prompt:
            mock_facts = {
                "key_facts": [
                    "采用分层多智能体（Supervisor-Worker）拓扑，将长任务拆分为独立子上下文并发执行。",
                    "通过标准化工具协议将工具执行层与编排层解耦，支持独立进程部署与统一 Schema 描述。",
                    "状态剪枝机制将网页正文提纯为少量可溯源事实卡片，显著降低进入全局状态的文本量。"
                ]
            }
            content_str = json.dumps(mock_facts, ensure_ascii=False)

        # 场景 B：Supervisor 大纲拆解（用 JSON Schema 字段名等指令特征词匹配，避免正文误命中）
        elif "section_id" in full_prompt or "summary_brief" in full_prompt or "Outline 大纲生成" in full_prompt:
            mock_data = {
                "topic": topic_name,
                "summary_brief": f"深入剖析【{topic_name}】的核心产业生态、商业变现路径、主流玩家布局与关键发展趋势。",
                "sections": [
                    {
                        "section_id": "sec_1",
                        "title": f"一、 {topic_name} 核心现状与格局演进",
                        "description": f"探讨 {topic_name} 的发展现状、核心商业模式与主流玩家市场份额。",
                        "research_questions": [f"{topic_name} 当前的核心市场规模如何？", "核心技术演进与商业模式有哪些变化？"],
                        "content": None,
                        "citations": [],
                        "is_completed": False
                    },
                    {
                        "section_id": "sec_2",
                        "title": f"二、 {topic_name} 典型落地实践与代表案例",
                        "description": f"梳理当前头部企业在 {topic_name} 领域的标杆实践与创新产品形态。",
                        "research_questions": ["主流头部厂商采用的核心破局策略是什么？", "用户留存与商业转化表现如何？"],
                        "content": None,
                        "citations": [],
                        "is_completed": False
                    },
                    {
                        "section_id": "sec_3",
                        "title": f"三、 {topic_name} 核心痛点与未来趋势展望",
                        "description": f"剖析在获客成本、内容合规与研发投入上的主要挑战，展望未来 2~3 年演进方向。",
                        "research_questions": ["目前制约进一步突破的核心瓶颈是什么？", "未来一至两年的关键增长点在哪里？"],
                        "content": None,
                        "citations": [],
                        "is_completed": False
                    }
                ]
            }
            content_str = json.dumps(mock_data, ensure_ascii=False)

        # 场景 C：Reflector 反思质检（用 JSON 字段名与人设指令词匹配，避免正文误命中）
        elif "is_sufficient" in full_prompt or "研究质量评审专家" in full_prompt:
            mock_eval = {
                "is_sufficient": True,
                "feedback_notes": ["各子章节事实论据详实", "关键工程维度已覆盖完整"],
                "missing_angles": []
            }
            content_str = json.dumps(mock_eval, ensure_ascii=False)
            
        # 场景 E：备选研报长文匹配 (保底关键词)
        elif "最终研报" in full_prompt or "合成" in full_prompt or "顶级科技报告主编" in full_prompt:
            content_str = (
                "# 大模型长程自主智能体(Long-Horizon Agent)工程化落地与挑战深度研究报告\n\n"
                "## 摘要 (Executive Summary)\n"
                "长程自主智能体（Long-Horizon Autonomous Agent）代表了大模型从简单问答交互向复杂系统自动化落地的关键跃迁。"
                "本报告深入调研了分层编排拓扑、标准工具协议（MCP）集成与状态维护机制，系统梳理了产业实践案例与核心瓶颈。\n\n"
                "## 一、 核心概念与技术架构演进\n"
                "传统智能体多基于单步 ReAct 或扁平单智能体循环，面对长跨度任务极易出现**注意力稀释**与**目标漂移** [1]。"
                "现代架构普遍演进为 **Supervisor-Worker 分层状态机（Hierarchical Graph）**，由规划总监拆解大纲，"
                "通过动态 Send 机制将子任务并行派发给多个 Worker 探针，大幅缩短端到端时延 [2]。\n\n"
                "## 二、 行业落地实践与典型代表案例\n"
                "当前工业界典型落地形态涵盖深度研报自动生成、代码仓库端到端改造与浏览器智能自动化等场景 [3]。"
                "以领先框架为例，普遍采用 **Model Context Protocol (FastMCP)** 规范，将外部搜索、网页深度抽取与数据仓库"
                "封装为标准化工具微服务，使 Agent 具备跨系统可移植性与细粒度权限管控能力 [4]。\n\n"
                "## 三、 落地核心瓶颈与未来趋势展望\n"
                "工程实践中的核心痛点在于**长程执行的成本与幻觉累积**。未经裁剪的原始网页会导致上下文急剧膨胀。"
                "前沿系统通过引入**状态剪枝（State Pruning）**与**反思熔断（Circuit Breaker）**机制，"
                "Worker 仅向全局状态回传提纯的事实卡片（Evidence Notes），将 Token 消耗降低 65% 并杜绝死循环 [1][2]。\n\n"
                "## 参考文献\n"
                "- [1] Anthropic / LangChain Research: Long-Horizon Agent Engineering (2025)\n"
                "- [2] OpenAI Deep Research Architecture & Technical Report (2025)\n"
                "- [3] Model Context Protocol (MCP) Specification (2025-2026)\n"
                "- [4] Enterprise Multi-Agent Systems Reliability Benchmark (2026)\n"
            )
        else:
            # 场景 F：Worker 子章节单章撰写正文保底
            content_str = (
                "经过全面调研与事实核验，本章节重点阐述了长程任务编排的核心突破与落地细节：\n\n"
                "1. **分层状态机隔离**：通过将总课题拆分为子章节，各 Worker 在独立子上下文中执行调研，有效避免了 Token 上下文污染 [1]。\n"
                "2. **MCP 工具解耦**：采用标准化 FastMCP 工具服务，统一了模型对外部环境的感知与执行接口 [2]。\n"
                "3. **反思质检闭环**：通过反思节点实时评审信息覆盖度，确保最终研报的权威性与严密性 [3]。"
            )
            
        generation = ChatGeneration(message=AIMessage(content=content_str))
        return ChatResult(generations=[generation])


class ResilientFailoverChatModel(ChatOpenAI):
    """
    具备主动容灾 (Failover) 能力的大模型包装类。
    
    【核心机制】：
    1. 正常执行路径：继承 `ChatOpenAI` 原生网络驱动，优先向配置的目标 API 端点发起请求。
    2. 超时与重试策略：设置 timeout=90.0 秒与 max_retries=2，给多步推理和超长研报合成留足生成时间。
    3. 异常捕获与判定：若捕获到 HTTP 403 (欠费停机)、AllocationQuota 超额或网络拒绝连接异常，
       立即记录详细告警日志，并在零等待状态下触发 `OfflineSimulationChatModel` 降级，保障业务流程平稳跑完。
    """
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any
    ) -> ChatResult:
        """
        带异常降级保护的消息生成入口。
        """
        try:
            # 尝试调用真实大模型端点
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as e:
            err_msg = str(e)
            logger.error(f"[LLM-Error] 真实大模型调用失败详情: {err_msg}")

            # 针对不同失败原因进行针对性日志诊断
            if "403" in err_msg or "quota" in err_msg.lower() or "AllocationQuota" in err_msg:
                logger.warning("[LLM-Failover] API 额度耗尽或 403 权限受限，触发自动容灾降级")
            else:
                logger.warning(f"[LLM-Failover] API 网络异常或不可达 ({err_msg[:80]}...)，触发自动容灾降级")

            # 显式登记降级事件：该节点输出为离线仿真内容，不得与真实产物混淆
            degradation_tracker.record(
                f"LLM 节点调用失败（模型={getattr(self, 'model_name', 'unknown')}，"
                f"原因={err_msg[:100]}），已降级为离线仿真输出，相关内容不代表真实检索结果"
            )

            # 实例化离线高保真适配器完成本次会话的生成工作
            sim_model = OfflineSimulationChatModel(api_key="mock", base_url="mock")
            return sim_model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def get_chat_model(model_name: Optional[str] = None, temperature: float = 0.2) -> ChatOpenAI:
    """
    大模型工厂函数：根据指定模型名和采样温度构造高可用 LLM 客户端实例。
    
    Args:
        model_name: 目标大语言模型名称 (如 qwen-plus, qwen-max, deepseek-chat)；若未指定则使用 settings.WORKER_MODEL
        temperature: 采样温度。学术研报与事实抽取推荐使用低温度 (0.1 ~ 0.3) 以保证逻辑收敛与事实准确性
        
    Returns:
        具备主动容灾机制的 ChatOpenAI 兼容实例
    """
    target_model = model_name or settings.WORKER_MODEL
    
    llm = ResilientFailoverChatModel(
        model=target_model,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=temperature,
        timeout=90.0,       # 设定 90 秒网络请求超时上限，包容长程推理延迟
        max_retries=2       # 允许底层重试 2 次网络临时抖动
    )
    
    return llm
