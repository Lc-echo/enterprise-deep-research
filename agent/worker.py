"""
Worker 独立调研节点 (Parallel Research Worker Node)
======================================================
在分层多智能体拓扑中充当并发探针执行者。

【核心技术特性与执行管线】：
1. 上下文隔离 (Context Isolation)：
   每个 Worker 仅接收专属的 `WorkerInput`，在独立的提示词上下文中执行，互不干扰；
2. 检索词智能提纯 (Query Refinement)：
   大模型规划的自然语言问题通常过于冗长（例如“...是如何协同工作的？”），通用搜索引擎在长句下会发生分词漂移。
   提纯算法会过滤口语助词，并主动注入“架构/白皮书/报告”等行业权威特征词；
3. 状态剪枝 (State Pruning)：
   调用 FastMCP 获取网页正文后，坚决不把数千字的原始 HTML/纯文本塞入全局状态，
   而是调用轻量模型将其提纯为 2~4 条高信噪比的核心事实证据卡片 (EvidenceNote)，压缩比达 85% 以上；
4. Map-Reduce 自动归约：
   Worker 节点完成章节撰写后，返回 `{"completed_sections": [completed_section], "global_notes": collected_notes}`，
   由 LangGraph 状态机底层的 `operator.add` 自动累加聚合至全局状态。
"""

import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm import get_chat_model
from core.config import settings
from core.observability import degradation_tracker, is_simulated_url
from core.parsing import structured_invoke
from agent.state import WorkerInput, EvidenceNote, Section, PrunedFacts
from agent.prompts import WORKER_PROMPT, STATE_PRUNING_PROMPT
# 默认经 MCP stdio 独立子进程调用工具；失败时 mcp_client 内部自动降级为进程内直连
from mcp_servers.mcp_client import web_search, fetch_web_content

logger = logging.getLogger("Agent-Worker")


def _refine_search_query(topic: str, section_title: str, raw_question: str) -> str:
    """
    检索词智能提纯器 (Search Query Refiner)。
    
    【算法考量】：
    将冗长的自然语言问答句转换为紧凑的高权重关键词组合。
    1. 剥离口语助词、疑问代词与标点符号（如“是什么”、“有哪些”、“深入分析”）；
    2. 提取课题名称与章节核心主题名；
    3. 主动追加“架构”、“报告”、“白皮书”等高信噪比锚定词，迫使搜索引擎置顶权威文献与深度分析。
    
    Args:
        topic: 研究总课题名称
        section_title: 当前章节标题
        raw_question: 规划大纲中的原始子研究问题
        
    Returns:
        str: 提纯后的高权重搜索引擎 Query
    """
    noise_tokens = [
        "是如何协同工作的？", "是如何工作的？", "是什么？", "有哪些？", "怎么做？",
        "目前在实际生产中的表现如何？", "有什么影响？", "如何实现？", "？", "?",
        "（", "）", "(", ")", "探讨", "分析", "剖析", "研究", "深入"
    ]
    clean_q = raw_question
    for token in noise_tokens:
        clean_q = clean_q.replace(token, " ")
    clean_q = " ".join(clean_q.split())
    
    clean_topic = topic.split("(")[0].strip()
    clean_sec = section_title.split("、")[-1].strip()
    
    return f"{clean_topic} {clean_sec} {clean_q} 架构 报告 白皮书".strip()


def research_worker_node(state_input: WorkerInput) -> Dict[str, Any]:
    """
    单个并发 Worker 节点的端到端执行流水线 (Worker Execution Pipeline)。
    
    【执行步骤】：
    步骤 1：遍历该章节的核心研究子问题，经 Query 提纯后调用 FastMCP `web_search` 工具；
    步骤 2：对候选网页调用 FastMCP `fetch_web_content` 进行正文清洗，校验正文字符数与有效性；
    步骤 3：执行状态剪枝 (`_prune_facts`)，抽取结构化事实证据并记录可溯源 Markdown 链接；
    步骤 4：基于事实卡片调用大模型撰写本章节专业论述正文 (`_draft_section`)；
    步骤 5：打包生成 Section 对象并返回 Map-Reduce 归约字典。
    
    Args:
        state_input: 由 Send API 派发的独立子状态字典
        
    Returns:
        Dict[str, Any]: 待聚合的 completed_sections 和 global_notes
    """
    section = state_input["section"]
    topic = state_input["topic"]
    
    logger.info(f"[Worker-{section.section_id}] 启动子章节调研: 《{section.title}》")
    
    collected_notes: List[EvidenceNote] = []
    citations: List[str] = []
    
    # -------------------------------------------------------------------------
    # 1. 针对该章节的子问题进行定向权威检索与抓取
    # -------------------------------------------------------------------------
    search_queries = section.research_questions if section.research_questions else [section.title]
    
    # 控制信源探索广度：每个章节重点探究前 2 个核心子问题，防止并发请求泛滥
    for query in search_queries[:2]:
        refined_query = _refine_search_query(topic, section.title, query)
        search_results = web_search(refined_query, max_results=settings.SEARCH_RESULTS_PER_QUERY)

        valid_items_count = 0
        if search_results:
            for item in search_results:
                title = item.get("title", "信源参考")
                url = item.get("url", "")
                snippet = item.get("snippet", "")

                # 防御性过滤：任何仿真 / 占位 URL 都不得进入调研链路
                if is_simulated_url(url):
                    continue

                # 尝试抓取网页正文清洗（限制正文上限为 2500 字符）
                raw_text = fetch_web_content(url, max_length=2500) if url else ""

                # 内容有效性与信噪比过滤：
                # 若正文抓取受阻（如 403 防爬阻断），退回使用搜索摘要 snippet；若两者皆空则舍弃
                if not raw_text or len(raw_text.strip()) < 80:
                    if len(snippet.strip()) >= 40:
                        raw_text = snippet
                    else:
                        continue  # 丢弃死链与无效空白网页

                # 状态剪枝：将大段文本提纯为高信噪比事实卡片
                note = _prune_facts(section.title, title, url, raw_text)
                collected_notes.append(note)
                citations.append(f"[{title}]({url})")
                valid_items_count += 1

                # 每个问题吸纳至多 2 篇高价值权威信源
                if valid_items_count >= 2:
                    break

    # 整章未获取任何真实信源：显式登记降级，不伪造文献卡片
    if not collected_notes:
        logger.warning(f"[Worker-{section.section_id}] 未获取到任何有效信源，章节将基于模型先验知识撰写")
        degradation_tracker.record(
            f"章节《{section.title}》未检索到任何可用真实信源，"
            f"该章节内容基于模型先验知识生成，不附带可验证引用"
        )
            
    # -------------------------------------------------------------------------
    # 2. 撰写本章节专业 Markdown 正文
    # -------------------------------------------------------------------------
    section_content = _draft_section(topic, section, collected_notes)
    
    # 封装已完成章节成果对象
    completed_section = Section(
        section_id=section.section_id,
        title=section.title,
        description=section.description,
        research_questions=section.research_questions,
        content=section_content,
        citations=citations,
        is_completed=True
    )
    
    logger.info(f"[Worker-{section.section_id}] 章节撰写完成，产出 {len(collected_notes)} 条事实卡片")
    
    # -------------------------------------------------------------------------
    # 3. 返回 Map-Reduce 归约载荷
    # -------------------------------------------------------------------------
    # 由 LangGraph 内部的 operator.add 自动将数组合并至全局 State 的 completed_sections
    return {
        "completed_sections": [completed_section],
        "global_notes": collected_notes
    }


def _prune_facts(section_title: str, source_title: str, url: str, raw_content: str) -> EvidenceNote:
    """
    状态剪枝提纯器 (State Pruning Fact Extractor)。
    
    利用低温度 (temperature=0.1) 的轻量模型对长文本执行极限定向提炼，
    只保留能回答当前章节核心问题的 2~4 条确定性事实结论。
    
    Args:
        section_title: 当前章节标题
        source_title: 信源标题
        url: 信源链接
        raw_content: 抓取到的原始文本片段
        
    Returns:
        EvidenceNote: 包含纯净事实论据的卡片模型实例
    """
    prompt = STATE_PRUNING_PROMPT.format(
        section_title=section_title,
        source_title=source_title,
        raw_content=raw_content[:1500]
    )
    
    try:
        llm = get_chat_model(settings.WORKER_MODEL, temperature=0.1)
        pruned = structured_invoke(llm, [HumanMessage(content=prompt)], PrunedFacts)
        facts = pruned.key_facts or [raw_content[:100]]
    except Exception as e:
        # 解析异常时不做事实转述（避免编造），直接截取原文片段作为保底证据
        logger.warning(f"[Worker] 事实剪枝解析失败 ({e})，回退为原文片段")
        degradation_tracker.record(
            f"信源《{source_title}》事实剪枝输出解析失败，已回退为原文摘录"
        )
        facts = [f"[原文摘录，未经模型提纯] {raw_content[:120]}"]
        
    return EvidenceNote(
        title=source_title,
        url=url,
        key_facts=facts,
        relevance_score=0.9
    )


def _draft_section(topic: str, section: Section, notes: List[EvidenceNote]) -> str:
    """
    子章节正文深度合成器 (Section Draft Synthesizer)。
    
    将提纯后的所有事实卡片规整为带有 Markdown 引文标记的 Prompt 证据列表，
    引导模型基于客观证据撰写严密的分析正文。
    
    Args:
        topic: 研究总课题
        section: 当前章节定义
        notes: 本章节收集到的事实证据卡片列表
        
    Returns:
        str: 撰写完成的章节正文 Markdown 字符串
    """
    # 将事实卡片转化为规范格式的证据列表
    evidence_text = "\n".join([
        f"- 来源 [{i+1}] [{n.title}]({n.url}): " + "; ".join(n.key_facts)
        for i, n in enumerate(notes)
    ])
    
    prompt = WORKER_PROMPT.format(
        topic=topic,
        section_title=section.title,
        section_description=section.description,
        evidence_notes=evidence_text if evidence_text else (
            "（本次未检索到任何外部信源。只能基于你已有的、置信度较高的通用知识进行框架性论述，"
            "必须明确标注'缺乏实时信源佐证'，严禁输出 [n] 角标、严禁编造文献名称、数据与链接。）"
        )
    )
    
    try:
        llm = get_chat_model(settings.WORKER_MODEL, temperature=0.3)
        res = llm.invoke([
            SystemMessage(content="你是一名客观专业的行业研报作者，行文逻辑清晰。"),
            HumanMessage(content=prompt)
        ])
        return res.content.strip()
    except Exception as e:
        # 大模型调用失败时的降级保底章节正文：明确声明无引用，不输出伪造角标
        logger.warning(f"[Worker] 章节撰写失败 ({e})，输出无引用保底文本")
        degradation_tracker.record(
            f"章节《{section.title}》正文撰写模型调用失败，已使用无引用保底文本"
        )
        return (
            f"### {section.title}\n\n"
            f"本章节针对 {section.description} 进行分析。"
            "由于正文生成环节模型调用失败，且未获取到可核验的事实证据，"
            "此处不提供具体数据与引文结论。\n\n"
            "> **降级提示**：本章节内容缺失，请在联网环境重新运行以获取完整调研结果。"
        )

