"""
Synthesizer 研报合成与双保险终审节点 (Report Synthesizer Node)
================================================================
在多智能体系统 Map-Reduce 拓扑中充当最后的 Reduce 汇聚归约器与“科技报告主编”。

【核心技术特性与创新亮点】：
1. 并发无序结果归约 (Concurrent Order Alignment)：
   由于各个 Worker 节点在并发网络 I/O 耗时不同，完成上报的顺序是随机无序的。
   Synthesizer 首先依据 `section_id`（如 sec_1, sec_2...）进行去重与拓扑升序重排，
   确保最终研报章节的递进逻辑完全符合 Supervisor 顶层规划；
2. 全局信源账本聚合与去重 (Citation Ledger Aggregation)：
   从全局事实卡片 (global_notes) 与章节引用列表 (citations) 中抽取真实 URL，
   生成全局规范编号的统一参考文献索引库；
3. 程序化“双保险”引文超链接置换机制 (Double-Insurance Fallback)：
   大模型在生成超长研报时，偶发性偷懒会将 URL 丢弃，退化为中文书名号。
   本节点引入后处理确定性校验：若正则检测到文末参考文献部分缺少 `http` 协议链接，
   程序会毫秒级无缝切入，用 FastMCP 验证过的真实超链接列表实施精准置换，确保 100% 可验证溯源；
4. 研报文件自动持久化：自动在 `reports/` 目录落盘存储带有安全文件名的 Markdown 报告。
"""

import os
import re
import logging
from typing import Dict, Any, List
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm import get_chat_model
from core.config import settings
from core.observability import degradation_tracker, is_simulated_url
from agent.state import ResearchState, Section
from agent.prompts import SYNTHESIZER_PROMPT

logger = logging.getLogger("Agent-Synthesizer")

# 无真实信源时统一使用的文末声明（禁止伪造参考文献）
NO_SOURCE_NOTICE = (
    "## 参考文献与信源链接 (References)\n\n"
    "> **降级声明**：本次运行未获取到任何可验证的外部信源（搜索引擎不可达、"
    "网页抓取失败或未配置检索 API Key）。本报告内容完全由模型先验知识生成，"
    "不包含可溯源的事实引用，不得作为决策依据。请在联网环境重新运行以获取带真实溯源的研报。"
)


def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    """
    终审与长文研报合成核心函数。
    
    【执行流水线】：
    1. 章节成果去重与有序重组：修复并发竞争带来的乱序问题；
    2. 全局信源库提取与 URL 去重；
    3. 驱动旗舰长文本大模型 (`SYNTHESIZER_MODEL`) 进行学术级 Markdown 篇章润色；
    4. 执行双保险超链接完整性校验与程序化自动替换；
    5. 本地磁盘自动落盘保存 (reports/Report_*.md)。
    
    Args:
        state: 全局状态机字典 (包含所有 completed_sections 与 global_notes)
        
    Returns:
        包含 final_report 与 total_sources_count 的成果字典
    """
    topic = state.get("topic", "深度研究报告")
    completed_sections = state.get("completed_sections", [])
    global_notes = state.get("global_notes", [])
    
    logger.info(f"[Synthesizer] 开始执行最终研报合成，正在合并 {len(completed_sections)} 个章节正文")
    
    # -------------------------------------------------------------------------
    # 1. 章节去重并按 section_id 严格升序重排 (解决并发 Worker 乱序完成痛点)
    # -------------------------------------------------------------------------
    unique_sections: Dict[str, Section] = {}
    for s in completed_sections:
        # 若存在同 ID 章节（如迭代轮次产生新版本），保留正文内容更详实充实的版本
        if s.section_id not in unique_sections or (s.content and len(s.content) > len(unique_sections[s.section_id].content or "")):
            unique_sections[s.section_id] = s
            
    sorted_sections: List[Section] = sorted(unique_sections.values(), key=lambda x: x.section_id)
    
    # -------------------------------------------------------------------------
    # 2. 拼接章节正文素材并聚合全局真实 URL
    # -------------------------------------------------------------------------
    sections_text_parts = []
    all_citations = []
    
    for idx, sec in enumerate(sorted_sections):
        sections_text_parts.append(f"## {sec.title}\n\n{sec.content or '（本章节内容生成中）'}\n")
        all_citations.extend(sec.citations)
        
    combined_sections_text = "\n".join(sections_text_parts)
    
    # 从 global_notes 中收集真实可溯源信源（仿真 / 占位域名一律剔除）
    unique_sources = []
    seen_urls = set()

    for note in global_notes:
        url = getattr(note, "url", "") or ""
        if url and url not in seen_urls and not is_simulated_url(url):
            seen_urls.add(url)
            clean_title = (note.title or "文献参考").strip().replace("\n", " ").strip("《》[]()")
            unique_sources.append({"title": clean_title, "url": url})

    # 补充从各个章节正文引用中利用正则捕获的 Markdown 超链接 [标题](URL)
    for sec in sorted_sections:
        for cite in sec.citations:
            m = re.search(r"\[?(.*?)\]?\((https?://[^\)]+)\)", cite)
            if m:
                t, u = m.group(1).strip().strip("《》[]()"), m.group(2).strip()
                if u not in seen_urls and not is_simulated_url(u):
                    seen_urls.add(u)
                    unique_sources.append({"title": t, "url": u})

    # 若过滤后一个真实信源都不剩，显式登记降级
    if not unique_sources:
        degradation_tracker.record(
            "最终研报未包含任何真实可溯源信源，参考文献部分已替换为降级声明"
        )
                    
    # 构建注入给大模型的标准化超链接信源清单
    references_lines = [
        f"[{i+1}] [{s['title']}]({s['url']})"
        for i, s in enumerate(unique_sources)
    ]
    if references_lines:
        references_text = "\n".join(references_lines)
    else:
        # 无真实信源时明确禁止模型编造文献（历史上模型会输出虚构的中文书名号文献）
        references_text = (
            "本次运行没有任何可用的外部信源。禁止输出任何参考文献条目、"
            "禁止编造文献名称、机构名或 URL；文末参考文献部分留空即可。"
        )

    # -------------------------------------------------------------------------
    # 3. 调用旗舰大模型进行全局行文润色与结构装配
    # -------------------------------------------------------------------------
    prompt = SYNTHESIZER_PROMPT.format(
        topic=topic,
        sections_text=combined_sections_text,
        references_text=references_text
    )

    try:
        # 使用 qwen-max 等旗舰模型进行终稿统稿
        llm = get_chat_model(settings.SYNTHESIZER_MODEL, temperature=0.2)
        res = llm.invoke([
            SystemMessage(content="你是一名顶级科技报告主编，负责最终研报的生成。"),
            HumanMessage(content=prompt)
        ])
        final_report_md = res.content.strip()
    except Exception as e:
        logger.warning(f"[Synthesizer] 旗舰模型合成失败 ({e})，使用结构化保底拼接")
        degradation_tracker.record("研报合成模型调用失败，已使用章节原文保底拼接")
        # 异常情况下的结构化保底拼接（不编造任何文献）
        final_report_md = (
            f"# {topic}：行业全景与工程落地深度研究报告\n\n"
            f"> **报告概述**：{state.get('summary_brief', '本报告对所涉课题进行了跨维度全链路剖析。')}\n\n"
            f"{combined_sections_text}\n\n"
        )

    # -------------------------------------------------------------------------
    # 4. 后处理校验（程序化"双保险"）
    #    - 有真实信源：确保文末参考文献 100% 带可点击 URL，缺失则用真实信源清单置换；
    #    - 无真实信源：模型输出的任何参考文献块（含编造书名/仿真链接）一律替换为降级声明。
    # -------------------------------------------------------------------------
    ref_headers = ["## 参考文献与信源链接", "## 参考文献", "## References", "## 参考文献与数据来源"]

    def _find_ref_header(text: str):
        for h in ref_headers:
            if h in text:
                return h
        return None

    if unique_sources:
        clean_refs_block = "## 参考文献与信源链接 (References)\n\n" + "\n".join([
            f"[{i+1}] [{s['title']}]({s['url']})  "
            for i, s in enumerate(unique_sources)
        ])

        found_header = _find_ref_header(final_report_md)
        if found_header:
            header_pos = final_report_md.rfind(found_header)
            after_header = final_report_md[header_pos:]
            # 检测文末参考文献是否遗漏真实 URL（仅统计非仿真域名的 http 链接）
            has_real_url = any(
                (("http://" in line) or ("https://" in line)) and not is_simulated_url(line)
                for line in after_header.splitlines()
            )
            if not has_real_url:
                final_report_md = final_report_md[:header_pos].rstrip() + "\n\n---\n\n" + clean_refs_block
        else:
            # 模型完全遗漏参考文献标题，直接在文末安全追加
            final_report_md = final_report_md.rstrip() + "\n\n---\n\n" + clean_refs_block
    else:
        # 无真实信源：移除模型可能编造的整个参考文献块，替换为降级声明
        found_header = _find_ref_header(final_report_md)
        if found_header:
            header_pos = final_report_md.rfind(found_header)
            final_report_md = final_report_md[:header_pos].rstrip()
        # 清除正文中悬空的数字角标 [n]（本报告无信源，不应残留任何引用标记；
        # 负向断言 (?!\() 避免误删 Markdown 链接 [文字](URL)）
        final_report_md = re.sub(r"\[\d{1,2}\](?!\()", "", final_report_md)
        final_report_md = final_report_md.rstrip() + "\n\n---\n\n" + NO_SOURCE_NOTICE

    # 若全链路存在任何降级事件，在报告最顶部插入醒目的降级横幅
    degradation_events = degradation_tracker.events
    if degradation_events:
        banner_lines = [
            "> **降级模式提示**：本次研究运行过程中存在以下降级环节，报告中相应内容的可信度受限：",
            *[f"> {i+1}. {evt}" for i, evt in enumerate(degradation_events)],
        ]
        banner = "\n".join(banner_lines) + "\n\n---\n\n"
        final_report_md = banner + final_report_md

    # -------------------------------------------------------------------------
    # 5. 自动持久化落盘至本地 reports/ 目录
    # -------------------------------------------------------------------------
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 过滤文件名中的非法字符，构建跨平台合规的文件名
    safe_topic = "".join([c for c in topic if c.isalnum() or c in (" ", "_", "-")]).strip().replace(" ", "_")
    report_file = reports_dir / f"Report_{safe_topic[:30]}.md"

    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(final_report_md)
        logger.info(f"[Synthesizer] 研报已成功持久化至文件: {report_file}")
    except Exception as e:
        logger.warning(f"[Synthesizer] 保存研报文件异常: {e}")

    logger.info(
        f"[Synthesizer] 深度研报生成完毕！总字符数: {len(final_report_md)}，"
        f"真实信源 {len(unique_sources)} 个，降级事件 {len(degradation_events)} 条"
    )

    return {
        "final_report": final_report_md,
        "total_sources_count": len(unique_sources),
        "degradation_events": degradation_events
    }

