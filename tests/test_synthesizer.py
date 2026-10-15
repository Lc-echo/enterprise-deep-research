"""Synthesizer 引文双保险、信源去重与无信源降级声明测试。"""

from types import SimpleNamespace

import pytest

from agent.state import Section, EvidenceNote
from agent import synthesizer as synth_mod
from core.observability import degradation_tracker


class FakeLLM:
    """返回固定 Markdown 内容的最小 LLM 替身。"""

    def __init__(self, content: str):
        self._content = content

    def invoke(self, messages):
        return SimpleNamespace(content=self._content)


@pytest.fixture
def patch_llm(monkeypatch):
    def _patch(content: str):
        monkeypatch.setattr(synth_mod, "get_chat_model", lambda *a, **k: FakeLLM(content))
    return _patch


def _section(section_id="sec_1", url=None, title="一、测试章节"):
    citations = [f"[测试文献]({url})"] if url else []
    return Section(
        section_id=section_id,
        title=title,
        description="测试描述",
        research_questions=["问题1？"],
        content="本章节基于真实信源展开论述。",
        citations=citations,
        is_completed=True,
    )


def test_double_insurance_restores_real_urls_when_model_drops_them(patch_llm):
    """模型输出的参考文献缺少 URL 时，必须用真实信源清单程序化替换。"""
    real_url = "https://arxiv.org/abs/2401.12345"
    patch_llm(
        "# 测试报告\n\n## 一、测试章节\n\n正文内容 [1]。\n\n"
        "## 参考文献\n\n- [1] 某篇论文标题（无链接）"
    )
    state = {
        "topic": "pytest_restore_urls",
        "summary_brief": "测试摘要",
        "completed_sections": [_section(url=real_url)],
        "global_notes": [EvidenceNote(title="某篇论文标题", url=real_url, key_facts=["事实一"], relevance_score=0.9)],
    }

    out = synth_mod.synthesizer_node(state)

    report = out["final_report"]
    assert real_url in report
    assert out["total_sources_count"] == 1
    assert out["degradation_events"] == []


def test_no_real_sources_replaces_references_with_notice(patch_llm):
    """无任何真实信源时，模型编造的参考文献必须被替换为降级声明。"""
    fake_url = "https://research.benchmark.org/sec_1"
    patch_llm(
        "# 测试报告\n\n## 一、测试章节\n\n正文。\n\n"
        "## 参考文献\n\n- [1] 全球产业与消费趋势权威洞察报告 (2025)\n- [2] 虚构的白皮书\n"
        f"- [3]({fake_url})"
    )
    state = {
        "topic": "pytest_no_sources",
        "summary_brief": "测试摘要",
        "completed_sections": [_section(url=fake_url)],
        "global_notes": [EvidenceNote(title="伪造卡片", url=fake_url, key_facts=["编造事实"], relevance_score=0.9)],
    }

    out = synth_mod.synthesizer_node(state)

    report = out["final_report"]
    assert "research.benchmark.org" not in report
    assert "降级声明" in report
    assert out["total_sources_count"] == 0
    assert out["degradation_events"]  # 至少有一条降级事件
    assert degradation_tracker.degraded is True


def test_degradation_banner_inserted_at_top(patch_llm):
    """全链路存在降级事件时，报告顶部必须插入降级横幅。"""
    patch_llm("# 测试报告\n\n## 一、测试章节\n\n无信源正文。\n\n## 参考文献\n\n（留空）")
    degradation_tracker.record("测试用例手动注入的降级事件")
    state = {
        "topic": "pytest_banner",
        "summary_brief": "",
        "completed_sections": [_section(url=None)],
        "global_notes": [],
    }

    out = synth_mod.synthesizer_node(state)

    report = out["final_report"]
    assert report.index("降级模式提示") < report.index("测试报告")
    assert "测试用例手动注入的降级事件" in report
