"""
pytest 全局配置：将项目根目录加入 sys.path，并提供离线测试公共 fixtures。
所有测试默认不访问真实网络与真实 LLM（通过 monkeypatch 注入离线仿真模型与假搜索）。
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 测试一律使用进程内 direct 模式，绝不拉起 MCP stdio 子进程（测试中的工具调用均由 fixture 替身接管）
os.environ["MCP_TRANSPORT"] = "direct"

from core.llm import OfflineSimulationChatModel  # noqa: E402
from core.observability import degradation_tracker  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_degradation_tracker():
    """每个测试前后清空降级账本，避免单例状态串扰。"""
    degradation_tracker.reset()
    yield
    degradation_tracker.reset()
    # 清理测试在 reports/ 下落盘的报告，避免污染项目目录
    reports_dir = ROOT / "reports"
    if reports_dir.exists():
        for f in reports_dir.glob("Report_pytest_*.md"):
            f.unlink()


@pytest.fixture
def offline_llm(monkeypatch):
    """将四个 Agent 节点模块的 get_chat_model 全部替换为离线仿真模型。"""
    import agent.supervisor
    import agent.worker
    import agent.reflector
    import agent.synthesizer

    sim = OfflineSimulationChatModel(api_key="mock", base_url="mock")
    for module in (agent.supervisor, agent.worker, agent.reflector, agent.synthesizer):
        monkeypatch.setattr(module, "get_chat_model", lambda *a, **k: sim)
    return sim


@pytest.fixture
def fake_search(monkeypatch):
    """替换 Worker 的搜索/抓取：返回带真实域名的假结果，正文抓取为空（走 snippet 兜底）。"""
    import agent.worker as worker_mod

    long_snippet = (
        "这是一条用于单元测试的模拟搜索摘要文本，长度必须超过四十个字符，"
        "才能通过工作节点对信源内容有效性的最低校验阈值，特此补足长度。"
    )

    def _fake_search(query, max_results=4):
        return [
            {
                "title": f"学术信源：{query[:8]}",
                "url": "https://arxiv.org/abs/0000.00000",
                "snippet": long_snippet,
            },
            {
                "title": f"官方技术博客：{query[:8]}",
                "url": "https://developer.aliyun.com/article/test-001",
                "snippet": long_snippet,
            },
        ]

    monkeypatch.setattr(worker_mod, "web_search", _fake_search)
    monkeypatch.setattr(worker_mod, "fetch_web_content", lambda url, max_length=4000: "")
