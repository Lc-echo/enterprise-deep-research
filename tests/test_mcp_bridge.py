"""
MCP 桥接器测试：返回值解包、stdio 连接失败时降级进程内直连、降级事件登记。
"""

import pytest

from mcp_servers import mcp_client
from mcp_servers import search_tools
from core.observability import degradation_tracker


def test_unwrap_result():
    assert mcp_client._unwrap_result({"result": [1, 2, 3]}) == [1, 2, 3]
    assert mcp_client._unwrap_result({"result": "ok"}) == "ok"
    assert mcp_client._unwrap_result("plain") == "plain"
    assert mcp_client._unwrap_result({"a": 1}) == {"a": 1}


def test_bridge_falls_back_to_direct_on_connect_failure(monkeypatch):
    bridge = mcp_client.MCPBridge()
    bridge._mode = "stdio"
    bridge._started = False

    def _boom(coro, timeout):
        coro.close()  # 未执行的协程显式关闭，避免 RuntimeWarning
        raise RuntimeError("stdio connect failed")

    monkeypatch.setattr(bridge, "_start_loop_thread", lambda: None)
    monkeypatch.setattr(bridge, "_run_async", _boom)
    # direct 兜底目标替换为假实现，避免真实联网
    monkeypatch.setattr(
        search_tools, "web_search",
        lambda query, max_results=4: [{"title": "direct", "url": "https://example.com", "snippet": "x" * 40}],
    )

    out = bridge.call_tool("web_search", {"query": "q", "max_results": 1})
    assert out[0]["url"] == "https://example.com"
    assert bridge._mode == "direct"
    events = degradation_tracker.events
    assert any("stdio 子进程连接失败" in e for e in events)


def test_bridge_protocol_error_falls_back(monkeypatch):
    bridge = mcp_client.MCPBridge()
    bridge._mode = "stdio"
    bridge._started = True
    bridge._loop = object()  # 让 _ensure_started 视为已启动

    def _boom(coro, timeout):
        coro.close()
        raise RuntimeError("protocol broken")

    monkeypatch.setattr(bridge, "_run_async", _boom)
    monkeypatch.setattr(
        search_tools, "fetch_web_content",
        lambda url, max_length=4000: "fallback body",
    )

    out = bridge.call_tool("fetch_web_content", {"url": "https://example.com", "max_length": 100})
    assert out == "fallback body"
    assert bridge._mode == "direct"
