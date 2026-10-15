"""
MCP 同步客户端桥接器 (MCP Synchronous Client Bridge)
=====================================================
将 `mcp_servers/search_tools.py` 作为**独立子进程**（MCP stdio server）拉起，
Worker 节点的同步函数调用经本桥接器转为标准 MCP 协议请求，
从而实现工具执行层与 Agent 编排层的真实进程隔离（工具崩溃不影响主进程）。

【运行机制】：
1. 守护线程持有独立 asyncio event loop，MCP Client 在该 loop 中保持长连接；
2. 懒启动：首次工具调用时才拉起子进程，整个进程生命周期复用；
3. 同步包装：LangGraph 的 Worker 是同步函数，通过 run_coroutine_threadsafe 跨线程提交；
4. 容灾降级：子进程拉起失败或协议调用异常时，自动降级为进程内直接调用（direct），
   并登记降级事件，保证流水线不中断；
5. 传输方式由配置 MCP_TRANSPORT 控制（stdio 默认；direct 仅供离线测试/调试）。
"""

import sys
import atexit
import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, List, Dict

from core.config import settings
from core.observability import degradation_tracker

logger = logging.getLogger("MCP-Client")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _unwrap_result(structured_content: Any) -> Any:
    """
    FastMCP 对工具返回值做了 {"result": <return value>} 包装，
    这里解包还原为工具函数的原始返回值。
    """
    if isinstance(structured_content, dict) and set(structured_content.keys()) == {"result"}:
        return structured_content["result"]
    return structured_content


class MCPBridge:
    """MCP stdio 长连接的同步桥接器（单例使用）。"""

    def __init__(self) -> None:
        self._mode: str = (settings.MCP_TRANSPORT or "stdio").lower()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._client = None
        self._start_lock = threading.Lock()
        self._started = False

    # ------------------------- 异步侧（运行于独立 loop 线程） -------------------------

    def _start_loop_thread(self) -> None:
        self._loop = asyncio.new_event_loop()

        def _run_loop() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=_run_loop, daemon=True, name="mcp-stdio-loop")
        self._loop_thread.start()

    async def _connect(self) -> None:
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport

        import os
        env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        transport = StdioTransport(
            command=sys.executable,
            args=["-m", "mcp_servers.search_tools"],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        self._client = Client(transport)
        await self._client.__aenter__()
        tools = await self._client.list_tools()
        logger.info(f"[MCP-Client] stdio 连接已建立，可用工具: {[t.name for t in tools]}")

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        result = await self._client.call_tool(tool_name, arguments)
        return _unwrap_result(result.structured_content)

    async def _close(self) -> None:
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass

    # ------------------------- 同步侧（供 Worker 直接调用） -------------------------

    def _run_async(self, coro, timeout: float) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _ensure_started(self) -> None:
        if self._mode == "direct":
            return
        if self._started:
            return
        with self._start_lock:
            if self._started:
                return
            try:
                self._start_loop_thread()
                self._run_async(self._connect(), timeout=30.0)
                self._started = True
            except Exception as e:
                logger.warning(f"[MCP-Client] stdio 子进程连接失败，降级为进程内直接调用: {e}")
                degradation_tracker.record(
                    f"MCP stdio 子进程连接失败（{str(e)[:100]}），工具层降级为进程内直接调用"
                )
                self._mode = "direct"
                self._started = False

    def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: float = 90.0) -> Any:
        """
        同步调用 MCP 工具；stdio 模式发生任何故障时自动降级到进程内直接调用。
        """
        if self._mode == "stdio":
            self._ensure_started()
        if self._mode == "stdio" and self._started:
            try:
                return self._run_async(self._call_tool(tool_name, arguments), timeout=timeout)
            except Exception as e:
                logger.warning(f"[MCP-Client] 工具 {tool_name} 协议调用失败，降级进程内直接调用: {e}")
                degradation_tracker.record(
                    f"MCP 工具 {tool_name} 调用失败（{str(e)[:100]}），本次降级为进程内直接调用"
                )
                self._mode = "direct"
        # direct 模式：进程内直接导入函数执行（离线测试 / 兜底路径）
        from mcp_servers import search_tools as direct
        return getattr(direct, tool_name)(**arguments)

    def shutdown(self) -> None:
        """进程退出时回收子进程与事件循环（尽力而为）。"""
        if self._client is not None and self._loop is not None:
            try:
                self._run_async(self._close(), timeout=10.0)
            except Exception:
                pass


# 全局单例
_bridge = MCPBridge()

# 进程退出时尽力回收 MCP 子进程，避免 stdio 子进程残留
atexit.register(_bridge.shutdown)


def web_search(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """经 MCP 协议（或降级直连）调用 web_search 工具。"""
    return _bridge.call_tool("web_search", {"query": query, "max_results": max_results}, timeout=90.0)


def fetch_web_content(url: str, max_length: int = 4000) -> str:
    """经 MCP 协议（或降级直连）调用 fetch_web_content 工具。"""
    return _bridge.call_tool(
        "fetch_web_content", {"url": url, "max_length": max_length}, timeout=30.0
    )
