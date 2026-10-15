"""
运行时降级追踪器 (Degradation Tracker)
========================================
深度研究流水线在真实运行中可能遭遇 LLM 额度耗尽、搜索引擎不可达、网页防爬阻断等外部故障。
历史实现会在这些场景静默切换到离线仿真 / 合成兜底数据，且仍返回 status=success，
导致使用者无法区分报告内容是真实检索产物还是模型先验知识。

本模块提供进程级、线程安全的降级事件账本：
- 每次研究任务开始时（Supervisor 首节点）reset()；
- 任何故障兜底路径必须调用 record() 登记事件，禁止静默；
- Synthesizer / API / CLI 统一读取 degraded 标志与事件明细并对外暴露。

注意：降级事件表示"该环节未按真实路径执行"，不代表任务失败；
但只要存在降级事件，最终报告就必须显式标注其可信范围。
"""

import threading
from typing import List


class DegradationTracker:
    """线程安全的降级事件收集器（LangGraph 并发 Worker 运行在多个线程中）。"""

    def __init__(self) -> None:
        self._events: List[str] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        """清空账本，在每次研究任务启动时调用。"""
        with self._lock:
            self._events = []

    def record(self, event: str) -> None:
        """登记一条降级事件；相同事件去重，避免并发 Worker 重复刷屏。"""
        with self._lock:
            if event not in self._events:
                self._events.append(event)

    @property
    def events(self) -> List[str]:
        with self._lock:
            return list(self._events)

    @property
    def degraded(self) -> bool:
        with self._lock:
            return len(self._events) > 0


# 全局单例
degradation_tracker = DegradationTracker()

# 历史实现中出现过的仿真 / 占位域名，一律视为不可信信源，禁止进入参考文献
SIMULATED_DOMAINS = (
    "research.benchmark.org",
    "research.agent-systems.org",
    "benchmark.mcp-protocol.org",
    "source.example.com",
)


def is_simulated_url(url: str) -> bool:
    """判断 URL 是否为历史遗留的仿真 / 占位信源。"""
    if not url:
        return True
    return any(domain in url for domain in SIMULATED_DOMAINS)
