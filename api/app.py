"""
FastAPI 异步微服务入口模块 (FastAPI Async Microservice & SSE Streaming)
========================================================================
为上层 Web UI、移动端或第三方企业系统提供高并发、异步非阻塞的 API 访问能力。

【核心功能端点】：
1. RESTful 同步端点 (`POST /api/v1/research/start`)：
   基于 `asyncio.to_thread` 在后台专用工作线程池中执行 LangGraph 同步图，
   避免长时间计算与密集网络 I/O 阻塞 FastAPI 主事件循环 (Event Loop)；
2. SSE 真流式推送端点 (`GET /api/v1/research/stream`)：
   集成 `sse-starlette` 与 LangGraph 原生 `.stream()` 生成器，
   将多智能体协作全周期的“启动 -> Supervisor大纲规划 -> Worker子任务探针 -> Reflector质检反思 -> Synthesizer长文合成 -> 闭环完成”
   以标准的 Server-Sent Events (SSE) 事件流实时推向前端，实现极致的实时透明交互；
3. 成果研报管理端点 (`GET /api/v1/reports`, `GET /api/v1/reports/{filename}`)：
   提供本地持久化 Markdown 研报的元数据检索与原文流式读取。
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from core.config import settings
from agent.graph import research_graph
from agent.state import ResearchState
from api.schemas import ResearchRequest, ResearchResponse, ReportSummary

# 初始化微服务全局日志格式
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API-Server")

# 创建 FastAPI 核心应用实例
app = FastAPI(
    title="Enterprise Deep Research Agent API",
    description="基于 LangGraph 与 FastMCP 的企业级自主深度研究多智能体系统后端微服务",
    version="1.0.0"
)

# 配置 CORS 跨域资源共享中间件：默认仅放行本机前端来源，可通过环境变量 CORS_ORIGINS 配置
# 禁止 allow_origins=["*"] 与 allow_credentials=True 的组合（浏览器规范禁止，且会放开跨站携带凭据）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 确保研报存储物理目录存在
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/v1/health", summary="健康检查接口")
async def health_check():
    """
    检测服务运行状态、时间戳及各个分层多智能体配置的模型版本。
    """
    return {
        "status": "healthy",
        "service": "Enterprise Deep Research Agent",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "supervisor": settings.SUPERVISOR_MODEL,
            "worker": settings.WORKER_MODEL,
            "synthesizer": settings.SYNTHESIZER_MODEL
        }
    }


@app.post("/api/v1/research/start", response_model=ResearchResponse, summary="同步发起深度研究任务")
async def start_research(req: ResearchRequest):
    """
    同步阻塞调用 LangGraph 深度研究状态机，直至整份研报生成完毕并返回完整 JSON。
    
    【工程考量】：
    LangGraph 的 `.invoke` 是同步阻塞操作。为了防止长达数十秒的运算卡死 FastAPI 的单线程异步 Event Loop，
    采用 `await asyncio.to_thread(research_graph.invoke, initial_state)` 将其放入后台线程池安全执行。
    """
    logger.info(f"[API] 收到同步研究请求: topic='{req.topic}', max_loops={req.max_loops}")
    
    # 构建初始状态字典
    initial_state: ResearchState = {
        "topic": req.topic,
        "summary_brief": "",
        "outline_sections": [],
        "completed_sections": [],
        "global_notes": [],
        "research_loop_count": 0,
        "max_loops": req.max_loops or settings.MAX_RESEARCH_LOOPS,
        "is_sufficient": False,
        "feedback_notes": [],
        "final_report": None,
        "total_sources_count": 0,
        "degradation_events": []
    }

    try:
        # 放入线程池执行，保障事件循环高并发响应其他请求
        final_state = await asyncio.to_thread(research_graph.invoke, initial_state)

        degradation_events = final_state.get("degradation_events", [])
        return ResearchResponse(
            topic=req.topic,
            summary_brief=final_state.get("summary_brief", ""),
            total_sections=len(set(s.section_id for s in final_state.get("completed_sections", []))),
            total_sources=final_state.get("total_sources_count", 0),
            report_markdown=final_state.get("final_report", ""),
            status="success" if not degradation_events else "success_degraded",
            degraded=bool(degradation_events),
            degradation_events=degradation_events
        )
    except Exception as e:
        logger.error(f"[API] 研究执行异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"研究任务执行失败: {str(e)}")


@app.get("/api/v1/research/stream", summary="SSE 实时流式推送研究进展")
async def stream_research(topic: str = Query(..., description="待研究的主题")):
    """
    基于 Server-Sent Events (SSE) 协议实时向客户端推送 Agent 思考链、大纲、节点完成与研报合成事件。
    
    【协议优势】：
    相比传统轮询 (Polling)，SSE 采用单向长连接，HTTP 协议天然支持，轻量且具备重连机制，
    非常适合大模型 Agent 这类单次长时运行、过程事件密集的流式推送场景。
    """
    logger.info(f"[API] 收到 SSE 流式订阅请求: topic='{topic}'")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """异步事件生成器：对接 LangGraph 的 stream 迭代器并格式化为标准 SSE 数据帧。"""
        initial_state: ResearchState = {
            "topic": topic,
            "summary_brief": "",
            "outline_sections": [],
            "completed_sections": [],
            "global_notes": [],
            "research_loop_count": 0,
            "max_loops": settings.MAX_RESEARCH_LOOPS,
            "is_sufficient": False,
            "feedback_notes": [],
            "final_report": None,
            "total_sources_count": 0,
            "degradation_events": []
        }
        
        # 发送起始事件
        yield json.dumps({
            "event": "start",
            "message": f"深度研究任务已启动：{topic}",
            "timestamp": datetime.now().isoformat()
        }, ensure_ascii=False)
        
        degradation_events: list = []
        try:
            # 遍历 LangGraph 流式执行产生的每一步状态切片。
            # 注意：research_graph.stream() 是同步迭代器，节点内部包含阻塞式网络 I/O，
            # 必须放入线程池迭代（iterate_in_threadpool），否则会阻塞 FastAPI 事件循环
            sync_iterator = iterate_in_threadpool(research_graph.stream(initial_state))
            async for step_output in sync_iterator:
                for node_name, node_state in step_output.items():
                    event_data = {
                        "event": "node_completed",
                        "node": node_name,
                        "timestamp": datetime.now().isoformat()
                    }

                    # 依据不同节点输出定制化进度消息
                    if node_name == "supervisor":
                        event_data["message"] = "Supervisor 规划大纲已完成"
                        event_data["sections_count"] = len(node_state.get("outline_sections", []))
                    elif node_name == "research_worker":
                        event_data["message"] = "Worker 子章节调研完成"
                        event_data["completed_sections"] = len(node_state.get("completed_sections", []))
                    elif node_name == "reflector":
                        event_data["message"] = "Reflector 反思与差距评估完成"
                        event_data["is_sufficient"] = node_state.get("is_sufficient", True)
                    elif node_name == "synthesizer":
                        event_data["message"] = "Synthesizer 研报长文合成完毕"
                        event_data["report_length"] = len(node_state.get("final_report", ""))
                        degradation_events = node_state.get("degradation_events", [])

                    yield json.dumps(event_data, ensure_ascii=False)
                    await asyncio.sleep(0.05)  # 微休眠以防缓冲区粘包

            # 发送完成事件（显式携带降级标志，前端可据此展示可信范围提示）
            yield json.dumps({
                "event": "done",
                "message": "深度研究全部流程闭环完成！",
                "degraded": bool(degradation_events),
                "degradation_events": degradation_events,
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False)
            
        except Exception as e:
            # 捕获异常并通过流式信道回传错误信息
            yield json.dumps({
                "event": "error",
                "message": f"流式处理异常: {str(e)}"
            }, ensure_ascii=False)

    return EventSourceResponse(event_generator())


@app.get("/api/v1/reports", response_model=list[ReportSummary], summary="获取所有已生成的深度研报列表")
async def list_reports():
    """
    遍历本地 reports/ 目录，按修改时间倒序返回所有生成的研报元数据清单。
    """
    summaries = []
    for file in REPORTS_DIR.glob("*.md"):
        stat = file.stat()
        summaries.append(ReportSummary(
            filename=file.name,
            size_bytes=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        ))
    return sorted(summaries, key=lambda x: x.modified_time, reverse=True)


@app.get("/api/v1/reports/{filename}", response_class=PlainTextResponse, summary="读取指定研报的 Markdown 原文")
async def get_report_content(filename: str):
    """
    读取指定研报的 Markdown 完整纯文本正文。

    安全约束：
    - 只允许读取 REPORTS_DIR 目录内、解析后的真实路径仍位于该目录内的文件（防路径遍历）；
    - 只允许 .md 后缀；
    - 禁止任何绝对路径或包含目录分隔符的输入。
    """
    reports_root = REPORTS_DIR.resolve()
    file_path = (reports_root / filename).resolve()

    # 路径遍历防御：解析后的真实路径必须仍在 reports 目录内
    if not file_path.is_relative_to(reports_root):
        raise HTTPException(status_code=400, detail="非法的文件路径")
    # 只允许读取 Markdown 研报，且必须是文件
    if file_path.suffix.lower() != ".md" or not file_path.is_file():
        raise HTTPException(status_code=404, detail="未找到该研报文件")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    # 本地直接运行当前脚本时启动 Uvicorn 高性能 ASGI 服务器
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=settings.API_PORT, reload=True)

