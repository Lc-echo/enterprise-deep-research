"""
交互式终端运行器与端到端度量工具 (CLI Interactive Runner)
===========================================================
基于 Rich 终端现代化渲染库构建的交互式测试与演示客户端。

【核心设计与交互价值】：
1. 跨平台编码安全：自动重配置 Windows 控制台的 stdout/stderr 编码为 UTF-8，彻底解决中文乱码与 GBK 字符集崩溃；
2. 动态进度反馈：使用 Rich Progress Spinner 呈现多智能体并行异步状态机运转过程；
3. 结构化大纲与度量表格：以彩色表格形式展示 Supervisor 拆解出的各章节标题、研究子问题与执行状态；
4. 深度研报即时渲染：在终端直接渲染 Markdown 标题、表格、角标与可点击溯源超链接，提供极佳的开发者体验 (DX)。
"""

import sys

# -----------------------------------------------------------------------------
# 跨平台终端编码自适应配置 (Windows UTF-8 Hotfix)
# -----------------------------------------------------------------------------
# 针对 Windows 控制台默认代码页 (GBK / CP936) 极易出现 UnicodeEncodeError 的历史痛点，
# 在运行时强制将输出缓冲通道重配置为 UTF-8，并采用 'replace' 策略优雅容错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from core.config import settings
from agent.graph import research_graph
from agent.state import ResearchState

# 初始化 Rich 终端操作句柄
console = Console()


def run_interactive_session():
    """
    终端交互会话主入口函数。
    
    【执行流水线】：
    1. 打印系统启动 Banner 与大模型配置概览；
    2. 获取用户待研究课题（支持命令行直接传参，亦支持终端实时交互输入）；
    3. 组装初始全局状态机字典 (ResearchState)；
    4. 启动 Rich 动效 Spinner 监听并执行 LangGraph 状态机；
    5. 结构化展示执行统计：
       - Supervisor 大纲拆解与章节状态明细表格
       - 反思轮次、剪枝卡片数量、信源总数与耗时看板
       - 最终研报 Markdown 预览与可验证超链接账本；
    6. 输出研报本地持久化绝对路径。
    """
    # 打印欢迎面板与模型配置
    console.print(Panel.fit(
        "[bold cyan]Enterprise Deep Research Agent[/bold cyan]\n"
        "[dim]基于 LangGraph 与 FastMCP 的企业级自主长程深度研究系统[/dim]\n"
        f"[green]后端模型配置:[/green] Supervisor={settings.SUPERVISOR_MODEL} | Worker={settings.WORKER_MODEL}",
        border_style="cyan"
    ))
    
    default_topic = "大模型长程自主智能体(Long-Horizon Agent)工程化落地与挑战"
    
    # -------------------------------------------------------------------------
    # 1. 课题输入解析 (支持 CLI 参数或交互式 prompt)
    # -------------------------------------------------------------------------
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:]).strip()
    else:
        console.print(f"\n[bold yellow]请输入待研究课题[/bold yellow] (输入后回车；直接回车则使用默认课题):")
        try:
            user_input = input(">> ").strip()
            topic = user_input if user_input else default_topic
        except (EOFError, KeyboardInterrupt):
            topic = default_topic
            
    console.print(f"\n[bold green]>>> 正在启动深度研究流水线:[/bold green] [white]{topic}[/white]\n")
    
    # 构建初始状态容器
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
    
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # 2. 动效 Spinner 驱动 LangGraph 图执行
    # -------------------------------------------------------------------------
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]正在进行多智能体协同调研与研报生成...", total=None)
        
        # 同步触发 LangGraph 工作流图，内部并发调度 Worker 与 Synthesizer
        final_state = research_graph.invoke(initial_state)
        progress.update(task, completed=True)
        
    elapsed = time.time() - start_time
    
    # -------------------------------------------------------------------------
    # 3. 输出结构化大纲拆解与任务看板
    # -------------------------------------------------------------------------
    console.print(Panel("[bold green][OK] 1. Supervisor 课题拆解与分层规划[/bold green]", border_style="green"))
    console.print(f"[bold]研究摘要:[/bold] {final_state.get('summary_brief', '')}\n")
    
    table = Table(title="[bold cyan]规划与完成章节大纲列表[/bold cyan]", show_header=True, header_style="bold magenta")
    table.add_column("章节 ID", style="dim", width=10)
    table.add_column("章节标题", width=35)
    table.add_column("核心研究问题", width=40)
    table.add_column("状态", justify="center", width=10)
    
    # 多轮迭代可能产生同 ID 章节，按 section_id 去重后展示（与 Synthesizer 口径一致）
    seen_section_ids = set()
    for s in final_state.get("completed_sections", []):
        if s.section_id in seen_section_ids:
            continue
        seen_section_ids.add(s.section_id)
        questions_str = "\n".join([f"- {q}" for q in s.research_questions])
        table.add_row(s.section_id, s.title, questions_str, "[green]已完成[/green]")

    console.print(table)

    # -------------------------------------------------------------------------
    # 4. 输出执行统计与效能度量面板
    # -------------------------------------------------------------------------
    degradation_events = final_state.get("degradation_events", [])
    console.print(Panel(
        f"[bold green][OK] 2. 反思评审与事实剪枝统计[/bold green]\n"
        f"- 迭代轮次: [cyan]{final_state.get('research_loop_count', 1)}[/cyan] 轮\n"
        f"- 提纯事实证据卡片: [cyan]{len(final_state.get('global_notes', []))}[/cyan] 张\n"
        f"- 独立真实信源总数: [cyan]{final_state.get('total_sources_count', 0)}[/cyan] 个\n"
        f"- 全链路端到端耗时: [cyan]{elapsed:.2f}[/cyan] 秒\n"
        f"- 降级事件: [cyan]{len(degradation_events)}[/cyan] 条",
        border_style="green"
    ))

    # 降级事件必须显著提示，避免把离线仿真产物误当作真实调研结论
    if degradation_events:
        warn_lines = "\n".join([f"[red]{i+1}. {evt}[/red]" for i, evt in enumerate(degradation_events)])
        console.print(Panel(
            f"[bold red]警告：本次运行存在降级环节，报告中相关内容不代表真实检索结果！[/bold red]\n\n{warn_lines}",
            title="[bold red]Degraded Mode[/bold red]",
            border_style="red"
        ))
    
    # -------------------------------------------------------------------------
    # 5. 最终 Markdown 研报预览与超链接信源验证
    # -------------------------------------------------------------------------
    final_report = final_state.get("final_report", "")
    console.print(Panel("[bold green][OK] 3. 最终研报成果预览[/bold green]", border_style="green"))
    console.print(Markdown(final_report[:1200] + "\n\n*(中段章节正文已省略，完整报告已持久化到 reports/ 目录)*"))
    
    # 提取文末参考文献并独立加亮展示
    ref_marker = None
    for kw in ["## 参考文献与信源链接", "## 参考文献", "## References"]:
        if kw in final_report:
            ref_marker = kw
            break
            
    if ref_marker:
        refs_snippet = final_report[final_report.rfind(ref_marker):]
        console.print(Panel(Markdown(refs_snippet), title="[bold cyan]溯源核验信源与超链接 (References)[/bold cyan]", border_style="cyan"))
    
    safe_topic = "".join([c for c in topic if c.isalnum() or c in (" ", "_", "-")]).strip().replace(" ", "_")
    report_file_path = Path("reports") / f"Report_{safe_topic[:30]}.md"
    console.print(f"\n[bold green][SUCCESS] 研究闭环圆满完成！[/bold green]")
    console.print(f"[bold cyan]完整 Markdown 研报路径:[/bold cyan] {report_file_path.resolve()}\n")


if __name__ == "__main__":
    run_interactive_session()

