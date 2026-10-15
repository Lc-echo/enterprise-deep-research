# Enterprise Deep Research Agent

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FastMCP](https://img.shields.io/badge/FastMCP-Stdio%20Isolated-green.svg)](https://github.com/jlowin/fastmcp)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async%20%2B%20SSE-teal.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-34%20Passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 **LangGraph** 与 **FastMCP** 的长程深度研究多智能体系统：给定一个研究课题，自动完成「大纲拆解 → 多章节并行检索 → 事实剪枝 → 反思补章 → 研报合成」，产出带真实信源超链接的 Markdown 研究报告，并提供 CLI 与 FastAPI（含 SSE 实时进度）两种使用方式。

## 架构概览

```
START
  └─► Supervisor（规划总监）
        │  首轮：将课题拆解为 N 个章节（Pydantic 强校验）
        │  回流：消费 Reflector 的反馈，只针对盲区增量补充章节
        │
        └─ Send API 动态扇出（仅派发未完成章节）
              ├─► Worker #1：检索 → 网页正文抽取 → 状态剪枝（事实卡片）→ 章节撰写
              ├─► Worker #2：…
              └─► Worker #N：…
                    │（operator.add 自动归约）
                    ▼
              Reflector（质检 + 熔断）
                    │ 信息不足且未达轮次上限 ──► 回流 Supervisor
                    │ 充分 / 触发熔断
                    ▼
              Synthesizer（章节排序去重 → 信源去重 → 统稿 → 引文双保险校验 → 落盘）
                    ▼
                  END
```

核心设计：

- **分层拓扑与动态并行**：`Supervisor-Worker` 分层，LangGraph `Send` 按大纲运行时扇出，章节间上下文隔离。
- **状态剪枝（State Pruning）**：网页原文不进全局状态，由轻量模型提纯为 2~4 条事实卡片，只回传高密度信息与真实 URL。
- **反思闭环（自校正）**：Reflector 评审覆盖度，反馈驱动 Supervisor 增量补章（新章节编号续接，不重复全量调研）；`max_loops` 硬熔断防止死循环与账单失控。
- **引文双保险 + 诚实降级**：Prompt 强制要求带 URL，合成后再用正则程序化校验，缺链接则用真实信源清单替换参考文献块；无真实信源时不渲染文献块，输出显式降级声明。
- **信源分级**：Tavily 高级检索 → 域名权威度（学术/官方/媒体 Tier 打分）重排 → 域名黑名单；DuckDuckGo 兜底。
- **真实 MCP 进程隔离**：工具集通过 `mcp_servers/mcp_client.py` 以 stdio 独立子进程运行（长连接 Client、懒启动、退出自动回收），编排层与工具层进程级隔离；连接失败自动降级进程内调用并记录降级事件。
- **结构化输出自愈**：所有节点的 JSON 输出统一经 `core/parsing.py` 解析（剥离代码围栏、Pydantic 校验、失败携带错误信息重试一次）。

## 目录结构

```
├── core/
│   ├── config.py          # pydantic-settings 配置（.env）
│   ├── llm.py             # LLM 工厂 + 离线仿真适配器 + 异常降级
│   ├── observability.py   # 降级事件追踪器（线程安全）
│   └── parsing.py         # 结构化输出统一解析（代码围栏剥离 / Pydantic 校验 / 重试）
├── agent/
│   ├── graph.py           # StateGraph 拓扑装配
│   ├── state.py           # ResearchState / Section / EvidenceNote 等数据契约
│   ├── prompts.py         # 全部 Prompt 模板
│   ├── supervisor.py      # 规划、定向补章、Send 路由
│   ├── worker.py          # 检索、抓取、剪枝、章节撰写
│   ├── reflector.py       # 质检与熔断路由
│   └── synthesizer.py     # 统稿、信源去重、引文校验、落盘
├── mcp_servers/
│   ├── search_tools.py    # FastMCP 工具服务：web_search / fetch_web_content（可独立 stdio 启动）
│   └── mcp_client.py      # MCP 同步桥接器：独立子进程长连接 + 故障降级直连
├── api/
│   ├── app.py             # FastAPI：同步 / SSE / 报告管理（CORS 白名单、路径遍历防护）
│   └── schemas.py         # 请求响应契约
├── scripts/
│   └── benchmark.py       # 离线可复现基准（并发加速 / 剪枝压缩率）
├── reports/               # 运行时输出（git 忽略）
├── examples/              # 示例报告（真实联网链路产出）
├── tests/                 # pytest 单元测试集（34 项全覆盖）
├── conftest.py            # pytest 全局 fixture（离线模型、假搜索、降级追踪重置）
├── run_cli.py             # 终端交互入口
└── LICENSE                # MIT 开源许可证
```

## 快速开始

```bash
# 1) 安装依赖（推荐 Python 3.11+）
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt

# 2) 配置凭据
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
# 编辑 .env 填入 OPENAI_API_KEY，可选填 TAVILY_API_KEY

# 3) 终端运行
python run_cli.py "大模型长程智能体的工程化落地与挑战"

# 4) 或启动 API 服务
uvicorn api.app:app --reload --port 8000
#   同步发起：POST /api/v1/research/start   {"topic": "...", "max_loops": 2}
#   SSE 进度：GET  /api/v1/research/stream?topic=...
#   报告列表：GET  /api/v1/reports
#   交互式文档：http://localhost:8000/docs
```

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI 兼容协议密钥（通义千问 / DeepSeek / OpenAI 等） | 空 |
| `OPENAI_BASE_URL` | OpenAI 兼容端点 | 通义千问 DashScope |
| `TAVILY_API_KEY` | Tavily 搜索 Key；留空则降级为 DuckDuckGo | 空 |
| `SUPERVISOR_MODEL` / `WORKER_MODEL` / `SYNTHESIZER_MODEL` | 分层模型选型 | qwen-plus / qwen-plus / qwen-max |
| `MAX_RESEARCH_LOOPS` | 最大调研轮次（熔断阈值） | 2 |
| `MAX_SECTIONS_PER_REPORT` | 首轮章节数上限 | 4 |
| `SEARCH_RESULTS_PER_QUERY` | 每个检索词召回数 | 4 |
| `API_PORT` | FastAPI 服务监听端口 | 8000 |
| `MCP_TRANSPORT` | 工具层传输：`stdio`（独立子进程）/ `direct`（进程内，调试用） | stdio |
| `CORS_ORIGINS` | API 允许的跨域来源（逗号分隔，默认仅本机） | localhost 系列 |

## 降级模式（重要且诚实的设计说明）

当 LLM 额度耗尽、搜索引擎不可达或网页全部防爬时，系统**不会伪造信源**：

1. 每次故障兜底都会登记到降级事件账本（`core/observability.py`）；
2. 搜索全通道失败 → 返回空结果，**不再产生虚构域名的文献卡片**；
3. 无真实信源 → 章节按模型先验知识撰写且不输出引用角标，文末参考文献替换为**降级声明**，报告顶部插入醒目的降级横幅；
4. API 响应返回 `status=success_degraded`、`degraded=true` 与 `degradation_events` 明细，CLI 同步红色警告。

离线仿真模型（`OfflineSimulationChatModel`）仅用于无 Key 的演示与测试，其产物会被完整标记，不会与真实检索结果混淆。

## 运行测试

```bash
pytest -v
```

覆盖 34 项测试：信源权威度打分排序、反思熔断路由、引文双保险与无信源降级声明、反思回流增量补章、FastMCP 子进程连接与降级、FastAPI 路径遍历防御与 CORS 白名单等。

## 基准数据

```bash
python scripts/benchmark.py    # 离线、约 10 秒、无 API 费用
```

- Send 扇出并发（4 章 × 1.5 秒 I/O 负载）：串行 6.0 秒 → 并发 1.5 秒，**加速约 4 倍、耗时降低约 75%**；
- 状态剪枝（1500 字固定正文）：提纯为 131 字事实卡片，**单篇压缩约 91%**；默认 4 章配置全报告理论压缩约 **95%**。

真实网络与真实 LLM 下数值会波动，脚本给出的是可复现的机制量级。

## 已知限制与改进方向

- MCP 工具层已实现 stdio 独立子进程模式；尚未覆盖 HTTP/SSE 远程传输与多工具进程池管理，是下一步方向。
- 无分布式任务队列（如 Celery / Temporal）与鉴权，长任务为线程池内异步执行，生产部署建议增加 Job 队列与网关鉴权。
- 暂无检索结果持久缓存与跨信源交叉验证，是后续主要迭代方向。

## 示例产出

[`examples/sample-report-long-horizon-agent.md`](examples/sample-report-long-horizon-agent.md) 为一次真实联网链路产出的报告（23 条真实信源链接），其中第二章节如实报告了"PDF 抓取为二进制流、无法提取有效正文"而未编造内容，体现了事实优先的降级策略。

## 开源协议

本项目基于 [MIT License](LICENSE) 协议开源。欢迎提交 Issue 与 Pull Request 共同完善！

