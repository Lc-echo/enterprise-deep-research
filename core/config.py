"""
配置管理模块 (Configuration Management Module)
==============================================
负责统一加载系统运行环境变量、大模型端点凭据以及多智能体执行控制策略。
采用 Pydantic-Settings 实现类型安全的配置注入，并按以下优先级加载参数：
1. 操作系统环境变量 (OS Environment Variables)
2. 项目根目录下的 .env 配置文件
3. 本模块内定义的默认兜底值 (Default Values)
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 定位并加载项目根目录下的 .env 配置文件
# 使用 resolve() 确保跨平台路径解析绝对准确
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    """
    系统全局配置类。
    
    封装所有与 LLM 服务提供商、检索服务、Agent 拓扑控制与服务端口相关的参数。
    支持在代码中通过 `settings.FIELD_NAME` 进行强类型访问。
    """
    
    # ==========================
    # 1. 大语言模型 (LLM) 接口配置
    # ==========================
    # OpenAI 兼容协议密钥 (支持通义千问 DashScope, DeepSeek, OpenAI 官方端点等)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # API 基础路径：默认采用阿里云百炼 DashScope 的 OpenAI 兼容接口，国内网络环境下访问极速稳定
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # ==========================
    # 2. 外部搜索引擎 (Search API) 配置
    # ==========================
    # Tavily AI Search API Key: 用于 Worker 探针调用专业学术/研报深度检索
    # 若留空，则系统会自动降级至 DuckDuckGo 或本地领域知识仿真库
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # ==========================
    # 3. 分层多智能体模型选型 (Model Selection Tiering)
    # ==========================
    # Supervisor 规划总监模型：负责宏观课题分解、生成大纲与子问题，推荐 qwen-plus 或 deepseek-chat
    SUPERVISOR_MODEL: str = os.getenv("SUPERVISOR_MODEL", "qwen-plus")
    
    # Worker 并发调研模型：负责高并发检索、网页抓取与事实卡片剪枝，追求高性价比与低延迟，推荐 qwen-plus
    WORKER_MODEL: str = os.getenv("WORKER_MODEL", "qwen-plus")
    
    # Synthesizer 研报合成模型：负责汇总万字正文、统一行文语调并编排学术级超链接，需要强逻辑与长文本能力，推荐 qwen-max
    SYNTHESIZER_MODEL: str = os.getenv("SYNTHESIZER_MODEL", "qwen-max")
    
    # ==========================
    # 4. 多智能体拓扑与控制阈值 (Loop & Concurrency Limits)
    # ==========================
    # 最大反思与补充调研轮次：作为硬性安全熔断器 (Circuit Breaker)，防止 Agent 在反思评估时陷入死循环造成账单失控
    MAX_RESEARCH_LOOPS: int = int(os.getenv("MAX_RESEARCH_LOOPS", "2"))
    
    # 单份研报生成的子章节上限：决定了 Supervisor 拆解子任务的数量及后续并发 Worker 分支的规模
    MAX_SECTIONS_PER_REPORT: int = int(os.getenv("MAX_SECTIONS_PER_REPORT", "4"))
    
    # 每个检索问题召回的候选网页数量上限：控制 Worker 的信源广度，平衡 Token 消耗与信息饱和度
    SEARCH_RESULTS_PER_QUERY: int = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "4"))
    
    # ==========================
    # 5. 后端微服务端口 (Server Port)
    # ==========================
    # FastAPI 异步服务监听端口号
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # CORS 允许来源（逗号分隔）；默认仅放行本机常用前端端口，严禁默认使用 "*"
    CORS_ORIGINS: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,http://localhost:3000"
    )

    # ==========================
    # 6. 工具层传输方式 (MCP Transport)
    # ==========================
    # stdio：以独立子进程启动 MCP server，通过标准输入输出走 MCP 协议（默认，名副其实的进程隔离）
    # direct：进程内直接导入函数调用（仅供离线测试/调试使用）
    MCP_TRANSPORT: str = os.getenv("MCP_TRANSPORT", "stdio")

    @property
    def cors_origin_list(self) -> list:
        """将逗号分隔的来源配置解析为去空白后的列表。"""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


# 全局单例配置对象，供整个应用模块导入使用
settings = Settings()
