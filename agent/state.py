"""
LangGraph 多智能体系统状态模型模块 (State Schema & Reducers)
============================================================
本模块定义了 LangGraph 有向图工作流中的全局状态 (ResearchState) 与局部子状态 (WorkerInput)，
以及在节点之间流转的数据载体模型 (EvidenceNote, Section, Outline)。

【核心技术设计要点】：
1. 强类型状态约定：采用 Pydantic v2 构建业务数据契约，严格约束 LLM 的结构化输出 (Structured Output)；
2. 经典 Map-Reduce 归约器 (Reducers)：
   在 LangGraph 中，当多个 Worker 节点并发执行完毕并向全局状态写入数据时，若使用普通赋值会发生覆盖。
   通过 `Annotated[List[...], operator.add]`，LangGraph 会自动将并发 Worker 返回的列表项执行增量累加；
3. 子上下文安全隔离 (Sub-Context Isolation)：
   定义专门的 `WorkerInput` 结构配合 LangGraph `Send` API，确保每个 Worker 仅持有本章节所需的目标问题与全局课题，
   避免多 Worker 之间 Prompt 上下文相互串扰与无谓的 Token 消耗膨胀。
"""

from typing import List, Optional, Annotated, Dict, Any
import operator
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class EvidenceNote(BaseModel):
    """
    状态剪枝 (State Pruning) 提纯事实证据卡片模型。
    
    【设计价值】：
    Worker 在调用工具抓取到数千字的原始网页文本后，不直接将原始 HTML/纯文本存入全局 State，
    而是由轻量模型提炼为 2~4 条核心事实论据。这样既保留了高密度信息与可溯源链接，
    又将后续合成节点面临的 Context Window 占用缩减了 80% 以上。
    """
    title: str = Field(description="来源标题、论文题目或信源机构名称")
    url: str = Field(description="原始来源 URL 或规范化文献检索标示")
    key_facts: List[str] = Field(description="核心事实论点列表，已剔除修饰性套话与无关广告")
    relevance_score: float = Field(default=1.0, description="与研究章节的相关性评分 (0.0 - 1.0)")


class Section(BaseModel):
    """
    深度研究报告的单个子章节大纲与内容定义契约。
    
    在流水线前期作为 Supervisor 拆解出的任务规格书 (Specification)，
    在流水线后期由 Worker 填入正文内容 (content) 与引用信源 (citations)，转变为已完成的章节成果。
    """
    section_id: str = Field(default="pending", description="章节唯一序号，例如 'sec_1'，用于后续顺序重排")
    title: str = Field(default="补充维度分析", description="章节标题，例如 '一、 核心概念与技术原理剖析'")
    description: str = Field(default="", description="本章节需研究的核心主题、边界与范围说明")
    research_questions: List[str] = Field(default_factory=list, description="指导 Worker 并发探针精准检索的具体子问题清单")
    content: Optional[str] = Field(default=None, description="Worker 撰写生成的 Markdown 章节正文，未完成时为 None")
    citations: List[str] = Field(default_factory=list, description="本章节正文中引用的来源列表，格式如 [标题](URL)")
    is_completed: bool = Field(default=False, description="当前章节是否已完成深入调研与正式撰写")


class Outline(BaseModel):
    """
    Supervisor 规划总监节点输出的完整研究大纲结构。

    作为整个深度研究任务的顶层蓝图 (Blueprint)，直接被 Pydantic 解析校验，确保下游 Map 派发的稳定性。
    """
    topic: str = Field(default="", description="用户输入的研究总课题")
    summary_brief: str = Field(description="课题的研究背景与核心诉求简述 (100字以内)")
    sections: List[Section] = Field(description="解构出的递进式子章节清单")


class RefinementSection(BaseModel):
    """反思补充章节契约（由模型输出，不强制要求由系统分配的 section_id）。"""
    title: str = Field(default="补充维度分析", description="补充章节标题")
    description: str = Field(default="", description="该补充章节需覆盖的盲区范围说明")
    research_questions: List[str] = Field(default_factory=list, description="指导 Worker 并发探针精准检索的具体子问题清单")


class RefinementPlan(BaseModel):
    """反思回流后 Supervisor 定向补充规划的输出结构（仅含增量章节）。"""
    sections: List[RefinementSection] = Field(default_factory=list, description="针对盲区的增量补充章节清单")


class ReflectionVerdict(BaseModel):
    """Reflector 质检节点的结构化判定结果。"""
    is_sufficient: bool = Field(default=True, description="信息覆盖度是否已达标")
    feedback_notes: List[str] = Field(default_factory=list, description="盲区或改进意见")
    missing_angles: List[str] = Field(default_factory=list, description="缺失的分析角度")


class PrunedFacts(BaseModel):
    """状态剪枝节点的结构化输出（提纯后的事实卡片内容）。"""
    key_facts: List[str] = Field(default_factory=list, description="提纯后的核心事实论据列表")


class ResearchState(TypedDict):
    """
    LangGraph 全局研究状态机字典 (Global State Machine Schema)。
    
    TypedDict 是 LangGraph 推荐的状态容器定义方式，支持在节点流转时进行渐进式更新。
    """
    # ------------------ 输入与顶层规划元数据 ------------------
    topic: str                                              # 用户输入的总课题名称
    summary_brief: str                                      # 课题概要与规划说明
    outline_sections: List[Section]                         # Supervisor 规划好的待完成章节基准大纲
    
    # ------------------ 并发 Map-Reduce 聚合状态 ------------------
    # 利用 Annotated[..., operator.add] 声明 Reducer 状态累加器：
    # 多个并发 Worker 节点返回的 {"completed_sections": [section]} 会被自动累加合并，绝不相互覆盖！
    completed_sections: Annotated[List[Section], operator.add]
    
    # 全局知识库笔记累加器：各个 Worker 产出的事实卡片集合
    global_notes: Annotated[List[EvidenceNote], operator.add]
    
    # ------------------ 控制流与反思熔断阈值 ------------------
    research_loop_count: int                                # 当前反思轮次计数器 (递增)
    max_loops: int                                          # 最大反思循环上限（硬性安全熔断器）
    is_sufficient: bool                                     # Reflector 反思节点判定的信息饱和度标志 (True/False)
    feedback_notes: List[str]                               # Reflector 反思节点提出的补充盲区或改进意见
    
    # ------------------ 最终产出成果 ------------------
    final_report: Optional[str]                             # Synthesizer 最终拼接润色后的 Markdown 深度研报全文
    total_sources_count: int                                # 报告中引用的去重后独立信源总数
    degradation_events: List[str]                           # 本次运行的降级事件账本（空列表表示全链路真实执行）


class WorkerInput(TypedDict):
    """
    通过 LangGraph 的 Send API 动态分发给 Worker 的独立子状态 (Isolated Sub-State)。
    
    【架构隔离考量】：
    WorkerInput 不包含全局的 completed_sections 或 global_notes，
    从而将每个 Worker 的提示词环境完全物理隔离，防止无关历史上下文造成的 Prompt 污染。
    """
    section: Section                                        # 当前 Worker 独立负责探究的子章节元数据
    topic: str                                              # 研究总课题名称 (提供宏观背景指引)

