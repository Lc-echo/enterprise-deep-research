"""
FastAPI 数据契约与校验模块 (Data Contracts & Schemas)
======================================================
基于 Pydantic v2 定义系统的 RESTful API 请求载荷、响应实体与文件元数据模型，
自动生成 OpenAPI (Swagger) 交互式文档与强类型输入校验。
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """
    发起深度研究任务的请求数据契约 (Request Payload)。
    """
    topic: str = Field(
        ...,
        description="待研究的主题或课题内容（如前沿技术、行业竞品、宏观经济等）",
        min_length=2,
        json_schema_extra={"example": "大模型长程 Agent 的工程化落地与挑战"}
    )
    max_loops: Optional[int] = Field(
        default=2,
        description="最大反思探索轮次上限，防止不可控死循环造成 API 账单失控",
        ge=1,
        le=5,
        json_schema_extra={"example": 2}
    )


class ResearchResponse(BaseModel):
    """
    深度研究同步完成响应数据实体 (Response Model)。
    """
    topic: str = Field(description="研究总课题名称")
    summary_brief: str = Field(description="Supervisor 规划的大纲全局摘要")
    total_sections: int = Field(description="生成的子章节总数")
    total_sources: int = Field(description="采纳并验证的独立权威真实信源数量")
    report_markdown: str = Field(description="最终合成的完整学术级 Markdown 深度研报全文")
    status: str = Field(default="success", description="任务执行状态标识")
    degraded: bool = Field(default=False, description="是否存在降级环节（LLM/搜索/抓取失败导致部分内容非真实检索产物）")
    degradation_events: List[str] = Field(default_factory=list, description="降级事件明细，空列表表示全链路真实执行")


class ReportSummary(BaseModel):
    """
    研报本地文件元数据模型 (Report File Metadata)。
    """
    filename: str = Field(description="Markdown 研报文件名")
    size_bytes: int = Field(description="文件大小（字节数）")
    modified_time: str = Field(description="最后修改/生成时间 (格式: YYYY-MM-DD HH:MM:SS)")

