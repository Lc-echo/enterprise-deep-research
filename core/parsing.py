"""
LLM 结构化输出统一解析模块 (Structured Output Parsing & Retry)
===============================================================
历史上每个节点各自手写"剥离 ```json 代码块 → json.loads → 构造 Pydantic 对象"的逻辑，
存在三个问题：
1. 逻辑重复且脆弱：模型输出不合规时直接抛错，只能静默换仿真数据；
2. 没有自我纠错机会：模型首次输出坏 JSON 时，没有任何"带错误信息的重试"；
3. 异常口径不统一。

本模块提供统一入口：
- 自动剥离 Markdown 代码块围栏；
- json.loads 后用 Pydantic 模型强校验；
- 失败时携带原始错误让模型**重试一次**（常见的 JSON 语法错误大多可自愈）；
- 仍失败则抛出 StructuredOutputError，由调用方决定降级策略并显式登记。
"""

import json
import logging
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger("Structured-Parsing")

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """LLM 结构化输出在重试后仍无法解析为目标 Schema。"""


def strip_code_fence(text: str) -> str:
    """剥离模型可能包裹的 ```json / ``` Markdown 围栏。"""
    text = text.strip()
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text


def parse_structured(text: str, schema: Type[T]) -> T:
    """
    将模型文本解析为指定 Pydantic 模型；解析失败抛 StructuredOutputError。
    """
    cleaned = strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise StructuredOutputError(f"输出不是合法 JSON: {e}") from e
    try:
        return schema(**data)
    except (ValidationError, TypeError) as e:
        raise StructuredOutputError(f"JSON 不符合 {schema.__name__} 结构: {e}") from e


def structured_invoke(llm, messages, schema: Type[T], *, retries: int = 1) -> T:
    """
    调用 LLM 并把结果解析为 schema 实例；首次失败时带错误信息重试 retries 次。

    Args:
        llm: 任何具备 .invoke(messages) -> AIMessage-like 的 LangChain 模型
        messages: 初始消息列表
        schema: 目标 Pydantic 模型类
        retries: 解析失败后的重试次数（默认 1）

    Returns:
        schema 的实例

    Raises:
        StructuredOutputError: 重试后仍无法解析
    """
    current_messages = list(messages)
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = llm.invoke(current_messages)
            content = getattr(response, "content", response)
            return parse_structured(content, schema)
        except StructuredOutputError as e:
            last_error = e
            if attempt < retries:
                logger.warning(f"[Structured-Parsing] 第 {attempt+1} 次解析失败（{e}），要求模型重试")
                # 将模型原输出与错误回灌，引导其自我纠正
                current_messages = list(messages) + [
                    AIMessage(content=getattr(response, "content", "")),
                    HumanMessage(content=(
                        f"你上一次的输出无法被解析，错误是：{e}。"
                        "请只输出一个可被标准 json.loads 解析的 JSON 对象，"
                        "不要输出 Markdown 代码块、注释或任何解释性文字。"
                    )),
                ]

    raise StructuredOutputError(f"经 {retries+1} 次尝试仍无法得到合法 {schema.__name__}: {last_error}")
