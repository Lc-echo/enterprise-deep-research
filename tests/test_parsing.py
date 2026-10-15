"""
core/parsing.py 结构化输出统一解析测试：代码围栏剥离、Schema 校验、失败重试、终态抛错。
"""

import pytest
from pydantic import BaseModel
from langchain_core.messages import AIMessage

from core.parsing import (
    strip_code_fence,
    parse_structured,
    structured_invoke,
    StructuredOutputError,
)


class Point(BaseModel):
    x: int
    y: int


class ScriptedLLM:
    """按调用顺序返回预设内容的假 LLM。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def invoke(self, messages):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return AIMessage(content=out)


def test_strip_code_fence_plain_and_fenced():
    assert strip_code_fence('{"x": 1}') == '{"x": 1}'
    assert strip_code_fence('```json\n{"x": 1}\n```') == '{"x": 1}'
    assert strip_code_fence('```\n{"x": 1}\n```') == '{"x": 1}'


def test_parse_plain_json():
    p = parse_structured('{"x": 1, "y": 2}', Point)
    assert p.x == 1 and p.y == 2


def test_parse_fenced_json():
    p = parse_structured('```json\n{"x": 3, "y": 4}\n```', Point)
    assert (p.x, p.y) == (3, 4)


def test_parse_invalid_json_raises():
    with pytest.raises(StructuredOutputError):
        parse_structured("not a json", Point)


def test_parse_schema_mismatch_raises():
    with pytest.raises(StructuredOutputError):
        parse_structured('{"x": 1}', Point)  # 缺 y


def test_structured_invoke_retry_then_succeed():
    llm = ScriptedLLM(['{"x": 1}', '{"x": 1, "y": 2}'])  # 首次缺字段，重试合规
    p = structured_invoke(llm, [], Point)
    assert (p.x, p.y) == (1, 2)
    assert llm.calls == 2
    # 重试时应把"错误纠正指令"回灌给模型
    # （ScriptedLLM 不记录 messages，这里通过调用次数间接验证）


def test_structured_invoke_retry_exhausted_raises():
    llm = ScriptedLLM(["garbage", "still garbage"])
    with pytest.raises(StructuredOutputError):
        structured_invoke(llm, [], Point, retries=1)
    assert llm.calls == 2  # 初次 + 1 次重试
