"""Reflector 熔断与条件路由测试。"""

from agent.reflector import reflect_node, route_after_reflection


def test_route_to_synthesizer_when_sufficient():
    assert route_after_reflection({"is_sufficient": True}) == "synthesizer"


def test_route_back_to_supervisor_when_insufficient():
    assert route_after_reflection({"is_sufficient": False}) == "supervisor"


def test_route_defaults_to_synthesizer_on_missing_flag():
    """状态缺失时默认放行至合成节点，避免空值导致卡死。"""
    assert route_after_reflection({}) == "synthesizer"


def test_circuit_breaker_short_circuits_without_llm():
    """达到最大轮次时必须无条件熔断，且不依赖 LLM 判定。"""
    state = {
        "topic": "熔断测试课题",
        "research_loop_count": 2,
        "max_loops": 2,
        "completed_sections": [],
    }
    result = reflect_node(state)
    assert result["is_sufficient"] is True
    assert "最大安全迭代轮数" in result["feedback_notes"][0]


def test_circuit_breaker_respects_limit_boundary(offline_llm):
    """current_loop < max_loops 时不直接熔断，而是调用 LLM 评估。"""
    state = {
        "topic": "边界测试课题",
        "research_loop_count": 1,
        "max_loops": 2,
        "completed_sections": [],
    }
    result = reflect_node(state)
    assert "is_sufficient" in result
