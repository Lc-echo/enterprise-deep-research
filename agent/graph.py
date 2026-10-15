"""
LangGraph 拓扑装配与图编译模块 (StateGraph Assembly & Compilation)
====================================================================
本模块负责将各个独立的 Agent 节点函数组装为一个完整的有向循环状态机 (Directed Cyclic Graph)，
并编译为可执行的 `CompiledStateGraph` 实例。

【图拓扑结构 (Graph Topology)】：
1. [START] -> [supervisor]：
   起点进入规划总监，进行课题解构与子任务大纲生成；
2. [supervisor] -(条件边 Send API: Map)-> [research_worker 并发集群]：
   动态扇出 (Fan-Out)，根据大纲章节列表派发 N 个独立的子任务探针并行调研；
3. [research_worker] -(状态归约 Reduce)-> [reflector]：
   所有并发 Worker 完成后，增量合并到全局状态，汇聚流转至质量评审节点；
4. [reflector] -(条件边决策判断)->
   ├── [is_sufficient=False 且未熔断] ──> 回流至 [supervisor]（闭环自校正环）
   └── [is_sufficient=True 或触发熔断] ──> 流转至 [synthesizer]（终止态归约）
5. [synthesizer] -> [END]：
   完成最终长文统稿润色、双保险超链接置换与磁盘持久化后抵达终点。
"""

import logging
from langgraph.graph import StateGraph, START, END
from agent.state import ResearchState
from agent.supervisor import plan_outline_node, route_to_workers
from agent.worker import research_worker_node
from agent.reflector import reflect_node, route_after_reflection
from agent.synthesizer import synthesizer_node

logger = logging.getLogger("Agent-Graph")


def build_research_graph():
    """
    组装并编译企业级深度研究多智能体系统的完整 StateGraph 状态拓扑。
    
    【核心实现流程】：
    1. 实例化 `StateGraph(ResearchState)`：绑定全局强类型状态机字典；
    2. 注册 4 个核心业务节点：supervisor, research_worker, reflector, synthesizer；
    3. 编排静态普通边与动态条件路由边 (Conditional Edges)；
    4. 编译输出 `CompiledStateGraph`，支持 `.invoke()` 同步执行与 `.stream()` 流式追踪。
    
    Returns:
        CompiledStateGraph: 可被 API 服务与 CLI 运行器调用的编译后图实例
    """
    logger.info("[Graph] 正在组装 LangGraph 多智能体研究工作流状态图...")
    builder = StateGraph(ResearchState)
    
    # =========================================================================
    # 步骤 1：注册核心计算节点 (Node Registration)
    # =========================================================================
    builder.add_node("supervisor", plan_outline_node)          # 顶层规划总监节点
    builder.add_node("research_worker", research_worker_node)  # 并发章节调研工作节点
    builder.add_node("reflector", reflect_node)                # 反思质检与熔断节点
    builder.add_node("synthesizer", synthesizer_node)          # 研报长文统稿合成节点
    
    # =========================================================================
    # 步骤 2：构建拓扑边与条件分支 (Edges & Conditional Routing)
    # =========================================================================
    
    # 起点边：接收用户初始状态输入，首个执行节点必须是规划总监 supervisor
    builder.add_edge(START, "supervisor")
    
    # 动态并行映射边 (Dynamic Fan-Out / Map Edge)：
    # supervisor 节点执行完毕后，执行 route_to_workers 函数。
    # 该函数返回 List[Send]，LangGraph 会自动为每个 Send 实例化一个并发的 research_worker 任务
    builder.add_conditional_edges(
        "supervisor",
        route_to_workers,
        ["research_worker"]
    )
    
    # 状态归约收敛边 (Fan-In / Reduce Edge)：
    # 所有并行派发的 research_worker 分支全部执行完毕后，
    # 其返回的章节数据通过 operator.add 归约汇聚，然后统一流入 reflector 节点
    builder.add_edge("research_worker", "reflector")
    
    # 自校正反思环与退出决策边 (Cyclic Feedback Edge)：
    # reflector 节点执行后，由 route_after_reflection 函数评估判定：
    # - 若判定信息饱和或触发安全熔断上限 -> 流向 synthesizer
    # - 若判定存在盲区且未达熔断上限 -> 闭环回流至 supervisor 形成学习环路
    builder.add_conditional_edges(
        "reflector",
        route_after_reflection,
        {
            "synthesizer": "synthesizer",
            "supervisor": "supervisor"
        }
    )
    
    # 终点边：synthesizer 完成长文润色与磁盘落盘后，工作流抵达终点 END
    builder.add_edge("synthesizer", END)
    
    # =========================================================================
    # 步骤 3：编译状态图
    # =========================================================================
    graph = builder.compile()
    logger.info("[Graph] LangGraph 状态图编译成功！")
    return graph


# 导出模块级编译好的全局图实例，供 FastAPI 微服务与 CLI 脚本直接复用
research_graph = build_research_graph()

