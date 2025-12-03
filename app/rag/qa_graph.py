from typing import TypedDict, List, Any # TypedDict用于定义结构化字典的键和类型

from langgraph.graph import StateGraph, START, END # StateGraph用于构建带状态的工作流图
# START / END：表示图的起始和终止节点
from langchain_core.messages import HumanMessage, AIMessage
# HumanMessage：代表用户输入 AIMessage：代表 AI 的系统或回答消息（这里用于注入系统提示）

from app.rag.prompts import QA_SYSTEM, QA_USER
from app.deps import get__llm, get_vs


class QAState(TypedDict, total=False):
    # 定义一个名为QAState的类型化字典,作为整个工作流的共享状态
    # total=False表示这些字段不是必须全部存在可按需填充
    question: str # 用户提出的问题
    text: str # 备用输入字段(可能用于非结构化文本)
    user_role: str # 用户角色,用于权限过滤
    docs: List[Any] # 检索到的相关文档列表(来自向量数据库)
    answer: str # 最终生成的回答
    messages: List[Any] # 可选的消息历史


def decide_retrieve(state: QAState) -> str:
    # 路由函数(用于add_conditional_edges)
    # 它读取当前状态,决定下一步去哪个节点
    # 目前硬编码返回"retrieve" ,表示总是先检索
    return "retrieve"

def decide_retrieve_node(state: QAState) -> dict:
    """
    这是一个实际的图节点 runnable：必须返回 dict(用于更新state)
    这里只是一个no-op节点，仅作为路由跳板
    真正路由在decide_retrieve()里完成(LangGraph的设计模式)
    """
    return {}

def retrieve(state: QAState) -> dict:
    vs = get_vs() # 获取向量存储实例(Chroma)

    role = state.get("user_role", "public")
    # 获取用户角色(默认"public",即公开内容)

    query = state.get("question") or state.get("text") or ""
    # 获取查询文本:优先用question,其次是text,否则是空字符串

    retriever = vs.as_retriever(
        #创建带权限过滤的检索器(k=8)
        search_kwargs={
            "k": 8,
            "filter": {"visibility": {"$in": ["public", role]}},
            # 只检索visibility字段为"public"或当前用户角色的文档
            # 使用MongoDB风格的$in查询(Chroma支持)
        }
    )
    docs = retriever.invoke(query) # 执行检索,得到文档列表

    if not docs:
        # 兜底策略:如果权限过滤后无结果,则取消过滤再试一次(返回所有可见性文档)
        # 返回包含docs、question和调试标记debug的字段(会合并到全局state)
        retriever2 = vs.as_retriever(search_kwargs={"k": 8})
        docs = retriever2.invoke(query)
        return {"docs": docs, "question": query, "debug": "fallback_unfiltered"}

    # 正常情况下返回检索结果和调试信息
    return {"docs": docs, "question": query, "debug": "filtered"}


def grade_evidence(state: QAState) -> str:
    """
    检索后判断是否有证据。
    如果docs非空 返回"good" 否则返回"bad"
    该函数用于后续条件路由
    """
    return "good" if state.get("docs") else "bad"


def generate_answer(state: QAState) -> dict:
    """带引用生成答案。"""
    llm = get__llm()
    # 获取LLM实例和文档列表
    docs = state.get("docs", [])

    context = "\n\n".join(
        # 构造带编号的上下文(最多取前6个文档)
        # 格式：[1] 文本内容\n(source=文件路径, page=页码)
        # 便于LLM引用和用户溯源
        f"[{i+1}] {d.page_content}\n(source={d.metadata.get('source')}, page={d.metadata.get('page')})"
        for i, d in enumerate(docs[:6])
    )

    prompt = QA_USER.format(question=state["question"], context=context) # 用真实问题和上下文填充QA_USER模板
    messages = [AIMessage(content=QA_SYSTEM), HumanMessage(content=prompt)] # 构造消息列表
    # 第一条是AIMessage,内容为系统提示(QA_SYSTEM) 这是LangChain模拟系统消息常见技巧
    # 第二条是用户问题(含上下文)
    ans = llm.invoke(messages).content # 调用LLM生成回答
    return {"answer": ans} # 返回更新状态


def refuse_or_clarify(state: QAState) -> dict:
    """无证据兜底。当检索不到文档时，返回礼貌的拒绝回答，并引导用户提供更多信息。"""
    return {
        "answer": "我没有在当前可见知识库中找到足够证据回答。请提供更具体的关键词/文档来源。"
    }


def build_qa_graph():
    """构建工作流图"""
    g = StateGraph(QAState) # 创建一个以QAState为状态类型的图

    # 注意：节点注册用 runnable（必须返回 dict更新state）
    # 注册四个节点,每个节点对应一个函数
    g.add_node("decide_retrieve", decide_retrieve_node)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate_answer)
    g.add_node("refuse", refuse_or_clarify)

    g.add_edge(START, "decide_retrieve") # 从起始节点直接进入decide_retrieve节点

    g.add_conditional_edges(
        "decide_retrieve",
        # 从decide_retrieve节点出发
        # 调用decide_retrieve()函数决定下一步
        decide_retrieve,
        {
            #若返回retrieve 跳转到retrieve节点
            "retrieve": "retrieve",
            #若返回direct 直接生成答案
            "direct": "generate",
        },
    )

    g.add_conditional_edges(
        "retrieve",
        # 从retrieve节点出发
        # 调用grade_evidence()判断是否有证据
        # good就生成答案 bad就拒绝回答
        grade_evidence,
        {
            "good": "generate",
            "bad": "refuse",
        },
    )
    # 无论是生成答案还是拒绝 都走向结束
    g.add_edge("generate", END)
    g.add_edge("refuse", END)

    # 编译图,返回一个可调用的LangGraph Runnable对象(可用于.invoke(state))
    return g.compile()