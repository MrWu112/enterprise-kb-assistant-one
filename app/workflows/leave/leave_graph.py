from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from app.deps import get__llm
from app.workflows.leave.models import LeaveState
from app.workflows.leave.rules import validate_leave

SLOT_SYSTEM = (
    "你是企业HR请假助手。"
    "你的任务是从用户请假描述中抽取结构化信息。"
    "只输出JSON，不要解释。"
)

SLOT_USER = """请从下面文本中抽取字段，输出严格 JSON：
{{
  "leave_type": "annual|sick|personal|other",
  "start_time": "YYYY-MM-DD HH:MM 或 null",
  "end_time": "YYYY-MM-DD HH:MM 或 null",
  "reason": "string 或 null"
}}

要求：
- 如果用户没有明确说开始/结束时间，就输出 null
- 时间必须是 ISO 8601 格式（YYYY-MM-DD HH:MM）
- 不要编造时间
- 只输出 JSON

文本：{text}
"""

# 将str格式的json变成一个python中的字典
def _safe_json_load(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    try:
        return json.loads(s)  # loads用于将一个符合json的str变成字典
    except Exception:
        return {}

# 检查一个str表示的时间是否符合iso格式，之后转换为datetime并返回
def _safe_iso(s: Any) -> str | None:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    try:
        datetime.fromisoformat(s)
        return s
    except Exception:
        print('日期格式转换失败')
        return None

def _extract_leave_id(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\bLV-[0-9a-fA-F]{6,12}\b", text)
    return m.group(0) if m else None

def decide_intent(state: LeaveState) -> str:
    """apply / query / cancel"""
    text = (state.get("text") or state.get("question") or "").lower()

    if any(k in text for k in ["取消", "撤销", "作废"]):
        return "cancel"

    if any(k in text for k in ["查询", "查", "状态", "进度", "结果"]):
        if any(k in text for k in ["请假", "年假", "病假", "事假", "休假", "调休", "假期", "申请", "单"]):
            return "query"

    return "apply"


def extract_slots_node(state: LeaveState) -> dict:
    llm = get__llm()
    text = state.get("text", "") or state.get("question", "") or ""

    messages = [
        SystemMessage(content=SLOT_SYSTEM),
        HumanMessage(content=SLOT_USER.format(text=text)),
    ]

    raw = llm.invoke(messages).content
    data = _safe_json_load(raw)

    req = state.get("req") or {}
    req.update({
        "leave_type": data.get("leave_type") or req.get("leave_type"),
        "start_time": _safe_iso(data.get("start_time")) or req.get("start_time"),
        "end_time": _safe_iso(data.get("end_time")) or req.get("end_time"),
        "reason": data.get("reason") or req.get("reason"),
    })
    req["requester"] = state.get("requester", "anonymous")
    return {"req": req}


def validate_node(state: LeaveState) -> dict:
    req = state.get("req") or {}
    missing, violations = validate_leave(req, balance_days=5.0)
    return {"missing_fields": missing, "violations": violations, "req": req}


def decide_next(state: LeaveState) -> str:
    if state.get("missing_fields") or state.get("violations"):
        return "need_info"
    return "confirm"


def need_info_node(state: LeaveState) -> dict:
    missing = state.get("missing_fields") or []
    violations = state.get("violations") or []

    tips = []
    if missing:
        tips.append("缺少信息：" + "、".join(missing))
    if violations:
        tips.append("规则问题：" + "；".join(violations))

    return {"answer": "；".join(tips) + "。请补充/修正后再说一次。"}


def confirm_node(state: LeaveState) -> dict:
    req = state.get("req") or {}
    ans = (
        "请确认你的请假信息：\n"
        f"- 类型：{req.get('leave_type')}\n"
        f"- 开始：{req.get('start_time')}\n"
        f"- 结束：{req.get('end_time')}\n"
        f"- 时长：{req.get('duration_days')} 天\n"
        f"- 原因：{req.get('reason') or '无'}\n"
        "回复“确认”提交，或直接回复修改后的信息。"
    )
    return {"answer": ans}

def decide_confirm(state: LeaveState) -> str:
    text = (state.get("text") or "").strip().lower()
    if text in {"确认", "确定", "yes", "ok", "submit"}:
        return "create"
    return "end"


def create_leave_node(state: LeaveState) -> dict:
    leave_id = "LV-" + uuid.uuid4().hex[:8]
    return {"leave_id": leave_id, "answer": f"已为你提交请假申请，编号 {leave_id}，等待审批。"}


def build_leave_graph():
    g = StateGraph(LeaveState)

    g.add_node("extract", extract_slots_node)
    g.add_node("validate", validate_node)
    g.add_node("need_info", need_info_node)
    g.add_node("confirm", confirm_node)
    g.add_node("create", create_leave_node)

    g.add_edge(START, "extract")
    g.add_edge("extract", "validate")

    g.add_conditional_edges(
        "validate",
        decide_next,
        {"need_info": "need_info", "confirm": "confirm"},
    )

    g.add_conditional_edges(
        "confirm",
        decide_confirm,
        {"create": "create", "end": END},
    )

    g.add_edge("need_info", END)
    g.add_edge("create", END)

    return g.compile()

