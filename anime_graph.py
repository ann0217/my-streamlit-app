"""
seed.yaml v1.1 — LangGraph: parse_profile → 장르·분기 3갈래 → ReAct → finalize.
"""

from __future__ import annotations

import json
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from anilist_client import search_first_media_id
from anime_tools import ANIME_TOOLS
from secrets_util import require_openai_api_key


class RecItem(BaseModel):
    title: str = Field(description="작품 제목")
    rationale_ko: str = Field(description="한국어 추천 근거, 스포일러 금지")
    anilist_id: int | None = Field(default=None, description="알 수 있으면 AniList id")


class RecOutput(BaseModel):
    items: list[RecItem] = Field(description="추천 작품 목록")


class GraphState(TypedDict, total=False):
    genre: str
    mood: str
    free_text: str
    extra_recommendation_slots: int
    user_profile: dict
    branch_state: dict
    tool_trace: list
    react_messages: list
    recommendations: list


BRANCH_HINTS: dict[str, str] = {
    "action": "액션·전투·SF·템포가 빠른 작품 후보를 우선 조사하세요.",
    "healing": "일상·감성·힐링·잔잔한 분위기 작품 후보를 우선 조사하세요.",
    "general": "장르와 분위기에 맞는 다양한 후보를 조사하세요.",
}


def _detect_route(genre: str, mood: str, free_text: str) -> Literal["action", "healing", "general"]:
    combined = f"{genre} {mood} {free_text}".lower()
    if any(
        k in combined
        for k in (
            "액션",
            "전투",
            "배틀",
            "action",
            "battle",
            "판타지",
            "sf",
            "로봇",
            "mecha",
        )
    ):
        return "action"
    if any(
        k in combined
        for k in (
            "일상",
            "힐링",
            "치유",
            "편안",
            "slice",
            "iyashikei",
            "감성",
            "잔잔",
        )
    ):
        return "healing"
    return "general"


def parse_profile(state: GraphState) -> GraphState:
    genre = (state.get("genre") or "").strip() or "미지정"
    mood = (state.get("mood") or "").strip() or "미지정"
    free_text = (state.get("free_text") or "").strip()
    extra = int(state.get("extra_recommendation_slots") or 0)
    extra = max(0, min(2, extra))
    route = _detect_route(genre, mood, free_text)
    total = 3 + extra
    user_profile = {
        "genre": genre,
        "mood": mood,
        "free_text": free_text,
        "total_count": total,
    }
    branch_state: dict = {"route": route, "hint": BRANCH_HINTS[route]}
    return {
        "extra_recommendation_slots": extra,
        "user_profile": user_profile,
        "branch_state": branch_state,
        "tool_trace": [],
    }


def _route_edge(state: GraphState) -> Literal["branch_action", "branch_healing", "branch_general"]:
    r = (state.get("branch_state") or {}).get("route", "general")
    if r == "action":
        return "branch_action"
    if r == "healing":
        return "branch_healing"
    return "branch_general"


def branch_action(state: GraphState) -> GraphState:
    bs = dict(state.get("branch_state") or {})
    bs["branch_label"] = "액션·전투·SF 계열 분기"
    return {"branch_state": bs}


def branch_healing(state: GraphState) -> GraphState:
    bs = dict(state.get("branch_state") or {})
    bs["branch_label"] = "일상·힐링·감성 계열 분기"
    return {"branch_state": bs}


def branch_general(state: GraphState) -> GraphState:
    bs = dict(state.get("branch_state") or {})
    bs["branch_label"] = "일반 분기"
    return {"branch_state": bs}


REACT_SYSTEM = """당신은 애니메이션 추천을 위한 조사 보조입니다.
반드시 도구(web_search_anime, anilist_search_anime, jikan_search_anime)를 적절히 사용해 사실·메타데이터를 수집하세요.
특정 작품의 반전·결말 등 스포일러는 인용하지 마세요.
수집이 끝나면 후보 작품과 근거를 한국어로 간결히 정리하세요."""


def _extract_tool_trace(messages: list) -> list[dict]:
    trace: list[dict] = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                trace.append(
                    {
                        "type": "tool_call",
                        "name": tc.get("name"),
                        "args": tc.get("args"),
                    }
                )
        typ = getattr(m, "type", None)
        if typ == "tool":
            trace.append(
                {
                    "type": "tool_result",
                    "name": getattr(m, "name", ""),
                    "preview": (str(m.content)[:1200] if m.content else ""),
                }
            )
    return trace


def _last_substantial_ai_text(messages: list) -> str:
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        c = m.content
        if isinstance(c, str) and c.strip() and not getattr(m, "tool_calls", None):
            return c.strip()
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str):
                return c.strip()
    return ""


def _make_react_agent():
    llm = ChatOpenAI(model="gpt-5-mini", api_key=require_openai_api_key())
    return create_react_agent(llm, ANIME_TOOLS, prompt=REACT_SYSTEM, version="v2")


_react_agent = None


def get_react_agent():
    global _react_agent
    if _react_agent is None:
        _react_agent = _make_react_agent()
    return _react_agent


def run_react(state: GraphState) -> GraphState:
    profile = state["user_profile"]
    bs = state["branch_state"]
    n = profile["total_count"]
    hint = bs.get("hint", "")
    label = bs.get("branch_label", "")
    task = f"""사용자 프로필:
- 장르: {profile['genre']}
- 분위기: {profile['mood']}
- 자유 입력: {profile.get('free_text', '') or '(없음)'}
- 최종 추천 개수(참고): {n}개 (기본 3 + 추가 {n - 3})

그래프 분기: {label}
분기 힌트: {hint}

위에 맞는 작품 후보를 도구로 조사한 뒤, 후보와 이유를 한국어로 정리하세요."""
    agent = get_react_agent()
    out = agent.invoke(
        {"messages": [HumanMessage(content=task)]},
        config={"recursion_limit": 50},
    )
    msgs = list(out["messages"])
    trace = _extract_tool_trace(msgs)
    return {"react_messages": msgs, "tool_trace": trace}


def _enrich_missing_anilist_ids(recs: list[dict]) -> list[dict]:
    """anilist_id가 비어 있으면 제목으로 AniList 검색해 id 보강."""
    out: list[dict] = []
    for r in recs:
        aid = r.get("anilist_id")
        ok = False
        if aid is not None:
            try:
                ok = int(aid) > 0
            except (TypeError, ValueError):
                ok = False
        if ok:
            out.append(r)
            continue
        tid = search_first_media_id((r.get("title") or "").strip())
        if tid is not None:
            out.append({**r, "anilist_id": tid})
        else:
            out.append(r)
    return out


def finalize(state: GraphState) -> GraphState:
    n = state["user_profile"]["total_count"]
    react_text = _last_substantial_ai_text(state.get("react_messages") or [])
    llm = ChatOpenAI(model="gpt-5-mini", api_key=require_openai_api_key())
    structured = llm.with_structured_output(RecOutput)

    sys_msg = f"""당신은 최종 추천 편집자입니다.
조사 결과를 바탕으로 정확히 {n}개의 작품만 추천 목록에 넣으세요.
rationale_ko에는 주요 반전·결말 등 스포일러를 쓰지 마세요.
일본 방송·판권 등 법적 논의는 하지 마세요.
가능한 한 각 항목에 anilist_id(AniList의 작품 숫자 id)를 채우세요. 조사 요약에 'id=숫자' 또는 AniList 도구 결과가 있으면 그 id를 사용합니다. 정말 알 수 없을 때만 null입니다."""

    human = f"""사용자 프로필(JSON): {json.dumps(state['user_profile'], ensure_ascii=False)}

에이전트 조사 요약:
{react_text[:12000]}

정확히 {n}개 항목의 items 배열을 채우세요."""

    out: RecOutput | None = None
    for _ in range(2):
        out = structured.invoke([SystemMessage(content=sys_msg), HumanMessage(content=human)])
        if out and len(out.items) == n:
            break
        human += f"\n\n(재시도: 반드시 정확히 {n}개 항목이어야 합니다.)"

    items = out.items if out else []
    recs = []
    for it in items[:n]:
        d = it.model_dump()
        recs.append(
            {
                "title": d["title"],
                "rationale_ko": d["rationale_ko"],
                "anilist_id": d.get("anilist_id"),
            }
        )
    while len(recs) < n:
        recs.append(
            {
                "title": "추천 보강 필요",
                "rationale_ko": "조사 결과가 부족하여 수동으로 후보를 늘려 주세요.",
                "anilist_id": None,
            }
        )
    recs = _enrich_missing_anilist_ids(recs[:n])
    return {"recommendations": recs}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("parse_profile", parse_profile)
    g.add_node("branch_action", branch_action)
    g.add_node("branch_healing", branch_healing)
    g.add_node("branch_general", branch_general)
    g.add_node("run_react", run_react)
    g.add_node("finalize", finalize)

    g.add_edge(START, "parse_profile")
    g.add_conditional_edges(
        "parse_profile",
        _route_edge,
        {
            "branch_action": "branch_action",
            "branch_healing": "branch_healing",
            "branch_general": "branch_general",
        },
    )
    g.add_edge("branch_action", "run_react")
    g.add_edge("branch_healing", "run_react")
    g.add_edge("branch_general", "run_react")
    g.add_edge("run_react", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def react_summary_text(state: GraphState) -> str:
    """Streamlit 등에서 조사 단계 최종 요약 텍스트로 사용."""
    return _last_substantial_ai_text(state.get("react_messages") or [])


def run_pipeline(
    *,
    genre: str,
    mood: str,
    free_text: str = "",
    extra_recommendation_slots: int = 0,
) -> GraphState:
    graph = get_compiled_graph()
    return graph.invoke(
        {
            "genre": genre,
            "mood": mood,
            "free_text": free_text,
            "extra_recommendation_slots": extra_recommendation_slots,
        }
    )
