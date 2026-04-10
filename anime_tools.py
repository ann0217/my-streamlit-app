"""
seed.yaml 대응 ReAct 도구: 웹 검색(DuckDuckGo), AniList, Jikan.
gpt-5-mini 도구 스키마만 사용 (temperature 등 없음).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from duckduckgo_search import DDGS
from langchain_core.tools import tool

from anilist_client import search_anime_summary_for_llm


@tool
def web_search_anime(query: str) -> str:
    """애니메이션 작품·감상평·리뷰 등 웹에서 짧은 정보를 검색합니다. 검색어는 영어 또는 일본어 제목을 포함하면 좋습니다."""
    q = (query or "").strip()
    if not q:
        return "검색어가 비어 있습니다."
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.text(q, max_results=6))
    except Exception as e:
        return f"[웹 검색 오류] {e}"
    if not rows:
        return f"[웹 검색] '{q}'에 대한 결과가 없습니다."
    lines = [f"[웹 검색: {q}]"]
    for i, r in enumerate(rows, 1):
        title = (r.get("title") or "")[:120]
        body = (r.get("body") or "")[:240]
        href = r.get("href") or ""
        lines.append(f"{i}. {title}\n   {body}\n   {href}")
    return "\n".join(lines)


@tool
def anilist_search_anime(query: str) -> str:
    """AniList 데이터베이스에서 애니메이션을 검색합니다. 제목·키워드·장르 힌트로 후보와 점수·연도를 얻습니다."""
    q = (query or "").strip()
    if not q:
        return "검색어가 비어 있습니다."
    return search_anime_summary_for_llm(q, per_page=8)


def _jikan_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"https://api.jikan.moe/v4{path}"
    with httpx.Client(timeout=25.0) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


@tool
def jikan_search_anime(query: str) -> str:
    """MyAnimeList(Jikan API)에서 애니메이션을 검색합니다. 후보 작품 목록과 점수·화수를 얻습니다."""
    q = (query or "").strip()
    if not q:
        return "검색어가 비어 있습니다."
    try:
        data = _jikan_get("/anime", {"q": q, "limit": 6, "order_by": "popularity"})
    except Exception as e:
        return f"[Jikan 검색 오류] {e}"
    items = (data.get("data") or []) if isinstance(data, dict) else []
    if not items:
        return f"[Jikan] '{q}'에 대한 결과가 없습니다."
    lines: list[str] = [f"[Jikan 검색: {q}]"]
    for it in items:
        mid = it.get("mal_id")
        title = (it.get("title") or it.get("title_english") or "")[:100]
        score = it.get("score")
        eps = it.get("episodes")
        year = it.get("year")
        genres = [g.get("name") for g in (it.get("genres") or [])][:5]
        lines.append(f"- mal_id={mid} | {title} | 점수={score} | 화수={eps} | 연도={year} | 장르={', '.join(genres)}")
    return "\n".join(lines)


ANIME_TOOLS = [web_search_anime, anilist_search_anime, jikan_search_anime]
