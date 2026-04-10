"""
AniList GraphQL API 클라이언트 (공개 쿼리, API 키 불필요).
문서: https://docs.anilist.co/guide/graphql/
엔드포인트: POST https://graphql.anilist.co
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import httpx

ANILIST_URL = "https://graphql.anilist.co"
DEFAULT_TIMEOUT = 30.0

QUERY_SEARCH_ANIME = """
query ($search: String, $page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
    }
    media(search: $search, type: ANIME, sort: SEARCH_MATCH) {
      id
      title {
        romaji
        english
        native
      }
      coverImage {
        large
        medium
        extraLarge
      }
      averageScore
      genres
      episodes
      seasonYear
      format
      status
    }
  }
}
"""

QUERY_MEDIA_BY_ID = """
query ($id: Int) {
  Media(id: $id, type: ANIME) {
    id
    title {
      romaji
      english
      native
    }
    description(asHtml: false)
    averageScore
    genres
    episodes
    seasonYear
    format
    status
    siteUrl
    coverImage {
      large
      medium
      extraLarge
    }
  }
}
"""

QUERY_COVERS_BY_IDS = """
query ($ids: [Int], $perPage: Int) {
  Page(page: 1, perPage: $perPage) {
    media(id_in: $ids, type: ANIME) {
      id
      coverImage {
        large
        medium
        extraLarge
      }
    }
  }
}
"""


class AniListError(RuntimeError):
    """GraphQL errors 또는 HTTP 오류."""


def _graphql(
    query: str,
    variables: dict[str, Any] | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    payload = {"query": query, "variables": variables or {}}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(ANILIST_URL, json=payload, headers=headers)
        r.raise_for_status()
        body = r.json()
    errs = body.get("errors")
    if errs:
        msg = json.dumps(errs, ensure_ascii=False)[:2000]
        raise AniListError(f"GraphQL errors: {msg}")
    return body


def search_anime(
    search: str,
    *,
    page: int = 1,
    per_page: int = 10,
) -> dict[str, Any]:
    """
    제목/키워드로 애니메이션 검색. 원본 `data` dict 반환.
    """
    s = (search or "").strip()
    if not s:
        raise ValueError("search는 비어 있으면 안 됩니다.")
    data = _graphql(
        QUERY_SEARCH_ANIME,
        {"search": s, "page": page, "perPage": per_page},
    )
    return data.get("data") or {}


def _pick_cover_url(cover: dict[str, Any] | None) -> str | None:
    if not cover:
        return None
    return (
        cover.get("extraLarge")
        or cover.get("large")
        or cover.get("medium")
        or None
    )


def search_first_media_id(query: str) -> int | None:
    """
    제목/키워드 검색의 첫 번째 결과 Media id.
    추천 JSON에 anilist_id가 비어 있을 때 포스터·링크 보강용.
    """
    q = (query or "").strip()
    if len(q) < 2:
        return None
    try:
        data = search_anime(q, page=1, per_page=5)
    except (AniListError, httpx.HTTPError, ValueError):
        return None
    page = data.get("Page") or {}
    media_list = page.get("media") or []
    if not media_list:
        return None
    mid = media_list[0].get("id")
    if mid is None:
        return None
    try:
        return int(mid)
    except (TypeError, ValueError):
        return None


def _norm_q(s: str) -> str:
    return " ".join((s or "").strip().split())


def search_media_id_by_title_candidates(
    *,
    title: str = "",
    title_native: str | None = None,
    title_english: str | None = None,
) -> int | None:
    """
    일본어 원제 → 공식 영문 → 표시용 title 순으로 AniList 검색을 각각 시도, 첫 성공 id 반환.
    동일 문구(대소문자 무시)는 한 번만 검색한다.
    """
    seen_lower: set[str] = set()
    for raw in (title_native, title_english, title):
        q = _norm_q(raw or "")
        if len(q) < 2:
            continue
        key = q.casefold()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        mid = search_first_media_id(q)
        if mid is not None:
            return mid
    return None


def fetch_cover_urls_by_ids(media_ids: list[int]) -> dict[int, str]:
    """
    AniList id 여러 개에 대해 포스터 이미지 URL을 한 번에 조회.
    실패 시 빈 dict 또는 부분 결과를 반환한다.
    """
    ids = sorted({int(i) for i in media_ids if i and int(i) > 0})
    if not ids:
        return {}
    per_page = max(len(ids), 10)
    try:
        data = _graphql(
            QUERY_COVERS_BY_IDS,
            {"ids": ids, "perPage": per_page},
        )
    except (AniListError, httpx.HTTPError):
        return {}

    page = (data.get("data") or {}).get("Page") or {}
    media_list = page.get("media") or []
    out: dict[int, str] = {}
    for m in media_list:
        mid = m.get("id")
        url = _pick_cover_url(m.get("coverImage"))
        if mid is not None and url:
            out[int(mid)] = url
    return out


def get_anime_by_id(media_id: int) -> dict[str, Any]:
    """AniList `Media` id로 단일 작품 조회."""
    if media_id <= 0:
        raise ValueError("media_id는 양의 정수여야 합니다.")
    data = _graphql(QUERY_MEDIA_BY_ID, {"id": media_id})
    return data.get("data") or {}


def search_anime_summary_for_llm(
    search: str,
    *,
    page: int = 1,
    per_page: int = 8,
) -> str:
    """
    LangGraph / ReAct 도구에서 쓰기 좋은 짧은 텍스트 요약.
    """
    try:
        data = search_anime(search, page=page, per_page=per_page)
    except (AniListError, httpx.HTTPError, ValueError) as e:
        return f"[AniList 검색 오류] {e}"

    page_obj = data.get("Page") or {}
    media_list = page_obj.get("media") or []
    if not media_list:
        return f"[AniList] '{search}'에 대한 결과가 없습니다."

    lines: list[str] = [f"[AniList 검색: {search!r}, 표시 {len(media_list)}건]"]
    for m in media_list:
        tid = m.get("id")
        title = m.get("title") or {}
        romaji = title.get("romaji") or ""
        english = title.get("english") or ""
        score = m.get("averageScore")
        genres = m.get("genres") or []
        year = m.get("seasonYear")
        ep = m.get("episodes")
        label = english or romaji or str(tid)
        lines.append(
            f"- id={tid} | {label} | 점수={score} | 연도={year} | 화수={ep} | 장르={', '.join(genres[:5])}"
        )
    return "\n".join(lines)


def get_anime_summary_for_llm(media_id: int) -> str:
    """단일 작품 상세를 도구 결과 문자열로."""
    try:
        data = get_anime_by_id(media_id)
    except (AniListError, httpx.HTTPError, ValueError) as e:
        return f"[AniList 조회 오류] {e}"

    media = data.get("Media")
    if not media:
        return f"[AniList] id={media_id} 작품을 찾지 못했습니다."

    title = media.get("title") or {}
    romaji = title.get("romaji") or ""
    english = title.get("english") or ""
    desc = (media.get("description") or "")[:500]
    score = media.get("averageScore")
    genres = media.get("genres") or []
    url = media.get("siteUrl") or ""
    label = english or romaji
    return (
        f"[AniList] id={media.get('id')} {label}\n"
        f"점수={score} | 장르={', '.join(genres)}\n"
        f"URL={url}\n"
        f"요약(일부): {desc}"
    )


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except OSError:
            pass

    parser = argparse.ArgumentParser(description="AniList GraphQL 스모크 테스트")
    parser.add_argument("query", nargs="?", default="Cowboy Bebop", help="검색어")
    parser.add_argument("--id", type=int, default=None, help="작품 id로 단일 조회")
    args = parser.parse_args()

    if args.id is not None:
        print(get_anime_summary_for_llm(args.id))
        return

    print(search_anime_summary_for_llm(args.query))


if __name__ == "__main__":
    main()
