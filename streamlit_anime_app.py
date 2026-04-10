"""
개인화 애니 추천 Streamlit (seed.yaml v1.1).
실행: streamlit run streamlit_anime_app.py
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

from secrets_util import get_openai_api_key

# 프로젝트 루트에서 모듈 로드
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st

from anilist_client import fetch_cover_urls_by_ids
from anime_graph import react_summary_text, run_pipeline


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_poster_urls(ids_tuple: tuple[int, ...]) -> dict[int, str]:
    """AniList 배치 조회 결과 캐시(동일 id 재방문 시 API 호출 감소)."""
    if not ids_tuple:
        return {}
    return fetch_cover_urls_by_ids(list(ids_tuple))


def _collect_anilist_ids(recs: list[dict]) -> tuple[int, ...]:
    out: list[int] = []
    for r in recs:
        aid = r.get("anilist_id")
        if aid is None:
            continue
        try:
            i = int(aid)
            if i > 0:
                out.append(i)
        except (TypeError, ValueError):
            continue
    return tuple(sorted(set(out)))


def _render_recommendation_cards(recs: list[dict]) -> None:
    """2열 그리드 + 포스터(AniList cover) + 테두리 컨테이너."""
    n = len(recs)
    if n == 0:
        st.warning("추천 항목이 없습니다.")
        return

    id_tuple = _collect_anilist_ids(recs)
    url_map = _cached_poster_urls(id_tuple) if id_tuple else {}

    cols_per_row = 2
    for row_start in range(0, n, cols_per_row):
        chunk = recs[row_start : row_start + cols_per_row]
        cols = st.columns(len(chunk))
        for col, r, idx in zip(cols, chunk, range(row_start + 1, row_start + 1 + len(chunk))):
            with col:
                with st.container(border=True):
                    title = r.get("title") or "(제목 없음)"
                    aid = r.get("anilist_id")
                    poster: str | None = None
                    if aid is not None:
                        try:
                            poster = url_map.get(int(aid))
                        except (TypeError, ValueError):
                            poster = None

                    img_col, txt_col = st.columns([1, 1.35])
                    with img_col:
                        if poster:
                            st.image(poster, use_container_width=True)
                        else:
                            st.caption(
                                "포스터 없음 (AniList id 없음)"
                                if aid is None
                                else "포스터 없음 (조회 실패)"
                            )
                    with txt_col:
                        st.markdown(f"**{idx}. {title}**")
                        st.write(r.get("rationale_ko", ""))
                        if aid:
                            st.link_button(
                                "AniList에서 보기",
                                f"https://anilist.co/anime/{aid}",
                                use_container_width=True,
                            )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="애니 맞춤 추천", page_icon="🎬", layout="wide")
    st.title("개인화 애니메이션 추천")
    st.caption("LangGraph ReAct · 장르·분위기 분기 · 기본 3작 + 옵션 0~2작 · gpt-5-mini")

    if not get_openai_api_key():
        st.error(
            "OPENAI_API_KEY가 없습니다. 로컬은 프로젝트 루트 `.env`, "
            "Streamlit Cloud는 **Settings → Secrets**에 `OPENAI_API_KEY`를 설정하세요."
        )
        st.stop()

    with st.sidebar:
        st.header("취향 입력")
        with st.form("preference_form", clear_on_submit=False):
            genre = st.selectbox(
                "장르(대표)",
                [
                    "액션/판타지",
                    "일상/힐링",
                    "SF/로봇",
                    "로맨스",
                    "스릴러/추리",
                    "기타",
                ],
                index=0,
            )
            mood = st.selectbox(
                "분위기",
                [
                    "밝고 가벼움",
                    "어둡고 무거움",
                    "감성·잔잔",
                    "긴장·몰입",
                    "유머·개그",
                    "기타",
                ],
                index=0,
            )
            free_text = st.text_area(
                "자유 입력 (좋아하는 작·싫은 것 등)",
                height=120,
                placeholder="예: 12화 내외, 스포는 최소로…",
            )
            extra = st.radio(
                "추가 추천",
                [0, 1, 2],
                horizontal=True,
                format_func=lambda x: f"+{x}작 (총 {3 + x}작)",
            )
            run = st.form_submit_button("추천 받기", type="primary", use_container_width=True)

    if not run:
        st.info("사이드바에서 취향을 입력한 뒤 **추천 받기**를 누르세요.")
        return

    with st.status("에이전트 실행 중… (ReAct · 도구 호출)", expanded=True) as status:
        st.caption("LangGraph 파이프라인 · 외부 검색·AniList·Jikan 호출이 포함될 수 있습니다.")
        try:
            result = run_pipeline(
                genre=genre,
                mood=mood,
                free_text=free_text,
                extra_recommendation_slots=extra,
            )
            status.update(label="추천 생성 완료", state="complete")
        except Exception as e:
            status.update(label="오류 발생", state="error")
            st.error(f"실행 오류: {e}")
            return

    st.toast("추천이 준비되었습니다.", icon="🎬")
    st.success("아래 탭에서 결과와 조사 과정을 확인할 수 있습니다.")

    recs = result.get("recommendations") or []
    prof = result.get("user_profile") or {}
    branch = result.get("branch_state") or {}
    trace = result.get("tool_trace") or []
    route = branch.get("route", "?")
    n_target = prof.get("total_count", len(recs))
    match_ok = len(recs) == n_target

    st.divider()
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "추천 작품 수",
            f"{len(recs)}작",
            delta=None if match_ok else "목표와 불일치",
            help=f"seed 기준 목표는 {n_target}작입니다.",
        )
    with m2:
        st.metric("도구 이벤트 수", len(trace), help="tool_call 및 tool_result 항목 수")
    with m3:
        st.metric("추가 슬롯", f"+{extra}작 (총 {3 + extra}작)")

    st.write("**분기**")
    bcol1, bcol2, _ = st.columns([2, 2, 6])
    with bcol1:
        st.badge(branch.get("branch_label", "") or "—")
    with bcol2:
        st.badge(f"route={route}")

    tab_rec, tab_debug = st.tabs(["추천", "조사 과정 · 디버그"])

    with tab_rec:
        _render_recommendation_cards(recs)
        payload = json.dumps(recs, ensure_ascii=False, indent=2)
        st.download_button(
            label="추천 목록 JSON 다운로드",
            data=payload.encode("utf-8"),
            file_name="anime_recommendations.json",
            mime="application/json",
        )

    with tab_debug:
        st.subheader("프로필 · 분기")
        c1, c2 = st.columns(2)
        with c1:
            st.json(prof)
        with c2:
            st.json(branch)

        st.subheader("도구 호출 순서")
        if trace:
            for i, ev in enumerate(trace, 1):
                kind = ev.get("type", "?")
                name = ev.get("name", "")
                with st.expander(f"{i}. [{kind}] {name}", expanded=False):
                    if kind == "tool_call":
                        st.json(ev.get("args", {}))
                    else:
                        st.text(ev.get("preview", "")[:4000])
        else:
            st.caption("도구 흔적이 없습니다.")

        st.subheader("ReAct 요약 텍스트")
        summary = react_summary_text(result)
        if summary:
            st.code(summary[:8000], language=None)
        else:
            st.caption("요약이 비어 있습니다.")

        st.subheader("통합 JSON (원시)")
        with st.expander("user_profile + branch + tool_trace", expanded=False):
            st.json({"user_profile": prof, "branch_state": branch, "tool_trace": trace})

        try:
            df = pd.DataFrame(recs)
            st.subheader("추천 표")
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
