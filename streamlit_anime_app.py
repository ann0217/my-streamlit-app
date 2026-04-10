"""
개인화 애니 추천 Streamlit (seed.yaml v1.1).
실행: streamlit run streamlit_anime_app.py
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

# 프로젝트 루트에서 모듈 로드
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

from anime_graph import react_summary_text, run_pipeline


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="애니 맞춤 추천", page_icon="🎬", layout="wide")
    st.title("개인화 애니메이션 추천")
    st.caption("LangGraph ReAct · 장르·분위기 분기 · 기본 3작 + 옵션 0~2작 · gpt-5-mini")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("`.env`에 `OPENAI_API_KEY`를 설정하세요.")
        st.stop()

    with st.sidebar:
        st.header("취향 입력")
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
            ["밝고 가벼움", "어둡고 무거움", "감성·잔잔", "긴장·몰입", "유머·개그", "기타"],
            index=0,
        )
        free_text = st.text_area("자유 입력 (좋아하는 작·싫은 것 등)", height=120, placeholder="예: 12화 내외, 스포는 최소로…")
        extra = st.radio(
            "추가 추천",
            [0, 1, 2],
            horizontal=True,
            format_func=lambda x: f"+{x}작 (총 {3 + x}작)",
        )
        run = st.button("추천 받기", type="primary")

    if not run:
        st.info("사이드바에서 취향을 입력한 뒤 **추천 받기**를 누르세요.")
        return

    with st.spinner("그래프 실행 중 (ReAct + 도구 호출, 시간이 걸릴 수 있습니다)…"):
        try:
            result = run_pipeline(
                genre=genre,
                mood=mood,
                free_text=free_text,
                extra_recommendation_slots=extra,
            )
        except Exception as e:
            st.error(f"실행 오류: {e}")
            return

    recs = result.get("recommendations") or []
    prof = result.get("user_profile") or {}
    branch = result.get("branch_state") or {}
    trace = result.get("tool_trace") or []

    st.subheader("추천 결과")
    st.write(
        f"**분기:** {branch.get('branch_label', '')} (`route={branch.get('route')}`) · "
        f"**개수:** {len(recs)}작 (목표 {prof.get('total_count', '?')}작)"
    )

    for i, r in enumerate(recs, 1):
        with st.container():
            st.markdown(f"#### {i}. {r.get('title', '')}")
            st.write(r.get("rationale_ko", ""))
            aid = r.get("anilist_id")
            if aid:
                st.caption(f"AniList id: {aid}")

    with st.expander("중간 단계 · 도구 흔적 (접기/펼치기)", expanded=False):
        st.json(
            {
                "user_profile": prof,
                "branch_state": branch,
                "tool_trace": trace,
            }
        )
        summary = react_summary_text(result)
        if summary:
            st.markdown("**ReAct 최종 요약 (조사 단계)**")
            st.code(summary[:8000], language=None)


if __name__ == "__main__":
    main()
