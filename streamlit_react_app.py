"""
ReAct 에이전트 Streamlit 데모 (실습 §8).
실행: streamlit run streamlit_react_app.py
스모크 테스트: python streamlit_react_app.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from secrets_util import get_openai_api_key, require_openai_api_key
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# --- Tool 4개 (요구: 3개 이상) ---


@tool
def translate_line(text: str, target_language: str) -> str:
    """한 줄짜리 텍스트를 target_language(예: English, Japanese, Korean)로 번역한 것처럼 시뮬레이션합니다."""
    lang = (target_language or "English").strip()
    return f"[mock 번역 → {lang}] {text}"


@tool
def get_weather_mock(city: str) -> str:
    """도시 이름을 받아 가짜 날씨 문자열을 반환합니다(실습용)."""
    c = (city or "").strip() or "서울"
    return f"{c}: 맑음, 기온 22°C, 습도 55% (mock 데이터)"


@tool
def safe_calculate(expression: str) -> str:
    """수학 계산을 수행합니다. 기본 연산 위주(노트북 예제와 동일한 제한)."""
    expr = (expression or "").strip()
    if not expr:
        return "빈 수식입니다."
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        val = eval(expr, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"{expression} = {val}"
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def run_python_one_liner(code: str) -> str:
    """한 줄 Python 표현식을 실행합니다(예: len('abc')). 제한된 안전 환경만 사용합니다."""
    c = (code or "").strip()
    if not c:
        return "빈 코드입니다."
    if "\n" in c:
        return "한 줄만 허용됩니다."
    safe_globals = {
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "round": round,
    }
    try:
        out = eval(c, {"__builtins__": {}}, safe_globals)  # noqa: S307
        return repr(out)
    except Exception as e:
        return f"실행 오류: {e}"


SYSTEM_PROMPT = """당신은 '실습용 ReAct 비서'입니다.
사용자 질문에 답하기 위해 번역·날씨(모의)·계산·짧은 코드 실행 도구를 적절히 선택하세요.
도구 결과를 바탕으로 한국어로 간결하게 답하세요.
이전 대화 내용을 참고해 후속 질문에 답할 수 있습니다."""


class ReActChatSession:
    """대화 히스토리를 유지하며 create_react_agent 그래프를 호출합니다."""

    def __init__(self) -> None:
        load_dotenv()
        self._llm = ChatOpenAI(model="gpt-5-mini", api_key=require_openai_api_key())
        tools = [
            translate_line,
            get_weather_mock,
            safe_calculate,
            run_python_one_liner,
        ]
        self._agent = create_react_agent(
            self._llm,
            tools,
            prompt=SYSTEM_PROMPT,
        )
        self._messages: list = []

    @property
    def messages(self) -> list:
        return self._messages

    def send(self, user_text: str) -> str:
        self._messages.append(HumanMessage(content=user_text))
        result = self._agent.invoke({"messages": self._messages})
        self._messages = list(result["messages"])
        return _extract_final_text(self._messages)

    def clear(self) -> None:
        self._messages = []


def _extract_final_text(messages: list) -> str:
    last = messages[-1]
    content = getattr(last, "content", "")
    if isinstance(content, str) and content.strip():
        return content
    return str(content)


def run_smoke_test() -> None:
    """최소 3턴 대화로 에이전트·도구·메모리를 점검합니다 (API 호출 발생)."""
    load_dotenv()
    if not get_openai_api_key():
        print("SKIP: OPENAI_API_KEY 없음")
        sys.exit(0)
    session = ReActChatSession()
    turns = [
        "서울 날씨 어때? 도구 써서 알려줘.",
        "방금 어느 도시 날씨 물어봤지? 한 단어로.",
        "2**12 는多少? safe_calculate 도구로 계산해줘.",
    ]
    for i, q in enumerate(turns, 1):
        print(f"\n--- 턴 {i} ---\n사용자: {q}")
        reply = session.send(q)
        print(f"비서: {reply[:800]}{'...' if len(reply) > 800 else ''}")
    print("\n[smoke] 완료 (3턴).")


def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="ReAct 비서", page_icon="🤖", layout="centered")
    st.title("나만의 ReAct 에이전트")
    st.caption("도구: 번역(mock)·날씨(mock)·안전 계산·한 줄 Python | 모델: gpt-5-mini")

    if "chat" not in st.session_state:
        try:
            st.session_state.chat = ReActChatSession()
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    if st.button("대화 초기화"):
        st.session_state.chat.clear()
        st.rerun()

    for m in st.session_state.chat.messages:
        role = getattr(m, "type", "")
        if role == "human":
            with st.chat_message("user"):
                st.write(m.content)
        elif role == "ai":
            with st.chat_message("assistant"):
                c = m.content
                st.write(c if isinstance(c, str) else str(c))
                if getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        st.caption(f"도구 호출: {tc['name']}({tc['args']})")
        elif role == "tool":
            with st.chat_message("assistant"):
                st.caption(f"도구 결과 ({getattr(m, 'name', '?')})")
                st.write(m.content)

    prompt = st.chat_input("메시지를 입력하세요…")
    if prompt:
        with st.spinner("생각 중…"):
            try:
                st.session_state.chat.send(prompt)
            except Exception as e:
                st.error(str(e))
        st.rerun()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Streamlit 없이 3턴 API 스모크 테스트",
    )
    args, _ = parser.parse_known_args()
    if args.smoke:
        run_smoke_test()
    else:
        run_streamlit()
