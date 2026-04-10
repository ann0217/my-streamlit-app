"""
OPENAI_API_KEY 로딩: 로컬 .env(os.environ) 우선, 이어서 Streamlit Secrets.

Streamlit Cloud에서는 Secrets에 OPENAI_API_KEY를 넣으면 됨.
로컬은 .env 또는 환경 변수.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


def get_openai_api_key() -> str | None:
    """비어 있지 않은 키 문자열 또는 None."""
    load_dotenv()
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if key:
        return key

    try:
        import streamlit as st  # type: ignore

        sec = getattr(st, "secrets", None)
        if sec is None:
            return None
        if "OPENAI_API_KEY" not in sec:
            return None
        raw = sec["OPENAI_API_KEY"]
        out = (str(raw) if raw is not None else "").strip()
        return out or None
    except Exception:
        # Streamlit 미실행 컨텍스트 등
        return None


def require_openai_api_key() -> str:
    k = get_openai_api_key()
    if not k:
        raise RuntimeError(
            "OPENAI_API_KEY가 없습니다. 로컬은 .env, Streamlit Cloud는 Secrets에 설정하세요."
        )
    return k
