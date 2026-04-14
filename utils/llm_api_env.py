"""
DeepSeek（OpenAI 兼容）HTTP API 配置。

优先级（从高到低）：
1. 环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY
2. 本仓库中的 utils/deepseek_local.py（见 deepseek_local.example.py，已 .gitignore）
3. 下方 DEFAULT_DEEPSEEK_API_KEY（填一次即可，无需每次 export；请勿提交到公开仓库）
"""
from __future__ import annotations

import os
from typing import Optional

DEFAULT_DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/v1/chat/completions"

# ---------------------------------------------------------------------------
# 若不想每次在终端 export，可在此填入 sk-...（仅本机使用；公开仓库提交前请清空）
# ---------------------------------------------------------------------------
DEFAULT_DEEPSEEK_API_KEY: str = ""


def _key_from_local_file() -> str:
    try:
        from utils.deepseek_local import DEEPSEEK_API_KEY as k
    except ImportError:
        return ""
    return (k or "").strip()


def get_deepseek_api_key() -> Optional[str]:
    for candidate in (
        os.getenv("DEEPSEEK_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        _key_from_local_file(),
        DEFAULT_DEEPSEEK_API_KEY.strip() or None,
    ):
        if candidate:
            return candidate.strip()
    return None


def get_deepseek_chat_completions_url() -> str:
    return os.getenv("DEEPSEEK_API_URL", DEFAULT_DEEPSEEK_CHAT_COMPLETIONS_URL)
