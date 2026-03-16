from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from openai import APIError

from app.core.config import settings


def create_llm(use_flash=False):
    """创建LLM实例（不带retry）
    """
    if use_flash:
        llm = ChatOpenAI(
            model=settings.LLM_FLASH_MODEL_NAME,
            temperature=0,
            api_key=settings.OPENAI_API_KEY_FLASH,
            base_url=settings.OPENAI_API_BASE_FLASH,
            use_responses_api=settings.USE_RESPONSES_API_FLASH,
            max_retries=5,
            timeout=120,
        )
    else:
        llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            use_responses_api=settings.USE_RESPONSES_API,
            max_retries=5,
            timeout=120,
        )
    return llm


def apply_retry(llm):
    """对LLM应用retry策略（在bind_tools之后调用）"""
    return llm.with_retry(
        stop_after_attempt=5,
        retry_if_exception_type=(APIError,),
        wait_exponential_jitter=True,
    )

