import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY_FLASH = os.getenv("OPENAI_API_KEY_FLASH")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_API_BASE_FLASH = os.getenv("OPENAI_API_BASE_FLASH")

    # Model names
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
    LLM_FLASH_MODEL_NAME = os.getenv("LLM_FLASH_MODEL_NAME")

    # Responses API
    USE_RESPONSES_API = os.getenv("USE_RESPONSES_API")
    USE_RESPONSES_API_FLASH = os.getenv("USE_RESPONSES_API_FLASH")

    # Local modeling workspace
    AGENT_WORKSPACE_DIR = os.getenv("AGENT_WORKSPACE_DIR")
    AGENT_WORKSPACE_SESSIONS_DIR = os.getenv("AGENT_WORKSPACE_SESSIONS_DIR")
    AGENT_WORKSPACE_PYTHON = os.getenv("AGENT_WORKSPACE_PYTHON")


settings = Settings()
