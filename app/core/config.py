import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY_FLASH = os.getenv("OPENAI_API_KEY_FLASH")

    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_API_BASE_FLASH = os.getenv("OPENAI_API_BASE_FLASH")

    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
    LLM_FLASH_MODEL_NAME = os.getenv("LLM_FLASH_MODEL_NAME")

    USE_RESPONSES_API = os.getenv("USE_RESPONSES_API")
    USE_RESPONSES_API_FLASH = os.getenv("USE_RESPONSES_API_FLASH")

    # E2B_API_KEY = os.getenv("E2B_API_KEY")
    PPIO_API_KEY = os.getenv("PPIO_API_KEY")
    PPIO_TEMPLATE = os.getenv("PPIO_TEMPLATE")

settings = Settings()