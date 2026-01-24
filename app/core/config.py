import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
    LLM_FLASH_MODEL_NAME = os.getenv("LLM_FLASH_MODEL_NAME")

    # E2B_API_KEY = os.getenv("E2B_API_KEY")
    PPIO_API_KEY = os.getenv("PPIO_API_KEY")

settings = Settings()