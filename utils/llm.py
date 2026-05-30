import os

from openai import OpenAI

from dotenv import load_dotenv


load_dotenv()


def get_client():

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "OPENROUTER_API_KEY not found in .env"
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


client = get_client()

DEFAULT_MODEL = (
    "qwen/qwen3-32b"
)