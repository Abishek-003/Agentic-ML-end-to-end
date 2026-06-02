from openai import OpenAI
import streamlit as st

def get_client():
    api_key = st.secrets["OPENROUTER_API_KEY"]

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

client = get_client()
DEFAULT_MODEL = "qwen/qwen3-32b"