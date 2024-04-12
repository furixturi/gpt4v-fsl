# https://qiita.com/cygkichi/items/ea1d0166afeec189a04f

from dotenv import load_dotenv

load_dotenv()

import os
from openai import AzureOpenAI


def init():
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY_AU"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_AU"),
    )


model = os.getenv("AZURE_OPENAI_MODEL_GPT4_V")

