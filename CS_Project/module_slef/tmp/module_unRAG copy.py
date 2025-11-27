# module_direct.py

from langchain.chat_models import init_chat_model
from typing import List, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import os
import time
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("請在 .env 設定 MISTRAL_API_KEY")

# 初始化 LLM
llm = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")

# Prompt 模板 (可依需求微調)
prompt = PromptTemplate.from_template("""# {title}
## System
{system}
## Question
{question}
## Answer
""")

# 定義資料結構
class State(TypedDict):
    title: str
    system: str
    question: str
    answer: str

# 全域最新相關連結 (direct 模型通常沒有，但保留結構)
latest_related_links: List[dict] = []

def get_latest_links() -> List[dict]:
    """
    回傳最近一次回答的相關連結
    direct 模型通常沒有 RAG 連結，因此回傳空列表
    """
    global latest_related_links
    return latest_related_links

def ask_direct(question: str) -> str:
    """
    直接使用 LLM 回答問題，不做 RAG 檢索
    """
    input_state = {
        "title": "NUU 直接問答",
        "system": "你是一個大學生助理，負責回答聯合大學網站相關問題。",
        "question": question
    }

    # 構建 prompt
    prompt_input = {
        "title": input_state["title"],
        "system": input_state["system"],
        "question": input_state["question"]
    }
    messages = prompt.invoke(prompt_input)
    response = llm.invoke(messages)
    # direct 模型沒有 RAG 連結
    global latest_related_links
    latest_related_links = []

    clean_answer = response.content.replace("\\u000A", "\n")
    return clean_answer
