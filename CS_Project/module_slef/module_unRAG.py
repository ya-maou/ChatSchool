# module_direct.py

from langchain.chat_models import init_chat_model
from typing import List, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
import os
import time
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()
# MISTRAL_API_KEY 已移除，因為不再使用

# 檢查必要的 API Key
# 這裡只檢查 GOOGLE_API_KEY，因為它是 Gemini 備用所需的
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
if not GOOGLE_API_KEY:
    # 即使不使用 Gemini 作為主要模型，也應檢查 API Key 以確保備用模型可用性
    print("警告：未設定 GOOGLE_API_KEY。若 LM Studio 連線失敗，LLM 將無法初始化。")


# --- [新增] LM Studio 配置 ---
LMSTUDIO_MODEL = "google/gemma-3-4B" # 假設的模型
LMSTUDIO_URL = "http://192.168.98.34:1234/v1" # 假設的 LM Studio 地址

# --- [新增] init_LMStudio 函式 ---
def init_LMStudio(model: str, base_url: str, api_key: str = ".", configurable_fields: None = None, config_prefix: str | None = None, **kwargs) -> BaseChatModel:
    """使用 LangChain 連接至 LM Studio 的 OpenAI 相容 API"""
    return init_chat_model(model=model, base_url=base_url, configurable_fields=configurable_fields, config_prefix=config_prefix, model_provider="openai", api_key=api_key, **kwargs)

# --- LLM 初始化 (關鍵變動區) ---

# 1. 初始化 Gemini 作為最終備用模型
llm_gemini_backup = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")

# 2. 嘗試初始化 LM Studio 模型作為主要工作模型 (lmstudio_llm)
try:
    lmstudio_llm = init_LMStudio(model=LMSTUDIO_MODEL, base_url=LMSTUDIO_URL)
    print(f"[Direct 模組] 成功連接 LM Studio ({LMSTUDIO_URL}) 作為主要 LLM。")
except Exception as e:
    print(f"[Direct 模組] LM Studio 連接失敗，錯誤：{e}")
    print("[Direct 模組] 回退使用 Gemini 2.0 Flash 作為主要 LLM。")
    lmstudio_llm = llm_gemini_backup


# Prompt 模板 (保持不變)
prompt = PromptTemplate.from_template("""# {title}
## System
{system}
## Question
{question}
## Answer
""")

# 定義資料結構 (保持不變)
class State(TypedDict):
    title: str
    system: str
    question: str
    answer: str

# 全域最新相關連結 (保持不變)
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
    # === 關鍵修改點：將語言遵循指令改為英文 ===
    input_state = {
        "title": "NUU Direct Q&A",
        "system":(
            "You are a helpful university assistant specializing in National United University (NUU) information. "
            "**[STRICTEST RULE]**: **YOU MUST AND ONLY MUST** answer in the **EXACT SAME LANGUAGE** the user asked the question in. "
            "**Example**: If the question is in English, the answer must be in English. If the question is in Japanese, the answer must be in Japanese. "
            "**[Mandate]**: Please strictly adhere to this language rule. "
            "\n\n"
            "你是一個專業的大學生助理，負責回答國立聯合大學(National United University)校園相關問題。"
        ), 
        "question": question
    }

    # 構建 prompt
    prompt_input = {
        "title": input_state["title"],
        "system": input_state["system"],
        "question": input_state["question"]
    }
    messages = prompt.invoke(prompt_input)
    
    try:
        # 呼叫 LM Studio 模型
        response = lmstudio_llm.invoke(messages)
    except Exception as e:
        # 如果呼叫失敗，返回錯誤訊息
        response = type('Response', (object,), {'content': f"LLM (LM Studio/Gemini) 呼叫失敗，錯誤：{e}"})

    # direct 模型沒有 RAG 連結
    global latest_related_links
    latest_related_links = []

    clean_answer = response.content.replace("\\u000A", "\n")
    return clean_answer