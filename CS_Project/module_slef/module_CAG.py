import os
import json
from time import time
from typing import Tuple, List, Dict
from dotenv import load_dotenv
import psycopg2
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

# ===【載入環境變數】===
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONN_STR")

if not GOOGLE_API_KEY:
    raise ValueError("請在 .env 設定 GOOGLE_API_KEY")
if not POSTGRES_CONNECTION_STRING:
    raise ValueError("請在 .env 設定 POSTGRES_CONN_STR")

LMSTUDIO_MODEL = "google/gemma-3-4B"
LMSTUDIO_URL = "http://192.168.98.34:1234/v1"

def init_LMStudio(model: str, base_url: str, api_key: str = ".", configurable_fields: None = None, config_prefix: str | None = None, **kwargs) -> BaseChatModel:
    """使用 LangChain 連接至 LM Studio 的 OpenAI 相容 API"""
    return init_chat_model(model=model, base_url=base_url, configurable_fields=configurable_fields, config_prefix=config_prefix, model_provider="openai", api_key=api_key, **kwargs)

llm_gemini_backup = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")

try:
    lmstudio_llm = init_LMStudio(model=LMSTUDIO_MODEL, base_url=LMSTUDIO_URL)
    print(f"[CAG 模組] 成功連接 LM Studio ({LMSTUDIO_URL}) 作為主要 LLM。")
except Exception as e:
    print(f"[CAG 模組] LM Studio 連接失敗，錯誤：{e}")
    print("[CAG 模組] 回退使用 Gemini 2.0 Flash 作為主要 LLM。")
    lmstudio_llm = llm_gemini_backup

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

CAG_CONTEXT_FILE = "C:/Users/User/Studio/ChatSchool/CS_Project/module_slef/cag_context.txt"
CAG_KNOWLEDGE_TEXT = ""

# ===【資料庫工具】===
def get_pg_connection():
    """建立 PostgreSQL 連線"""
    conn_str = POSTGRES_CONNECTION_STRING.replace("postgresql+psycopg2://", "postgresql://")
    return psycopg2.connect(conn_str)

# ===【知識庫載入】===
def load_cag_knowledge():
    global CAG_KNOWLEDGE_TEXT
    try:
        with open(CAG_CONTEXT_FILE, "r", encoding="utf-8") as f:
            CAG_KNOWLEDGE_TEXT = f.read()
    except FileNotFoundError:
        CAG_KNOWLEDGE_TEXT = ""

# ===【Prompt 建構】===
def build_cag_prompt(question: str) -> str:
    prompt_template = PromptTemplate.from_template("""
    你是聯合大學校園資訊的專家，請嚴格根據你所提供的「校園知識庫」來回答問題。

    **【最高優先規則】**
    1. 必須嚴格依據「校園知識庫」中的內容回答。
    2. 回答內容必須簡潔、準確，並使用使用者提問的語言。
    3. 如果「校園知識庫」中找不到答案，請誠實地回答：「Not Found : 校園知識庫中未提供此資訊。」

    **【校園知識庫】**
    ------------------------------------------------
    {context}
    ------------------------------------------------

    **【使用者問題】**
    {question}

    **【你的回答】**
    """)
    return prompt_template.invoke({
        "context": CAG_KNOWLEDGE_TEXT if CAG_KNOWLEDGE_TEXT else "無知識庫內容。",
        "question": question
    }).text.strip()

# ===【向量快取機制】 (保持不變) ===
def add_to_cag_cache(question: str, answer: str):
    """將問答嵌入後儲存至 PostgreSQL 向量快取"""
    embedding = embedding_model.embed_query(question)
    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO cag_cache (question, embedding, answer)
        VALUES (%s, %s, %s)
    """, (question, embedding, answer))
    conn.commit()
    cur.close()
    conn.close()
    print("[CAG] 新快取已寫入 PostgreSQL。")

def query_cag_cache(question: str, threshold: float = 0.9):
    """從 PostgreSQL 查詢相似問題的快取回答"""
    embedding = embedding_model.embed_query(question)
    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT question, answer, 1 - (embedding <=> %s::vector) AS similarity
        FROM cag_cache
        ORDER BY embedding <=> %s::vector
        LIMIT 1;
    """, (embedding, embedding))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row and row[2] >= threshold:
        # print(f"[CAG] 相似快取命中，相似度={row[2]:.3f}")
        return row[1]
    return None

# ===【主問答流程 (MODIFIED: 使用 lmstudio_llm) 】===
def ask_cag(question: str) -> Tuple[str, List[Dict[str, str]]]:
    if not CAG_KNOWLEDGE_TEXT:
        return "CAG 知識庫載入失敗，無法提供校園快速資訊。", []

    # Step 1: 查詢快取 (保持不變)
    cached_answer = query_cag_cache(question)
    if cached_answer:
        return cached_answer, []

    # Step 2: 呼叫 LLM
    # print("[CAG] 無快取，呼叫 LLM...")
    full_prompt = build_cag_prompt(question)
    start_time = time()
    try:
        # 將 llm.invoke 替換為 lmstudio_llm.invoke
        response = lmstudio_llm.invoke(full_prompt) # <--- *** 替換成 LM Studio 模型 ***
        answer = response.content.strip()
    except Exception as e:
        # 這裡的錯誤訊息需要反映當前使用的模型可能發生的錯誤
        answer = f"LLM (LM Studio/Gemini) 呼叫失敗，錯誤：{e}"
    duration = time() - start_time
    # print(f"[CAG] 生成時間: {duration:.2f} 秒")

    return answer, []

def init_cag_module():
    load_cag_knowledge()
    print("[CAG] 模組初始化完成，知識庫載入成功。")