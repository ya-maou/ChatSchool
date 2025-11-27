# module_RAG.py

# ===【向量嵌入與搜尋：LangChain / Chroma / Google Generative AI】===
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel # 確保匯入
from langchain_core.messages import SystemMessage, HumanMessage # 確保匯入

# ===【LangGraph 工作流程】===
from langgraph.graph import START, StateGraph

# ===【資料庫：PostgreSQL】===
import psycopg2

# ===【資料結構與型別定義】===
from pydantic import BaseModel
from typing import List, TypedDict

# ===【系統工具】===
import json
import os
import time
import requests
from dotenv import load_dotenv
from langdetect import detect
import numpy as np
from typing import Tuple, List

# 全域變數
latest_related_links: List[dict] = []

# --- 載入環境變數 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# MISTRAL_API_KEY 已移除，因為不再使用

if not GOOGLE_API_KEY:
    raise ValueError("請在 .env 設定 GOOGLE_API_KEY")

# --- PostgreSQL 連線 ---
DB_CONFIG = {
    "host": "localhost",
    "user": "postgres",
    "password": "password",
    "database": "postgres",
    "port": 5432
}

# --- [新增] init_LMStudio 函式 ---
def init_LMStudio(model: str, base_url: str, api_key: str = ".", configurable_fields: None = None, config_prefix: str | None = None, **kwargs) -> BaseChatModel:
    """使用 LangChain 連接至 LM Studio 的 OpenAI 相容 API"""
    return init_chat_model(model=model, base_url=base_url, configurable_fields=configurable_fields, config_prefix=config_prefix, model_provider="openai", api_key=api_key, **kwargs)

# --- [新增] translate 函式 (通用翻譯，將取代 translate_with_mistral) ---
def translate(model: BaseChatModel, lang: str, content: str):
    """通用翻譯功能：使用 BaseChatModel 翻譯內容到目標語言。"""
    # 這裡的 prompt 包含了您需要的「聯合大學」固定翻譯指令
    prompt_text = f"請將以下內容翻譯成英文(若有聯合大學的字樣，請一律翻成 National United University)，不要說多餘的話：\n{content}"
    
    return model.invoke([
        SystemMessage(content=f'Only answer the translate content to [{lang}] from the user\'s input.'),
        HumanMessage(content=prompt_text),
    ]).content.strip()


# --- LLM 初始化 (關鍵變動區) ---
# 設置 LM Studio 資訊 (假設與上一個例子的設定相同)
LMSTUDIO_MODEL = "google/gemma-3-4B" 
LMSTUDIO_URL = "http://192.168.98.34:1234/v1" 

# 初始化 LLM (Gemini 僅作為最終備用)
llm = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

try:
    # 嘗試連接 LM Studio 作為 RAG 流程的主要模型
    lmstudio_llm = init_LMStudio(model=LMSTUDIO_MODEL, base_url=LMSTUDIO_URL)
except Exception as e:
    print(f"[LMStudio RAG 模組初始化失敗] 無法連線至 {LMSTUDIO_URL}。錯誤: {e}")
    # 若失敗，回退到 Gemini 作為所有 LLM 任務的備用模型
    lmstudio_llm = llm 


# --- Prompt 模板 (保持不變) ---
prompt = PromptTemplate.from_template("""# {title}
## System
{system}
## Retrieval Context
{context}
## Task
{task}
## Question
{question}
## Answer
""")

# --- LangGraph 狀態結構 (保持不變) ---
class State(TypedDict):
    title: str
    system: str
    task: str
    question: str
    query: dict
    context: List[Document]
    answer: str

def analyze_query(state: State):
    question = state["question"]
    try:
        lang = detect(question)
    except:
        lang = "zh"
    return {"query": {"language": lang}}

# --- 檢索文章 (MODIFIED: 使用 lmstudio_llm 翻譯) ---
def retrieve(state: State):
    global latest_related_links
    latest_related_links = []

    question = state["question"]
    try:
        # 將翻譯邏輯替換為使用 lmstudio_llm 的 translate 函數
        translated_question = translate(lmstudio_llm, "English", question)
        print(f"[retrieve] 翻譯結果: {translated_question}")
    except Exception as e:
        print(f"[翻譯錯誤] {e}")
        translated_question = question

    query_vector = embeddings.embed_query(translated_question)
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()
    elif not isinstance(query_vector, list):
        query_vector = list(query_vector)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    top_k = 10

    sql = f"""
    SELECT A.id, A.title, A.url, AC.chunk_content,
           (1 - (AC.articles_embedding <=> %s::vector)) AS vector_score
    FROM ARTICLES_CHUNKS AC
    JOIN ARTICLES A ON A.id = AC.article_id
    WHERE AC.articles_embedding IS NOT NULL
    ORDER BY AC.articles_embedding <=> %s::vector
    LIMIT {top_k};
    """
    params = (query_vector, query_vector)
    cur.execute(sql, params)
    rows = cur.fetchall()

    retrieved_docs = []
    seen_urls = set()
    for row in rows:
        article_id, title, url, content, vector_score = row
        doc = Document(
            page_content=content,
            metadata={
                "article_id": article_id,
                "title": title,
                "url": url,
                "vector_score": float(vector_score)
            }
        )
        retrieved_docs.append(doc)
        if url and url not in seen_urls:
            latest_related_links.append({"title": title, "url": url})
            seen_urls.add(url)

    cur.close()
    conn.close()
    return {"context": retrieved_docs}

# --- 生成回答 (MODIFIED: 使用 lmstudio_llm 生成) ---
def generate(state: State):
    docs_text = "\n\n".join(doc.page_content for doc in state["context"])
    prompt_input = {
        "title": state["title"],
        "system": state["system"],
        "task": state["task"],
        "context": docs_text,
        "question": state["question"]
    }
    messages = prompt.invoke(prompt_input)
    response = lmstudio_llm.invoke(messages) # <--- *** 使用 LMStudio ***
    return {"answer": response.content.strip()}

# --- LangGraph 對話圖 (保持不變) ---
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# --- 取得最新連結 (保持不變) ---
def get_latest_links() -> List[dict]:
    global latest_related_links
    return latest_related_links

# --- 問答函數 (保持不變) ---
def ask_rag(question: str) -> Tuple[str, List[dict]]:
    """
    問答函數
    :param question: 使用者問題
    :param version: 選擇 RAG 版本 ('v6' 或 'v7')
    :return: Tuple(answer_text, latest_related_links)
    """

    input_state = {
        "title": "NUU Information Retrieval Q&A",
        "system": (
            "You are a professional university assistant specializing in National United University (NUU) website-related questions.\n"
            "**[STRICTEST PRIORITY RULE]**: You **MUST** strictly answer using the **EXACT SAME LANGUAGE** as the user's question."
        ),
        # 將語言切換規則前置且明確化
        "task": (
            "[LANGUAGE RULE]: You **MUST** strictly answer using the **EXACT SAME LANGUAGE** as the user's question (e.g., if the question is in English, the answer must be in English; if in Japanese, the answer must be in Japanese).\n"
            "[MAIN TASK]: Answer the user's question as detailedly as possible based on the provided retrieval context.\n"
            "[NO DATA HANDLING]: If no relevant data is found in the context, you **MUST** strictly return: `Not Found` and honestly explain to the user in the **EXACT SAME LANGUAGE** as the user's question that no relevant data was found."
        ),
        "question": question
    }

    # 選擇 graph 與對應連結函數
    selected_graph = graph
    # get_links = latest_related_links # 此行不再需要，因為我們直接使用全局變數

    # 呼叫 RAG 流程
    result = selected_graph.invoke(input_state)
    raw_answer = result.get("answer", "")
    clean_answer = raw_answer.replace("\\u000A", "\n")

    # 取得最新文章連結
    links = latest_related_links

    return clean_answer, links