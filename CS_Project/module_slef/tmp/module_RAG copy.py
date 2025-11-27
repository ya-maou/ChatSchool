# module_RAG.py

# ===【向量嵌入與搜尋：LangChain / Chroma / Google Generative AI】===
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model

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
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not GOOGLE_API_KEY or not MISTRAL_API_KEY:
    raise ValueError("請在 .env 設定 GOOGLE_API_KEY 和 MISTRAL_API_KEY")

# --- PostgreSQL 連線 ---
DB_CONFIG = {
    "host": "localhost",
    "user": "postgres",
    "password": "password",
    "database": "postgres",
    "port": 5432
}

# --- 初始化 LLM 與 Embeddings ---
llm = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# --- Prompt 模板 ---
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

# --- LangGraph 狀態結構 ---
class State(TypedDict):
    title: str
    system: str
    task: str
    question: str
    query: dict
    context: List[Document]
    answer: str

# --- Mistral API ---
def call_mistral(prompt_text: str):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",
        "temperature": 0.1,
        "top_p": 1,
        "max_tokens": 7800,
        "messages": [{"role": "user", "content": prompt_text}]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Mistral API 錯誤：{response.status_code}, {response.text}")
    return response.json()

def translate_with_mistral(text: str, max_retries=3, delay=2) -> str:
    for attempt in range(max_retries):
        try:
            prompt_text = f"請將以下內容翻譯成英文(若有聯合大學的字樣，請一律翻成 National United University)，不要說多餘的話：\n{text}"
            response = call_mistral(prompt_text)
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[翻譯失敗] 第 {attempt + 1} 次，錯誤：{e}")
            time.sleep(delay)
    raise Exception("Mistral 翻譯失敗，超過最大重試次數")

# --- 分析問題 (僅語言偵測) ---
def analyze_query(state: State):
    question = state["question"]
    try:
        lang = detect(question)
    except:
        lang = "zh"
    return {"query": {"language": lang}}

# --- 檢索文章 (純語意向量檢索) ---
def retrieve(state: State):
    global latest_related_links
    latest_related_links = []

    question = state["question"]
    try:
        translated_question = translate_with_mistral(question)
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

# --- 生成回答 ---
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
    response = llm.invoke(messages)
    return {"answer": response.content.strip()}

# --- LangGraph 對話圖 ---
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# --- 取得最新連結 ---
def get_latest_links() -> List[dict]:
    global latest_related_links
    return latest_related_links

def ask_rag(question: str) -> Tuple[str, List[dict]]:
    """
    問答函數
    :param question: 使用者問題
    :param version: 選擇 RAG 版本 ('v6' 或 'v7')
    :return: Tuple(answer_text, latest_related_links)
    """

    input_state = {
        "title": "NUU 資訊檢索問答",
        "system": (
            "你是一個專業的大學生助理，負責回答聯合大學網站相關問題。\n"
            "【最高優先規則】: 你必須**嚴格**依據使用者提問的語言進行回答。\n"
        ),
        # 將語言切換規則前置且明確化
        "task": (
            "【語言規則】: 嚴格根據使用者問題的語言來回答 (例如，問題是英文，回答就必須是英文; 問題是日文，回答就必須是日文)。"
            "【主要任務】: 根據提供的資料內容，盡可能詳盡地回答使用者的問題。"
            "【無資料處理】: 如果找不到相關資料，請嚴格回傳: Not Found 並且使用**使用者提問的語言**向使用者誠實說明找不到相關資料。"
        ),
        "question": question
    }

    # 選擇 graph 與對應連結函數
    selected_graph = graph
    get_links = latest_related_links

    # 呼叫 RAG 流程
    result = selected_graph.invoke(input_state)
    raw_answer = result.get("answer", "")
    clean_answer = raw_answer.replace("\\u000A", "\n")

    # 取得最新文章連結
    links = latest_related_links

    return clean_answer, links
