# ===【向量嵌入與搜尋：LangChain / Chroma / Google Generative AI】===
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ===【LangChain 核心模組】===
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model

# ===【LangGraph 工作流程】===
from langgraph.graph import START, StateGraph

# ===【資料庫：PostgreSQL】===
import psycopg2
import psycopg2.extras

# ===【資料結構與型別定義】===
from pydantic import BaseModel
from typing import List, TypedDict

# ===【系統與工具】===
import json
import os
import time
import requests
from dotenv import load_dotenv

from typing import List

import psycopg2
import numpy as np

latest_related_links: List[dict] = []


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not GOOGLE_API_KEY or not MISTRAL_API_KEY:
    raise ValueError("未設定 GOOGLE_API_KEY & MISTRAL_API_KEY")

# PostgreSQL 連線
conn = psycopg2.connect(
    host="localhost",
    user="postgres",
    password="password",
    database="postgres",
    port=5432
)

# 初始化 LLM
llm = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Prompt 模板
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

# 資料結構
class Search(BaseModel):
    keywords: List[str]

class State(TypedDict):
    title: str
    system: str
    task: str
    question: str
    query: Search
    context: List[Document]
    answer: str

# --- Mistral API ---
def call_mistral(prompt):
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
        "messages": [{"role": "user", "content": prompt}]
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
            print(f"Mistral 翻譯：{response.status_code}, {response.text}")
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[翻譯失敗] 第 {attempt + 1} 次嘗試，錯誤：{e}")
            time.sleep(delay)
    raise Exception("Mistral 翻譯失敗，超過最大重試次數")

# --- 分析使用者查詢 (保持不變) ---
from langdetect import detect

def analyze_query(state: State):
    """
    從使用者問題萃取關鍵字：
    1. 先用 LLM 提取原語言關鍵字
    2. 若非英文，翻譯成英文再提取英文關鍵字
    3. 返回結構化結果: {"zh_keywords": [...], "en_keywords": [...]}
    """
    question = state["question"]
    structured_llm = llm.with_structured_output(Search)
    
    # 1. 先提取原語言關鍵字
    original_keywords = structured_llm.invoke(
        f"請從以下問題中儘可能萃取出**使用者需要查詢文章**的關鍵字。可以重複，只要有可能就列出，不限制數量，動詞與常用助詞可以忽略。問題：{question}"
    )
    try:
        lang = detect(question)
    except:
        lang = "zh"
    
    zh_keywords = original_keywords.keywords if hasattr(original_keywords, "keywords") else original_keywords.get("keywords", [])
    en_keywords = []

    # 3. 如果不是英文，翻譯成英文再提取關鍵字
    if lang != "en":
        try:
            translated_question = translate_with_mistral(question)
            en_kw_obj = structured_llm.invoke(
                f"請從以下英文問題中儘可能萃取關鍵字。問題：{translated_question}"
            )
            en_keywords = en_kw_obj.keywords if hasattr(en_kw_obj, "keywords") else en_kw_obj.get("keywords", [])
        except Exception as e:
            print(f"[關鍵字英文翻譯萃取失敗] {e}")
            en_keywords = []


    all_keywords = list(set(zh_keywords + en_keywords)) 
    
    print(f"[analyze_query] 原語言關鍵字: {zh_keywords}")
    print(f"[analyze_query] 英文關鍵字: {en_keywords}")

    return {
        "query": {
            "keywords": all_keywords, 
            "zh_keywords": zh_keywords,
            "en_keywords": en_keywords
        }
    }


def retrieve(state: State):
    """
    使用 PostgreSQL + pgvector 進行多模檢索，並作為 LangGraph 節點。
    
    參數:
        state: State — LangGraph 的狀態字典，包含 question 和 query (keywords)
    回傳:
        dict — 包含檢索結果 (context)
    """
    global latest_related_links
    
    # 1. 計算 query_vector
    question = state["question"]
    # 翻譯成英文以供 embedding 模型使用 (如果您的 embedding 模型是為英文優化的，如 text-embedding-004)
    # 如果您的模型已經針對中文優化，可以跳過翻譯。這裡假設需要翻譯。
    try:
        translated_question = translate_with_mistral(question)
    except Exception as e:
        print(f"[翻譯錯誤] 無法翻譯問題，使用原文。錯誤: {e}")
        translated_question = question
        
    query_vector = embeddings.embed_query(translated_question)
    # 將 numpy array 轉換為 list，以符合 psycopg2 的參數要求

    # 修正: 安全地將向量轉換為 list，如果它本身不是 list
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()
    elif not isinstance(query_vector, list):
        # 如果不是 ndarray 也不是 list，則嘗試將其轉換為 list，或記錄錯誤
        try:
             query_vector = list(query_vector)
        except TypeError:
             print(f"[ERROR] 向量類型無法轉換: {type(query_vector)}")
             raise

    # 2. 提取關鍵詞
    keywords_obj = state.get("query")
    if isinstance(keywords_obj, dict) and "keywords" in keywords_obj:
        keywords = keywords_obj["keywords"]
    elif hasattr(keywords_obj, "keywords"): # 處理 pydantic 模型
        keywords = keywords_obj.keywords
    else:
        keywords = []

    print(f"[retrieve] 關鍵詞：{keywords[:5]}...")

    # 3. 執行資料庫檢索
    conn = psycopg2.connect(
        host="localhost",
        user="postgres",
        password="password",
        database="postgres",
        port=5432
    )
    cur = conn.cursor()
    top_k = 10 
    
    # --- SQL 查詢模板 (保持不變) ---
    if keywords and len(keywords) > 0:
        # --- 含關鍵字的情況 ---
        sql = f"""
        WITH keyword_match AS (
            SELECT AK.article_id, COUNT(*) AS keyword_score
            FROM ARTICLES_KEYWORDS AK
            JOIN KEYWORDS K ON AK.keyword_id = K.id
            WHERE K.keyword = ANY(%s)
            GROUP BY AK.article_id
        ),
        base_candidates AS (
            SELECT AC.article_id, AC.chunk_content, AC.chunk_token,
                   (1 - (AC.articles_embedding <=> %s::vector)) AS vector_score
            FROM ARTICLES_CHUNKS AC
            WHERE AC.articles_embedding IS NOT NULL
            ORDER BY AC.articles_embedding <=> %s::vector
            LIMIT 300
        )
        SELECT A.id, A.title, A.url, BC.chunk_content,
               BC.vector_score,
               COALESCE(KM.keyword_score, 0) AS keyword_score,
               (BC.vector_score * 0.7 + COALESCE(KM.keyword_score, 0) * 0.3) AS final_score
        FROM base_candidates BC
        JOIN ARTICLES A ON A.id = BC.article_id
        LEFT JOIN keyword_match KM ON BC.article_id = KM.article_id
        ORDER BY final_score DESC
        LIMIT {top_k};
        """
        # 注意：這裡 keywords 必須是 list[str]，query_vector 必須是 list[float]
        params = (keywords, query_vector, query_vector)
    else:
        # --- 僅語意搜尋 ---
        sql = f"""
        SELECT A.id, A.title, A.url, AC.chunk_content,
               (1 - (AC.articles_embedding <=> %s::vector)) AS vector_score,
               0 AS keyword_score,
               (1 - (AC.articles_embedding <=> %s::vector)) AS final_score
        FROM ARTICLES_CHUNKS AC
        JOIN ARTICLES A ON A.id = AC.article_id
        WHERE AC.articles_embedding IS NOT NULL
        ORDER BY AC.articles_embedding <=> %s::vector
        LIMIT {top_k};
        """
        params = (query_vector, query_vector, query_vector)

    cur.execute(sql, params)
    rows = cur.fetchall()

    retrieved_docs = []
    related_links_set = set()
    latest_related_links = [] # 重置全域連結列表
    
    for row in rows:
        article_id, title, url, content, vector_score, keyword_score, final_score = row
        
        # 建立 LangChain Document
        doc = Document(
            page_content=content,
            metadata={
                "article_id": article_id,
                "title": title,
                "url": url,
                "vector_score": float(vector_score),
                "keyword_score": int(keyword_score),
                "final_score": float(final_score)
            }
        )
        retrieved_docs.append(doc)
        
        # 收集相關連結 (確保唯一性)
        if url and url not in related_links_set:
            latest_related_links.append({
                "title": title,
                "url": url
            })
            related_links_set.add(url)

    cur.close()
    conn.close()
    
    # 4. 返回 LangGraph 狀態
    return {"context": retrieved_docs}


# --- 生成回答---
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
    return {"answer": response.content}

# --- 對話圖---
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()


# --- 取得最新連結---
def get_latest_links() -> List[dict]:
    global latest_related_links
    return latest_related_links