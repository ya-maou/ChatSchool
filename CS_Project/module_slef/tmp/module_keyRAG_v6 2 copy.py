# ===【向量嵌入與搜尋：LangChain / Chroma / Google Generative AI】===
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ===【LangChain 核心模組】===
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel # 確保匯入
from langchain_core.messages import SystemMessage, HumanMessage # 確保匯入

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

import numpy as np

latest_related_links: List[dict] = []


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# MISTRAL_API_KEY 已移除

if not GOOGLE_API_KEY:
    raise ValueError("未設定 GOOGLE_API_KEY")


# --- [新增] init_LMStudio 函式 ---
def init_LMStudio(model: str, base_url: str, api_key: str = ".", configurable_fields: None = None, config_prefix: str | None = None, **kwargs) -> BaseChatModel:
    """使用 LangChain 連接至 LM Studio 的 OpenAI 相容 API"""
    return init_chat_model(model=model, base_url=base_url, configurable_fields=configurable_fields, config_prefix=config_prefix, model_provider="openai", api_key=api_key, **kwargs)

# --- [新增] translate 函式 ---
def translate(model: BaseChatModel, lang: str, content: str):
    """通用翻譯功能：使用 BaseChatModel 翻譯內容到目標語言。"""
    return model.invoke([
        SystemMessage(content=f'Only answer the translate content to [{lang}] from the user\'s input.'),
        HumanMessage(content=content),
    ]).content


# PostgreSQL 連線 (保持不變)
conn = psycopg2.connect(
    host="localhost",
    user="postgres",
    password="password",
    database="postgres",
    port=5432
)

# 初始化 LLM (保持不變)
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

# 資料結構 (保持不變)
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
            # --- [替換點 1] 使用新的 translate 函式和 llm ---
            # 將 NUU 的翻譯指令明確放入 content 中，以確保翻譯質量
            translation_content = f"請將以下內容翻譯成英文，若有『聯合大學』的字樣，請一律翻成『National United University』。原文：\n{question}"
            translated_question = translate(llm, "English", translation_content)

            en_kw_obj = structured_llm.invoke(
                f"請從以下英文問題中儘可能萃取關鍵字。問題：{translated_question}"
            )
            en_keywords = en_kw_obj.keywords if hasattr(en_kw_obj, "keywords") else en_kw_obj.get("keywords", [])
        except Exception as e:
            # 翻譯失敗時，不需要重試機制，直接略過英文關鍵字即可
            print(f"[關鍵字英文翻譯萃取失敗] {e}")
            en_keywords = []


    all_keywords = list(set(zh_keywords + en_keywords)) 
    
    # print(f"[analyze_query] 原語言關鍵字: {zh_keywords}")
    # print(f"[analyze_query] 英文關鍵字: {en_keywords}")

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
    """
    global latest_related_links
    
    # 1. 計算 query_vector
    question = state["question"]
    # 翻譯成英文以供 embedding 模型使用
    try:
        # --- [替換點 2] 使用新的 translate 函式和 llm ---
        translation_content = f"請將以下內容翻譯成英文，若有『聯合大學』的字樣，請一律翻成『National United University』。原文：\n{question}"
        translated_question = translate(llm, "English", translation_content)
    except Exception as e:
        print(f"[翻譯錯誤] 無法翻譯問題，使用原文。錯誤: {e}")
        translated_question = question
        
    query_vector = embeddings.embed_query(translated_question)
    # 將 numpy array 轉換為 list，以符合 psycopg2 的參數要求

    # 修正: 安全地將向量轉換為 list
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()
    elif not isinstance(query_vector, list):
        try:
             query_vector = list(query_vector)
        except TypeError:
             print(f"[ERROR] 向量類型無法轉換: {type(query_vector)}")
             raise

    # 2. 提取關鍵詞 (保持不變)
    keywords_obj = state.get("query")
    if isinstance(keywords_obj, dict) and "keywords" in keywords_obj:
        keywords = keywords_obj["keywords"]
    elif hasattr(keywords_obj, "keywords"): # 處理 pydantic 模型
        keywords = keywords_obj.keywords
    else:
        keywords = []

    print(f"[retrieve] 關鍵詞：{keywords[:5]}...")

    # 3. 執行資料庫檢索 (保持不變)
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
    latest_related_links = [] 
    
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
        
        # 收集相關連結
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


# --- 生成回答 (保持不變)---
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

# --- 對話圖 (保持不變)---
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()


# --- 取得最新連結 (保持不變)---
def get_latest_links() -> List[dict]:
    global latest_related_links
    return latest_related_links