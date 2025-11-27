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
from langdetect import detect # 確保導入

latest_related_links: List[dict] = []


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("未設定 GOOGLE_API_KEY")


# --- [新增] init_LMStudio 函式 (保持不變) ---
def init_LMStudio(model: str, base_url: str, api_key: str = ".", configurable_fields: None = None, config_prefix: str | None = None, **kwargs) -> BaseChatModel:
    """使用 LangChain 連接至 LM Studio 的 OpenAI 相容 API"""
    return init_chat_model(model=model, base_url=base_url, configurable_fields=configurable_fields, config_prefix=config_prefix, model_provider="openai", api_key=api_key, **kwargs)

# --- [新增] translate 函式 (保持不變) ---
def translate(model: BaseChatModel, lang: str, content: str):
    """通用翻譯功能：使用 BaseChatModel 翻譯內容到目標語言。"""
    return model.invoke([
        SystemMessage(content=f'Only answer the translate content to [{lang}] from the user\'s input.'),
        HumanMessage(content=content),
    ]).content


# --- LLM 初始化 (關鍵變動區) ---

# 設置 LM Studio 資訊 (與 chat.py 保持一致)
LMSTUDIO_MODEL = "google/gemma-3-4B" 
LMSTUDIO_URL = "http://192.168.98.34:1234/v1" 

# 初始化 LLM (Gemini 僅作為最終備用，如果 LMStudio 連線失敗)
llm = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

try:
    # 嘗試連接 LM Studio 作為 RAG 流程的主要模型
    lmstudio_llm = init_LMStudio(model=LMSTUDIO_MODEL, base_url=LMSTUDIO_URL)
except Exception as e:
    print(f"[LMStudio RAG 模組初始化失敗] 無法連線至 {LMSTUDIO_URL}。錯誤: {e}")
    # 若失敗，回退到 Gemini 作為所有 LLM 任務的備用模型
    lmstudio_llm = llm 

# PostgreSQL 連線 (保持不變)
conn = psycopg2.connect(
    host="localhost",
    user="postgres",
    password="password",
    database="postgres",
    port=5432
)

# Prompt 模板 (保持不變)
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


# --- 分析使用者查詢 (MODIFIED: 全部使用 lmstudio_llm) ---
def analyze_query(state: State):
    """
    從使用者問題萃取關鍵字：
    1. 使用 **lmstudio_llm** 提取原語言關鍵字。
    2. 若非英文，使用 **lmstudio_llm** 翻譯，然後用 **lmstudio_llm** 提取英文關鍵字。
    3. 返回結構化結果。
    """
    question = state["question"]
    # 結構化輸出使用 lmstudio_llm
    structured_llm = lmstudio_llm.with_structured_output(Search)
    
    # 1. 先提取原語言關鍵字
    original_keywords = structured_llm.invoke( # <--- *** 使用 LMStudio ***
        f"請從以下問題中儘可能萃取出**使用者需要查詢文章**的關鍵字並嚴格輸出。可以重複，只要有可能就列出，不限制數量，動詞與常用助詞可以忽略。問題：{question}"
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
            # --- [替換點 1] 翻譯環節切換到 lmstudio_llm ---
            translation_content = f"請將以下內容翻譯成英文，若有『聯合大學』的字樣，請一律翻成『National United University』。原文：\n{question}"
            translated_question = translate(lmstudio_llm, "English", translation_content) # <--- *** 切換到 LMStudio ***
            print(f"[analyze_query] 翻譯結果: {translated_question}")
            # 英文關鍵字提取切換到 lmstudio_llm
            en_kw_obj = structured_llm.invoke( # <--- *** 使用 LMStudio ***
                f"請從以下英文問題中儘可能萃取關鍵字。問題：{translated_question}"
            )
            en_keywords = en_kw_obj.keywords if hasattr(en_kw_obj, "keywords") else en_kw_obj.get("keywords", [])
        except Exception as e:
            print(f"[關鍵字英文翻譯萃取失敗] {e}")
            en_keywords = []

    print(f"[analyze_query] 原語言關鍵字: {zh_keywords}")
    print(f"[analyze_query] 英文關鍵字: {en_keywords}")

    all_keywords = list(set(zh_keywords + en_keywords)) 
    
    return {
        "query": {
            "keywords": all_keywords, 
            "zh_keywords": zh_keywords,
            "en_keywords": en_keywords
        }
    }


# --- 檢索 (MODIFIED: 翻譯環節切換到 lmstudio_llm) ---
# --- 檢索 (MODIFIED: 新增檢索結果少於 5 筆時，回退到純語義搜尋的邏輯) ---
def retrieve(state: State):
    """
    使用 PostgreSQL + pgvector 進行多模檢索，並作為 LangGraph 節點。
    """
    global latest_related_links
    top_k = 10 # 設定最終返回的數量

    # 1. 計算 query_vector (翻譯和嵌入)
    question = state["question"]
    try:
        # 檢索前的翻譯環節切換到 lmstudio_llm
        translation_content = f"請將以下內容翻譯成英文，若有『聯合大學』的字樣，請一律翻成『National United University』。原文：\n{question}"
        translated_question = translate(lmstudio_llm, "English", translation_content)
    except Exception as e:
        print(f"[翻譯錯誤] 無法翻譯問題，使用原文。錯誤: {e}")
        translated_question = question
        
    query_vector = embeddings.embed_query(translated_question)

    # 修正: 安全地將向量轉換為 list
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()
    elif not isinstance(query_vector, list):
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

    # 3. 執行資料庫檢索
    conn = psycopg2.connect(
        host="localhost",
        user="postgres",
        password="password",
        database="postgres",
        port=5432
    )
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # 使用 DictCursor 方便處理
    
    # 執行包含關鍵字的混合檢索
    if keywords and len(keywords) > 0:
        sql_hybrid = f"""
        WITH keyword_match AS (
            SELECT AK.article_id, COUNT(*) AS keyword_score
            FROM ARTICLES_KEYWORDS AK
            JOIN KEYWORDS K ON AK.keyword_id = K.id
            WHERE K.keyword = ANY(%s)
            GROUP BY AK.article_id
        ),
        base_candidates AS (
            SELECT AC.article_id, AC.chunk_content,
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
        params_hybrid = (keywords, query_vector, query_vector)
        cur.execute(sql_hybrid, params_hybrid)
        rows = cur.fetchall()

        if len(rows) < 2:
            print(f"[檢索回退] 混合檢索結果少於 5 筆 ({len(rows)} 筆)。回退到純語義搜尋。")
            
            # 清空結果，準備執行純語義搜尋
            rows = [] 
        else:
            print(f"[檢索成功] 執行混合檢索，獲得 {len(rows)} 筆結果。")
            
    # 如果一開始就沒有關鍵字，或者混合檢索結果少於 5 筆 (回退邏輯)
    if not keywords or len(rows) < 1:
        sql_semantic = f"""
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
        params_semantic = (query_vector, query_vector, query_vector)
        cur.execute(sql_semantic, params_semantic)
        rows = cur.fetchall()
        
        if not keywords:
             print(f"[檢索模式] 執行純語義搜尋，獲得 {len(rows)} 筆結果。")
        else:
             print(f"[回退結果] 執行純語義搜尋，獲得 {len(rows)} 筆結果。")


    # 4. 處理檢索結果 (保持不變)
    retrieved_docs = []
    related_links_set = set()
    latest_related_links = []
    
    for row in rows:
        # 使用 DictCursor，可以透過 key 存取
        doc = Document(
            page_content=row['chunk_content'],
            metadata={
                "article_id": row['id'],
                "title": row['title'],
                "url": row['url'],
                "vector_score": float(row['vector_score']),
                "keyword_score": int(row['keyword_score']),
                "final_score": float(row['final_score'])
            }
        )
        retrieved_docs.append(doc)
        
        # 收集相關連結
        if row['url'] and row['url'] not in related_links_set:
            latest_related_links.append({
                "title": row['title'],
                "url": row['url']
            })
            related_links_set.add(row['url'])

    cur.close()
    conn.close()
    
    # 5. 返回 LangGraph 狀態
    return {"context": retrieved_docs}


# --- 生成回答 (MODIFIED: 切換到 lmstudio_llm 並修正 Prompt 格式) ---
def generate(state: State):
    docs_text = "\n\n".join(doc.page_content for doc in state["context"])
    question = state["question"]
    
    # 1. 建立 SystemMessage (角色扮演, 語言規則, 上下文)
    full_system_instruction = (
        "您是一位專業的國立聯合大學助理，負責回答與 NUU 網站相關的問題。\n"
        # ⚠️ 強調輸出語言，並指示模型在處理完上下文後進行翻譯
        f"[!!! 輸出語言指令 !!!]: 您的**最終回答**必須使用：user提問的原語言！\n"
        "您應先理解上下文內容，然後根據使用者的問題，將您的答案**嚴格翻譯並輸出**為指定的目標語言。\n"
        "[無資料處理]: 如果找不到相關資訊，請使用目標輸出語言誠實告知使用者。\n\n"
        "--- Retrieval Context (請使用這些資訊來回答) ---\n"
        f"{docs_text}"
    )

    system_message = SystemMessage(content=full_system_instruction)
    
    # 2. 建立 HumanMessage
    human_message = HumanMessage(content=question)

    # 3. 呼叫 LLM
    response = lmstudio_llm.invoke([system_message, human_message]) 
    return {"answer": response.content}

# --- 對話圖 (保持不變)---
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()


# --- 取得最新連結 (保持不變)---
def get_latest_links() -> List[dict]:
    global latest_related_links
    return latest_related_links