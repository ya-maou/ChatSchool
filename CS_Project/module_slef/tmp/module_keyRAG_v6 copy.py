# ===【向量嵌入與搜尋：LangChain / Chroma / Google Generative AI】===
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ===【LangChain 核心模組】===
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# ===【LangGraph 工作流程】===
from langgraph.graph import START, StateGraph, END

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
from langdetect import detect
import numpy as np
from deep_translator import GoogleTranslator

# ===【全域變數】===
latest_related_links: List[dict] = []

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # 實際執行時應該拋出錯誤，這裡為了不中斷程式碼流先保留
    print("[警告] 未設定 GOOGLE_API_KEY，Embeddings 可能無法正常運作。")
    pass

# ===【PostgreSQL 連線】===
conn = None
try:
    conn = psycopg2.connect(
        host="localhost",
        user="postgres",
        password="password",
        database="postgres",
        port=5432
    )
except psycopg2.OperationalError as e:
    print(f"[PostgreSQL 連線錯誤] 請檢查連線設定和資料庫狀態: {e}")
    conn = None # 如果連線失敗，將 conn 設為 None

# ===【LLM HTTP Wrapper for LMStudio】===
class LMStudioWrapper:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def invoke(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 7800
        }
        try:
            r = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            # 處理連線失敗錯誤，提供明確提示
            print(f"[LMStudio 連線錯誤] 無法連線至 {self.base_url}。請確認 LMStudio 伺服器已啟動。")
            return "LMStudio 連線失敗"
        except requests.RequestException as e:
            print(f"[LMStudio 請求錯誤] {e}")
            return "LMStudio 請求失敗或連線逾時"

# 初始化 LMStudio (請根據您的實際 IP 和埠號修改)
LMSTUDIO_MODEL = "google/gemma-3-4B"
LMSTUDIO_URL = "http://192.168.98.34:1234" # 確保這裡的 URL 是正確且唯一的
llm = LMStudioWrapper(LMSTUDIO_MODEL, LMSTUDIO_URL)

# ===【Embeddings】===
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ===【Prompt Template】===
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

# ===【資料結構】===
class Search(TypedDict):
     zh_keywords: List[str]
     en_keywords: List[str]
     keywords: List[str]

class State(TypedDict):
    title: str
    system: str
    task: str
    question: str
    query: Search
    context: List[Document]
    answer: str

# ===【Google Translate 翻譯】===
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests
import requests

def translate_with_google(text: str, source: str = "auto", target: str = "en", max_retries=3, delay=2) -> str:
    
    if not text or text.strip() == "":
        return ""
        
    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source=source, target=target)
            translated_text = translator.translate(text)
            
            # --- 關鍵：功能性檢查 ---
            # 如果翻譯結果是 None 或與原文相同，我們視為功能性失敗，並觸發重試。
            if not translated_text or translated_text.lower().strip() == text.lower().strip():
                # 這裡不拋出異常，而是讓它進入下一個判斷，以便重試
                if attempt < max_retries - 1:
                     print(f"[翻譯功能失敗] 嘗試 {attempt + 1}: 翻譯結果與原文相同，進行重試...")
                     time.sleep(delay)
                     continue # 繼續下一次嘗試
                else:
                    print(f"[翻譯功能失敗] 已達最大重試次數 ({max_retries})，返回原文。")
                    return text # 返回原文

            # ***關鍵清理***
            if translated_text:
                translated_text = translated_text.replace("聊天摘要:", "").replace("用戶訊息:", "").replace("Question:", "").replace("User Message:", "").replace(":", "").strip()
                print(f"翻譯 (成功): {translated_text}")
                
            return translated_text
            
        except TooManyRequests as e:
            # 捕獲頻率限制錯誤
            print(f"[翻譯失敗] 第 {attempt + 1} 次嘗試，錯誤：Google 翻譯請求頻率過高。")
            if attempt < max_retries - 1:
                time.sleep(delay * 2) # 延長等待時間
            else:
                return text
                
        except requests.exceptions.RequestException as e:
            # 捕獲所有網路連線、超時、HTTP 錯誤
            print(f"[翻譯失敗] 第 {attempt + 1} 次嘗試，錯誤：網路連線問題或請求錯誤 ({e})")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return text
                
        except Exception as e:
            # 捕獲其他未預期的錯誤
            print(f"[翻譯失敗] 第 {attempt + 1} 次嘗試，發生未預期錯誤：{e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return text
                
    # 這是額外的保護，確保如果 for 迴圈結束，仍然會返回文本
    return text

# ===【分析使用者查詢】===
def analyze_query(state: State):
    """
    從使用者問題萃取關鍵字 (已強化 Prompt 和清理)
    """
    question = state["question"]
    
    # 嘗試偵測語言
    try:
        lang = detect(question)
    except:
        lang = "zh"
    
    # --- 1. 萃取中文/原語言關鍵字 ---
    zh_keywords = []
    try:
        # 強化 Prompt 要求只輸出逗號分隔的關鍵字
        zh_prompt = f"請根據以下問題，萃取出使用者需要查詢文章的關鍵字。請**嚴格**只輸出一個逗號分隔的關鍵字列表，不要包含任何解釋性文字、符號、標題或句子。問題：{question}"
        zh_res = llm.invoke(zh_prompt)
        
        # 為了更強健地處理換行符、星號等，先替換並清理
        zh_res = zh_res.replace('\n', ',').replace('*', '').replace(':', '')
        
        if zh_res not in ["LMStudio 連線失敗", "LMStudio 請求失敗或連線逾時"]:
            zh_keywords = [kw.strip() for kw in zh_res.split(",") if kw.strip()]
            zh_keywords = [kw for kw in zh_keywords if kw]
    except Exception as e:
        print(f"[中文關鍵字萃取失敗] {e}")
    
    # --- 2. 翻譯問題並萃取英文關鍵字 ---
    en_keywords = []
    translated_question = ""
    
    if lang != "en":
        try:
            # 關鍵：只翻譯原始的 question
            translated_question = translate_with_google(question, target="en")
        except Exception as e:
            print(f"[翻譯錯誤] 無法將問題翻譯成英文: {e}")
            translated_question = question # 翻譯失敗則使用原文

        # 只有在翻譯結果有意義時才進行英文關鍵字萃取
        if translated_question and translated_question != question:
            try:
                en_prompt = f"Extract keywords from the following English question, strictly as a comma-separated list without any explanatory text, symbols, headers, or sentences. Question: {translated_question}"
                en_res = llm.invoke(en_prompt)
                
                en_res = en_res.replace('\n', ',').replace('*', '').replace(':', '')
                
                if en_res not in ["LMStudio 連線失敗", "LMStudio 請求失敗或連線逾時"]:
                    en_keywords = [kw.strip() for kw in en_res.split(",") if kw.strip()]
                    en_keywords = [kw for kw in en_keywords if kw]

            except Exception as e:
                print(f"[英文關鍵字萃取失敗] {e}")

    # --- 3. 合併結果 ---
    all_keywords = list(set(zh_keywords + en_keywords))

    print(f"[analyze_query] 原語言關鍵字: {zh_keywords}")
    print(f"[analyze_query] 英文關鍵字: {en_keywords}")
    
    return {
        "query": {
            "zh_keywords": zh_keywords,
            "en_keywords": en_keywords,
            "keywords": all_keywords # 方便 retrieve 函數統一使用
        }
    }

# ===【檢索函數】===
def retrieve(state: State):
    """
    PostgreSQL + pgvector 檢索 (已修正 UndefinedColumn 錯誤)
    """
    global latest_related_links

    if conn is None:
        print("[Retrieve 錯誤] PostgreSQL 連線失敗，跳過檢索。")
        return {"context": []}

    question = state["question"]
    try:
        # 這裡也要翻譯，因為這是用於 Embedding
        translated_question = translate_with_google(question)
    except Exception:
        translated_question = question
    
    query_vector = embeddings.embed_query(translated_question)
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()
    elif not isinstance(query_vector, list):
        query_vector = list(query_vector)
    
    keywords_obj = state.get("query")
    keywords = keywords_obj.get("keywords", []) if isinstance(keywords_obj, dict) else []

    cur = conn.cursor()
    top_k = 10

    if keywords:
        sql = f"""
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
        params = (keywords, query_vector, query_vector)
    else:
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
        if url and url not in related_links_set:
            latest_related_links.append({"title": title, "url": url})
            related_links_set.add(url)

    cur.close()
    return {"context": retrieved_docs}

# ===【生成回答】===
def generate(state: State):
    # 將檢索到的文章內容合併成文字
    docs_text = "\n\n".join(doc.page_content for doc in state["context"])
    
    # 準備 Prompt 參數
    prompt_input = {
        "title": state["title"],
        "system": state["system"],
        "task": state["task"],
        "context": docs_text,
        "question": state["question"]
    }
    
    # 修正點：使用 ** 解開字典，將其作為關鍵字參數傳遞
    try:
        messages = prompt.invoke(prompt_input)
        messages_str = prompt.format_prompt(**prompt_input).to_string()
        print("Prompt messages:\n", messages.text)
    except TypeError as e:
        print(f"[Prompt Format 錯誤] {e}")
        return {"answer": "提示詞格式化失敗。"}
    
    # 呼叫 LMStudio
    response_content = llm.invoke(messages_str)
    
    return {"answer": response_content}


# ===【LangGraph 對話圖】===
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# ===【取得最新連結】===
def get_latest_links() -> List[dict]:
    global latest_related_links
    return latest_related_links