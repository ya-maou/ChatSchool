# chat.py
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.utils.safestring import mark_safe
import json
import time
import markdown2
import threading 
from ..models import ChatHistory, Message

from module_slef.module_unRAG import ask_direct as ask_rag_direct
from module_slef.module_output import ask_cag, ask_rag_key
from module_slef.module_CAG import init_cag_module, add_to_cag_cache 
from module_slef.module_RAG import ask_rag
from module_slef.J1_module_KGRAG_llm import ask_enhanced_hybrid_rag
from module_slef.J2_module_GraphRAG_v1_llm import ask_question as ask_graphrag
from module_slef.J3_module_StuRAG_llm import ask_question as ask_sturag

init_cag_module()

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel 
from langchain_core.messages import HumanMessage
import os                                                             
from dotenv import load_dotenv
import random
import psycopg2
from django.shortcuts import render

# 載入.env 檔案內容
load_dotenv()
# 從環境變數中取得金鑰
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONN_STR")

def get_pg_connection():
    conn_str = POSTGRES_CONNECTION_STRING.replace("postgresql+psycopg2://", "postgresql://")
    return psycopg2.connect(conn_str)

# --- [新增] init_LMStudio 函式 ---
def init_LMStudio(model: str, base_url: str, api_key: str = ".", configurable_fields: None = None, config_prefix: str | None = None, **kwargs) -> BaseChatModel:
    """使用 LangChain 連接至 LM Studio 的 OpenAI 相容 API"""
    return init_chat_model(model=model, base_url=base_url, configurable_fields=configurable_fields, config_prefix=config_prefix, model_provider="openai", api_key=api_key, **kwargs)

llm = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")

# 確保這些變數已定義
LMSTUDIO_MODEL = "google/gemma-3-4B" 
LMSTUDIO_URL = "http://192.168.98.34:1234/v1" 

try:
    lmstudio_llm = init_LMStudio(model=LMSTUDIO_MODEL, base_url=LMSTUDIO_URL)
except Exception as e:
    # 如果連線失敗，則將其設為 None 或一個能返回錯誤的替身，但為了流程順暢，我們先使用預設 LLM 作為備用 (雖然不是最佳實踐)
    print(f"[LMStudio 初始化失敗] 無法連線至 {LMSTUDIO_URL}。錯誤: {e}")
    lmstudio_llm = llm # 暫時用 llm (Gemini) 替代，以便程式碼能繼續執行

def chat_view(request):
    if not request.user.is_authenticated:
        return render(request, 'chat.html', {
            'chat_history': [],
            'selected_chat': None,
            'messages': [],
            'user_email': None,
            'random_question': "輸入訊息"
        })

    chat_history = ChatHistory.objects.filter(user=request.user).order_by('-id')
    chat_history_id = request.GET.get('chat_history_id')

    if chat_history_id:
        selected_chat = get_object_or_404(ChatHistory, id=chat_history_id, user=request.user)
        messages = selected_chat.messages.all().order_by('timestamp')
    else:
        selected_chat = None
        messages = []

    for msg in messages:
        # 將 content 的 Markdown 轉換為 HTML
        text = msg.content.strip()
        html = markdown2.markdown(text, extras=["fenced-code-blocks", "tables", "strike"])
        msg.rendered = mark_safe(html)

    # 從 cag_cache 隨機抓一條 question
    random_question = "輸入訊息"
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        cur.execute("SELECT question FROM cag_cache ORDER BY RANDOM() LIMIT 1;")
        row = cur.fetchone()
        if row and row[0]:
            random_question = row[0]
        cur.close()
        conn.close()
    except Exception as e:
        print("DB 取隨機問題失敗:", e)

    return render(request, 'chat.html', {
        'chat_history': chat_history,
        'selected_chat': selected_chat,
        'messages': messages,
        'user_email': request.user.username,
        'random_question': random_question
    })

@csrf_exempt
def send_message(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get('content', '').strip()
        chat_history_id = data.get('chat_history_id')
        if not user_message:
            return JsonResponse({'status': 'error', 'message': '內容不得為空'}, status=400)
        if chat_history_id:
            try:
                chat_history = ChatHistory.objects.get(id=chat_history_id, user=request.user)
                Message.objects.create(
                    chat_history=chat_history,
                    username='user',
                    content=user_message
                )
                return JsonResponse({'status': 'success', 'message': '訊息已新增至既有對話'})
            except ChatHistory.DoesNotExist:
                return JsonResponse({'status': 'error', 'message': '找不到指定對話或對話不屬於當前用戶'}, status=404)
        else:
            new_chat = ChatHistory.objects.create(
                user=request.user,
                title=user_message[:30] or '新對話'
            )
            Message.objects.create(
                chat_history=new_chat,
                username='user',
                content=user_message
            )
            return JsonResponse({'status': 'success', 'message': '已建立新對話', 'chat_history_id': new_chat.id})
    return JsonResponse({'status': 'error', 'message': '只允許 POST 請求'}, status=405)

def classify_question(question: str) -> str:
    """
    根據使用者的訊息內容判斷使用哪個模型回答
    回傳：
        - "direct"：可直接回答
        - "cag"：使用快速檢索 (CAG)
        - "rag"：一般檢索 (純語義檢索)
        - "rag_key"：關鍵字檢索 (語義 + 關鍵字檢索)
        - "kgrag"：知識圖譜增強檢索 (Weaviate + Neo4j + PRF)
        - "graphrag"：社群圖譜檢索 (Global/Local Search + 社群摘要)
        - "sturag"：學生專用優化檢索 (向量搜尋 + Neo4j 增強)
    """
    llm_to_use = lmstudio_llm 
    
    llm_prompt = f"""
You are a routing assistant for a Q&A system. Your task is to strictly classify the following user question into one of seven categories based on its content and retrieval needs:

1.  If the user's message is general, non-school related, does not require retrieval, or asks a subjective question: return 'direct'
2.  If the user's message concerns **basic, common, and frequently asked campus information (CAG - Campus Answer Guide)**: return 'cag'
3.  If the user's message requires further data retrieval (RAG) and is best handled by **pure semantic vector search**: return 'rag'
4.  If the user's message requires data retrieval (RAG) and can be significantly improved by **keyword analysis (semantic + keyword search)**: return 'rag_key'
5.  If the user's message requires **complex information retrieval with knowledge graph enhancement, graph relationship analysis, and pseudo-relevance feedback (PRF)** for comprehensive answers: return 'kgrag'
6.  If the user's message requires **community-based global search or hierarchical knowledge exploration**, such as "What are all the departments?", "Overall structure?", "All related services?": return 'graphrag'
7.  If the user's message is **student-specific queries requiring optimized vector search with Neo4j enhancement**: return 'sturag'

User's message:
{question}

You **MUST ONLY** respond with one of the following exact strings: 'direct', 'cag', 'rag', 'rag_key', 'kgrag', 'graphrag', or 'sturag'
"""
    
    try:
        response = llm_to_use.invoke(llm_prompt)
        model_choice = response.content.strip().lower()
        print(f"[LLM Classification] Question classification result: {model_choice}")
    except Exception as e:
        print(f"[LLM Classification ERROR] {e}. Falling back to 'rag_key'.")
        model_choice = "rag_key" 
    
    # 修改驗證列表
    if model_choice not in ["direct", "cag", "rag", "rag_key", "kgrag", "graphrag", "sturag"]:
        return "rag_key"
        
    return model_choice


@csrf_exempt
def chat_api(request):
    """主要聊天 API：自動選擇模型 (CAG / RAG / Keyword-RAG / Direct)，並進行 fallback 機制"""

    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()
        chat_id = data.get("chat_history_id")
        selected_model = data.get("model", "rag")
    except Exception:
        return JsonResponse({"error": "Invalid JSON format"}, status=400)

    if not user_message:
        return JsonResponse({"error": "No message provided"}, status=400)

    # === 建立或取得聊天歷史 ===
    history = None
    if chat_id:
        history = ChatHistory.objects.filter(id=chat_id, user=request.user).first()
    if not history:
        history = ChatHistory.objects.create(
            user=request.user,
            title=user_message[:50] or "New Chat"
        )

    # === 儲存使用者訊息 ===
    Message.objects.create(
        chat_history=history,
        username="user",
        content=user_message
    )

    # === 模型選擇 ===
    if selected_model in [None, "auto", "auto-detect"]:
        selected_model = classify_question(user_message)

    history_summary = history.summary or ""
    prompt = f"聊天摘要:\n{history_summary}\n\n用戶訊息:\n{user_message}"

    response_content, related_links, duration = "", [], 0

    def run_model(func, *args, **kwargs):
        """統一計算執行時間與回傳結構"""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        if isinstance(result, tuple):
            return (*result, elapsed)
        else:
            return result, [], elapsed

    # === 模型呼叫區 ===
    if selected_model == "direct":
        response_content, related_links, duration = run_model(ask_rag_direct, prompt)

    elif selected_model == "cag":
        response_content, related_links, duration = run_model(ask_cag, prompt)

        # --- fallback 層級 ---
        if "Not Found" in response_content or not response_content.strip():
            print("[CAG] No answer found → fallback to RAG")
            response_content, related_links, duration = run_model(ask_rag, prompt)

            if "Not Found" in response_content or not response_content.strip():
                print("[RAG] No answer found → fallback to Keyword-RAG")
                response_content, related_links, duration = run_model(ask_rag_key, prompt)

    elif selected_model == "rag":
        response_content, related_links, duration = run_model(ask_rag, prompt)

        if "Not Found" in response_content or not response_content.strip():
                print("[RAG] No answer found → fallback to Keyword-RAG")
                response_content, related_links, duration = run_model(ask_rag_key, prompt)

    elif selected_model == "rag_key":
        response_content, related_links, duration = run_model(ask_rag_key, prompt)
    elif selected_model == "kgrag":
        start = time.time()
        # KGRAG 現在回傳 (answer, related_links)
        response_content, related_links = ask_enhanced_hybrid_rag(prompt)
        duration = time.time() - start
    elif selected_model == "graphrag":
        start = time.time()
        # GraphRAG 回傳 (answer, related_links)
        response_content, related_links = ask_graphrag(prompt)
        duration = time.time() - start
    elif selected_model == "sturag":
        start = time.time()
        response_content, related_links = ask_sturag(prompt)
        duration = time.time() - start
    else:
        return JsonResponse({"error": f"Invalid model: {selected_model}"}, status=400)


    # === 處理圖片連結 ===
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
    image_links = [
        link["url"] for link in related_links
        if link and "url" in link and link["url"].lower().endswith(image_extensions)
    ]

    # === 儲存機器人回應 ===
    # 這是最後一個必須同步完成的步驟，之後可以回傳
    Message.objects.create(
        chat_history=history,
        username="bot",
        content=response_content,
        model_name=selected_model,
        related_links=related_links or [],
        image_links=image_links or [],
        duration=duration
    )

    # *** 新增：將耗時的後續任務移至背景執行緒 ***
    thread = threading.Thread(
        target=post_response_tasks,
        args=(history, user_message, response_content, selected_model)
    )
    thread.start()
    # **********************************************

    # === 回傳結果 (立即回傳，不等待背景任務) ===
    return JsonResponse({
        "model": selected_model,
        "response": response_content,
        "related_links": related_links,
        "image_links": image_links,
        "chat_history_id": history.id,
        "duration": round(duration, 3)
    })

# ==============================================================================
# 後台非同步任務區 (已修改為使用 LangChain BaseChatModel)
# ==============================================================================

def post_response_tasks(history, user_message, response_content, selected_model):
    """
    在回傳給使用者後，於背景執行的耗時任務：
    1. 判斷並寫入 CAG 快取 (涉及 LLM 呼叫)。
    2. 更新聊天摘要 (涉及 LLM 呼叫)。
    """
    time.sleep(10)
    # === 1. CAG 快取寫入邏輯 (包含 LLM 判斷) ===
    is_valid_cag_answer = not "Not Found" in response_content and response_content.strip()

    if is_valid_cag_answer:
        # 判斷問題是否為公共、非私人性質，且回答是否有效
        public_check_prompt = f"""
        請先判斷以下問題是否屬於**通用、公用**的**校園常識或知識**，而非用戶個人隱私、私人情感或單次聊天脈絡的客製化問題。
        然後，判斷如果回答是「找不到資料」或無法提供資訊，請視為無效回答。
        
        規則：
        1. 如果問題屬於公共校園知識且回答可提供有效資訊，回傳 'YES'。
        2. 否則（私人問題或回答無效），回傳 'NO'。
        
        用戶問題：{user_message}
        """
        
        try:
            # *** 替換點 1: 使用 lmstudio_llm.invoke ***
            # LangChain BaseChatModel.invoke() 返回 Message 物件
            public_check_response = lmstudio_llm.invoke(public_check_prompt)
            public_check_response_content = public_check_response.content
            
            print(f"[CAG LLM Response] {public_check_response_content.strip()}")
            
            # 處理可能的錯誤回覆 (LangChain 的 invoke 在連線失敗時通常會拋出 Exception，而不是回傳錯誤字串)
            is_public_question = public_check_response_content.strip().upper() == 'YES'

        except Exception as e:
            print(f"[CAG Cache Check ERROR] LLM classification failed (LMStudio might be down): {e}")
            is_public_question = False

        if is_public_question:
            add_to_cag_cache(user_message, response_content)
            # print("[CAG] 存入快取")

    # === 2. 更新聊天摘要 (需呼叫 LLM) ===
    try:
        # 傳遞 lmstudio_llm 實例
        update_chat_summary(history, lmstudio_llm) 
        print("[Summary] Chat summary updated in background.")
    except Exception as e:
        print(f"[Summary ERROR] Failed to update chat summary: {e}")


# ==============================================================================
# 輔助函式區 (已修改為使用 LangChain BaseChatModel)
# ==============================================================================

MAX_SUMMARY_LENGTH = 5000 

# 修正：接受 llm 實例作為參數，其類型為 BaseChatModel
def update_chat_summary(history: ChatHistory, llm_model: BaseChatModel):
    time.sleep(10)
    """
    使用滾動摘要更新 ChatHistory.summary
    - 只使用舊摘要 + 最新訊息
    - 控制摘要長度
    """
    # 由於在後台執行緒中操作，確保資料庫 ORM 查詢的獨立性是安全的
    messages = history.messages.all().order_by('timestamp')
    if not messages.exists():
        return

    # 取最後一條訊息（bot 或 user）
    latest_msg = messages.last()
    if not latest_msg:
        return

    # 滾動摘要 prompt：舊摘要 + 最新訊息
    old_summary = history.summary or ""
    rolling_text = f"舊摘要:\n{old_summary}\n\n最新訊息:\n{latest_msg.username}: {latest_msg.content}"

    summary_prompt = f"""
請將以下聊天內容整理成結構化摘要，保留主要問題、回答及重點：
{rolling_text}
"""
    try:
        # *** 替換點 2: 使用傳入的 llm_model.invoke ***
        summary_response = llm_model.invoke(summary_prompt)
        summary_response_content = summary_response.content
    except Exception as e:
        print(f"[Summary WARNING] LLM 呼叫失敗，跳過摘要更新: {e}")
        return

    new_summary = summary_response_content.strip()

    # 長度控制
    if len(new_summary) > MAX_SUMMARY_LENGTH:
        # 裁切前面 + 再用 LLM 壓縮
        compress_prompt = f"""
摘要太長，請將以下摘要壓縮到 {MAX_SUMMARY_LENGTH} 字以內，保留主要問題與回答：
{new_summary}
"""
        try:
            # *** 替換點 3: 使用傳入的 llm_model.invoke ***
            compress_response = llm_model.invoke(compress_prompt)
            compress_response_content = compress_response.content
            new_summary = compress_response_content.strip()
        except Exception as e:
            print(f"[Summary WARNING] LLM 壓縮失敗，使用原始摘要 (可能過長): {e}")

    # 更新資料庫
    history.summary = new_summary
    history.save()

def chat_history_api(request):
    if request.user.is_authenticated:
        chat_histories = ChatHistory.objects.filter(user=request.user).order_by('-id').values('id', 'title')
        return JsonResponse(list(chat_histories), safe=False)
    else:
        return JsonResponse({'status': 'error', 'message': '未登入，無法查看聊天紀錄'}, status=401)

@require_http_methods(["DELETE"])
@login_required
def delete_chat_history(request, chat_id):
    try:
        chat = ChatHistory.objects.get(id=chat_id, user=request.user)
        chat.delete()
        return JsonResponse({'success': True})
    except ChatHistory.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Chat not found'}, status=404)
    
def get_links(request):
    """
    提供給前端，以取得最近一次問答的相關連結。
    """
    if request.method == 'GET':
        # 這裡需要一個函式來取得 links
        # 由於您沒有提供 get_latest_links，這裡暫時假設它能被呼叫
        try:
            links = get_latest_links()
        except NameError:
            # Placeholder for missing function
            links = [] 
        return JsonResponse({'links': links})
    return JsonResponse({'error': 'Invalid request method'}, status=400)