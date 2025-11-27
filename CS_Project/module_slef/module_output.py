from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel 
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Tuple, List

#移除 LM Studio 相關設定，改用 Gemini 模型名稱
GEMINI_MODEL = "gemini-2.5-flash" 

#移除 init_LMStudio 函式，改用通用的 init_chat_model
# def init_LMStudio(...) -> BaseChatModel:
#     ...

try:
    #直接初始化 Gemini 作為 RAG 流程的主要模型
    lmstudio_llm: BaseChatModel = init_chat_model(model=GEMINI_MODEL, model_provider="google_genai")
    print(f"[module_output] LLM 初始化成功: {GEMINI_MODEL}")
except Exception as e:
    print(f"[module_output LLM 初始化失敗] 無法連線至 {GEMINI_MODEL}。錯誤: {e}")
    # 如果初始化失敗，將其設為 None 讓上層邏輯處理
    lmstudio_llm: BaseChatModel = None 

from .module_keyRAG_v6 import graph as graph_v6, get_latest_links as get_links_v6

def ask_rag_key(question: str) -> Tuple[str, List[dict]]:
    """
    Question-answering function
    :param question: User question
    :return: Tuple(answer_text, latest_related_links)
    """

    input_state = {
        "title": "NUU Information Retrieval QA",
        #(建議修改) 由於切換到 Gemini，請將這些指令改為中文以增強中文問題的語言遵循性
        "system": (
            "You are a professional university assistant responsible for answering questions related to the NUU website.\n"
            "[Highest Priority Rule]: You must strictly answer in the same language as the user query.\n"
        ),
        # Language switching rules are explicitly defined
        "task": (
            "[Language Rule]: Strictly respond in the same language as the user question "
            "(e.g., if the question is in English, answer in English; if in Japanese, answer in Japanese). "
            "[Main Task]: Provide as detailed an answer as possible based on the provided information. "
            "[No Data Handling]: If no relevant information is found, honestly inform the user using the same language as their question."
        ),
        "question": question
        # Note: module_keyRAG_v6 現在會使用這個 lmstudio_llm (即 Gemini)
    }

    # Select graph and corresponding link function
    selected_graph = graph_v6
    get_links = get_links_v6

    # Invoke the RAG process
    result = selected_graph.invoke(input_state)
    raw_answer = result.get("answer", "")
    clean_answer = raw_answer.replace("\\u000A", "\n")

    # Get the latest article links
    links = get_links()

    return clean_answer, links

from .module_CAG import ask_cag           

def ask_cag_wrapper(question: str) -> Tuple[str, list]: 
    """
    CAG 快速檢索方法：
    主要針對校園簡介、系所、地點、行政單位等固定資訊。
    """
    answer = ask_cag(question)       
    # 保持回傳類型一致性
    return answer, []