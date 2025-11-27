# module_output.py

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel 
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Tuple, List

LMSTUDIO_MODEL = "google/gemma-3-4B" 
LMSTUDIO_URL = "http://192.168.98.34:1234/v1" 


def init_LMStudio(model: str, base_url: str, api_key: str = ".", configurable_fields: None = None, config_prefix: str | None = None, **kwargs) -> BaseChatModel:
    """使用 LangChain 連接至 LM Studio 的 OpenAI 相容 API"""
    return init_chat_model(model=model, base_url=base_url, configurable_fields=configurable_fields, config_prefix=config_prefix, model_provider="openai", api_key=api_key, **kwargs)

try:

    lmstudio_llm: BaseChatModel = init_LMStudio(model=LMSTUDIO_MODEL, base_url=LMSTUDIO_URL)
    print(f"[module_output] LMStudio LLM 初始化成功: {LMSTUDIO_URL}")
except Exception as e:
    print(f"[module_output LMStudio 初始化失敗] 無法連線至 {LMSTUDIO_URL}。錯誤: {e}")

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
        # Note: lmstudio_llm is already a BaseChatModel instance.
        # If module_keyRAG_v6 needs it, it should import and use it internally.
        # "llm_client": lmstudio_llm # Pass LLM instance here if graph requires it
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