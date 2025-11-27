# module_output.py

# 匯入不同版本的 graph，使用別名區分
from .module_keyRAG_v6 import graph as graph_v6, get_latest_links as get_links_v6

from typing import Tuple, List

def ask_rag_key(question: str) -> Tuple[str, List[dict]]:
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
            "【無資料處理】: 如果找不到相關資料，請使用**使用者提問的語言**向使用者誠實說明找不到相關資料。"
        ),
        "question": question
    }

    # 選擇 graph 與對應連結函數
    selected_graph = graph_v6
    get_links = get_links_v6

    # 呼叫 RAG 流程
    result = selected_graph.invoke(input_state)
    raw_answer = result.get("answer", "")
    clean_answer = raw_answer.replace("\\u000A", "\n")

    # 取得最新文章連結
    links = get_links()

    return clean_answer, links

from .module_CAG import ask_cag          # <-- 新增/替換成正確的函式名稱

def ask_cag_wrapper(question: str) -> Tuple[str, list]: # 為了避免與匯入的 ask_cag 衝突，這裡使用 wrapper 名稱
    """
    CAG 快速檢索方法：
    主要針對校園簡介、系所、地點、行政單位等固定資訊。
    """
    # answer, links = get_campus_info(question)  <-- 刪除或註釋掉這一行
    answer = ask_cag(question)       # <-- 使用新的正確函式名稱
    return answer
