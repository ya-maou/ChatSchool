#新0804data - ContentSummaryEn0913 集合 (英文摘要) + Neo4j 圖譜資料
#用戶問題 → 向量檢索 → 獲得文檔 → 從文檔內容提取實體 → 查詢三元組
from typing import List, TypedDict, Optional, Dict
from langchain_core.documents.base import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
import weaviate
from weaviate import connect_to_local
import weaviate.classes as wvc
import google.generativeai as genai
from neo4j import GraphDatabase
import os
import dotenv
import time
import re
import requests
import traceback
import json

# 加載環境變數
dotenv.load_dotenv()

# 設定 API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MISTRAL_API_KEY = os.getenv("J_MISTRAL_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Setup LLM
from langchain.chat_models import init_chat_model
llm: BaseChatModel = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")

# 檢查文本是否包含中文字符
def is_chinese(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(chinese_pattern.search(text))

# 調用 Mistral API
def call_mistral(prompt, max_retries=3, base_delay=5):
    """調用 Mistral API（加入 429 錯誤處理）"""
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
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # 超過容量限制，等待後重試
                wait_time = base_delay * (2 ** attempt)  # 指數退避
                print(f"⚠️ API 容量限制 (429)，等待 {wait_time} 秒後重試... (嘗試 {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise Exception(f"{response.status_code}, {response.text}")
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Mistral API 錯誤：{e}")
            else:
                print(f"⚠️ 請求失敗，{base_delay} 秒後重試...")
                time.sleep(base_delay)
    
    raise Exception("超過最大重試次數")
def choose_search_strategy_with_mistral(question: str) -> str:
    """使用 Mistral 判斷應該使用哪種搜尋策略"""
    
    prompt = f"""請判斷以下問題應該使用哪種搜尋策略：

問題：{question}

搜尋策略說明：
1. **Global Search (全局搜尋)**：
   - 適用於需要理解整個資料集、多個來源的問題
   - 例如：「學校有哪些學院？」「整體架構如何？」「所有的系所」
   - 需要綜合多個社群的資訊

2. **Local Search (局部搜尋)**：
   - 適用於針對特定實體、具體細節的問題
   - 例如：「資工系的電話」「某某大樓的地址」「如何聯絡XX部門」
   - 需要精確定位特定資訊

請只回答 "global" 或 "local"，不要其他說明："""
    
    try:
        response = call_mistral(prompt)
        strategy = response["choices"][0]["message"]["content"].strip().lower()
        
        # 清理回應（移除可能的引號或多餘文字）
        if 'global' in strategy:
            return 'global'
        elif 'local' in strategy:
            return 'local'
        else:
            # 預設使用 local（更安全）
            print(f"⚠️ Mistral 回應不明確: {strategy}，預設使用 local")
            return 'local'
            
    except Exception as e:
        print(f"❌ Mistral 策略判斷失敗: {e}，預設使用 local")
        return 'local'
    
# 使用 Mistral 提取實體
def extract_entities_with_mistral(text: str) -> List[str]:
    """使用 Mistral 從文字中提取實體名稱"""
    prompt = f"""請從以下文字中提取重要的實體名稱，特別是：
    - 學校名稱（如：國立聯合大學）
    - 學院名稱（如：資訊學院）
    - 系所名稱（如：資訊工程系）
    - 部門名稱（如：學務處、教務處）
    - 中心名稱（如：計算機中心）
    
    文字：{text}
    
    請只回傳實體名稱，每個實體一行，不要其他說明："""
    
    try:
        response = call_mistral(prompt)
        entities_text = response["choices"][0]["message"]["content"].strip()
        
        # 分割成個別實體
        entities = [entity.strip() for entity in entities_text.split('\n') if entity.strip()]
        
        # 過濾掉太短的實體
        filtered_entities = [e for e in entities if len(e) >= 2 and len(e) <= 20]
        
        print(f"🤖 Mistral提取到實體: {filtered_entities}")
        return filtered_entities
        
    except Exception as e:
        print(f"❌ Mistral實體提取失敗: {e}")
        return []
def extract_concepts_for_global_search(text: str) -> dict:
    """針對 Global Search 提取分層概念（核心 + 輔助）
    
    Returns:
        {
            'core': ['核心概念1', '核心概念2'],      # 必須匹配
            'auxiliary': ['輔助概念1', '輔助概念2']  # 加分項
        }
    """
    prompt = f"""請分析以下問題，提取關鍵概念並分為兩類：

問題：{text}

請分類為：
1. **核心概念**（必須匹配，1-3個）：查詢的主要對象或核心主題
   - 例如問「有哪些科系」→ 核心概念是「科系」「學系」「學院」
   - 例如問「學生服務」→ 核心概念是「服務」「學生」
   
2. **輔助概念**（加分項，2-4個）：相關的領域、屬性或背景詞彙
   - 例如問「有哪些科系」→ 輔助概念是「學術單位」「教育」「教學」
   - 例如問「學生服務」→ 輔助概念是「行政」「支援」「資源」

回答格式（嚴格遵守）：
核心：概念1, 概念2, 概念3
輔助：概念4, 概念5

回答："""
    
    try:
        response = call_mistral(prompt)
        content = response["choices"][0]["message"]["content"].strip()
        
        # 解析回應
        core_concepts = []
        auxiliary_concepts = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('核心：') or line.startswith('核心:'):
                core_part = line.split('：')[-1].split(':')[-1]
                core_concepts = [c.strip() for c in core_part.split(',') if c.strip()]
            elif line.startswith('輔助：') or line.startswith('輔助:'):
                aux_part = line.split('：')[-1].split(':')[-1]
                auxiliary_concepts = [c.strip() for c in aux_part.split(',') if c.strip()]
        
        # 過濾長度
        core_concepts = [c for c in core_concepts if 2 <= len(c) <= 15][:3]
        auxiliary_concepts = [c for c in auxiliary_concepts if 2 <= len(c) <= 15][:4]
        
        print(f"🔑 核心概念: {core_concepts}")
        print(f"➕ 輔助概念: {auxiliary_concepts}")
        
        return {
            'core': core_concepts,
            'auxiliary': auxiliary_concepts
        }
        
    except Exception as e:
        print(f"❌ 概念提取失敗: {e}")
        # 降級：使用原始方法
        basic_concepts = text.replace('?', '').replace('？', '').split()
        return {
            'core': basic_concepts[:3],
            'auxiliary': []
        }
# 翻譯管理器 - 支援外部翻譯檔案
class TranslationManager:

    def __init__(self, translation_file_path=r"CS_Project\CS_App\translate\translation_mapping.json"):
        self.translation_file = translation_file_path
        self.translation_dict = self._load_translation_dict()
    
    # 直接從 self.translation_file 讀取翻譯映射檔案,不存在則報錯
    def _load_translation_dict(self) -> dict:
        try:
            with open(self.translation_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            print(f"✅ 成功載入翻譯映射檔案,條目數:{len(mapping)}")
            return mapping
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到翻譯映射檔案:{self.translation_file},請先建立檔案。")
        except Exception as e:
            print(f"❌ 載入翻譯映射失敗: {e}")
            return {}
    
    # 從映射檔案中獲取翻譯
    def get_translation(self, chinese_text: str) -> Optional[str]:
        return self.translation_dict.get(chinese_text)
    
    # 使用映射檔案進行翻譯,未命中則使用Mistral
    def translate_with_mapping(self, text: str) -> str:
        # 先檢查映射檔案
        direct_translation = self.get_translation(text)
        if direct_translation:
            print(f"📄 映射檔案翻譯: {text} → {direct_translation}")
            return direct_translation
        
        # 檢查是否包含映射中的詞語
        translated_parts = []
        remaining_text = text
        
        for chinese_term, english_term in self.translation_dict.items():
            if chinese_term in remaining_text:
                remaining_text = remaining_text.replace(chinese_term, f"__TRANSLATED_{len(translated_parts)}__")
                translated_parts.append((chinese_term, english_term))
        
        if translated_parts:
            # 部分命中,組合翻譯
            final_translation = remaining_text
            for i, (chinese_term, english_term) in enumerate(translated_parts):
                final_translation = final_translation.replace(f"__TRANSLATED_{i}__", english_term)
            
            # 如果還有剩餘中文,用Mistral翻譯
            if is_chinese(final_translation):
                final_translation = self._mistral_translate(final_translation)
            
            print(f"🔄 混合翻譯: {text} → {final_translation}")
            return final_translation
        
        # 完全未命中,使用Mistral
        return self._mistral_translate(text)
    
    # 使用Mistral進行翻譯
    def _mistral_translate(self, text: str) -> str:
        try:
            response = call_mistral(f"請將以下中文翻譯成英文,只回傳翻譯結果:{text}")
            translated = response["choices"][0]["message"]["content"].strip()
            if translated.startswith('"') and translated.endswith('"'):
                translated = translated[1:-1]
            print(f"🤖 Mistral翻譯: {text} → {translated}")
            return translated
        except Exception as e:
            print(f"❌ Mistral翻譯失敗: {e}")
            return text

# 初始化翻譯管理器
translation_manager = TranslationManager()

# 使用翻譯管理器進行翻譯（映射檔案優先，然後Mistral）
def translate_with_mistral(text: str, max_retries=3, delay=2) -> str:
    for attempt in range(max_retries):
        try:
            return translation_manager.translate_with_mapping(text)
        except Exception as e:
            print(f"❌ [翻譯失敗] 第 {attempt + 1} 次嘗試，錯誤：{e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    raise Exception("翻譯失敗，超過最大重試次數")

# Gemini 嵌入器
class GeminiEmbedder:
    def embed_query(self, text: str) -> List[float]:
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            return response['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 768  

# 初始化 Weaviate 和 Neo4j 連接
class ContentSummaryEn0913RAG:
    
    def __init__(self):
        try:
            # 初始化 Weaviate 連接
            self.client = connect_to_local()
            print("✅ 成功連接到 Weaviate")
            
            # 初始化 Neo4j 連接
            self.neo4j_driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "neo4j0804"),
                database="nuu-data-0804"
            )
            print("✅ 成功連接到 Neo4j")
            
            self.triplet_driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "neo4j0804"),
                database="nuu-triplet"
            )
            print("✅ 成功連接到 Neo4j 三元組圖譜")
            
            # 初始化嵌入器
            self.embedder = GeminiEmbedder()
            print("✅ 初始化 Gemini 嵌入器")
            
            self._check_community_status()
            
            # 只載入 ContentSummaryEn0913 集合
            try:
                self.collection = self.client.collections.get("ContentSummaryEn0913")
                self.text_field = "english_summary"
                print(f"📚 已載入 ContentSummaryEn0913 集合 (文本欄位: {self.text_field})")
            except Exception as e:
                print(f"⚠️ 無法載入 ContentSummaryEn0913 集合: {e}")
                self.collection = None
            
        except Exception as e:
            print(f"❌ 系統初始化失敗: {e}")
            self.client = None
            self.neo4j_driver = None
            self.embedder = None
            self.collection = None
    def _check_community_status(self):
        """檢查社群資料是否已建構"""
        try:
            with self.triplet_driver.session(database="nuu-triplet") as session:
                result = session.run("""
                    MATCH (c:Community)
                    WHERE c.summary IS NOT NULL
                    RETURN count(c) as count
                """)
                count = result.single()['count']
                
                if count == 0:
                    print("⚠️ 警告：未找到社群摘要資料")
                    print("   請先執行 build_communities.py")
                else:
                    print(f"✅ 已載入 {count} 個社群摘要")
        except:
            print("⚠️ 無法檢查社群狀態")
    def _determine_best_level_with_mistral(self, query: str) -> int:
        """使用 Mistral 判斷最佳檢索層級
        
        Returns:
            0: 底層（具體實體細節）
            1: 中層（類別、部門層級）
            2: 高層（全局、架構層級）
        """
        prompt = f"""請判斷以下問題應該在知識圖譜的哪個層級檢索：

    問題：{query}

    層級說明：
    - **Level 2 (高層級)**：回答需要整體架構、全局視角、跨領域綜合資訊
    例如：「學校有哪些學院？」「整體組織架構」「所有的院系」
    
    - **Level 1 (中層級)**：回答需要某個類別、某個領域的綜合資訊
    例如：「有哪些科系？」「工程相關的系所」「學生服務有哪些？」
    
    - **Level 0 (底層級)**：回答需要具體實體的詳細資訊
    例如：「資工系的電話」「某某大樓在哪裡」「如何聯絡XX部門」

    請只回答數字 0、1 或 2，不要其他說明："""
        
        try:
            response = call_mistral(prompt)
            level_str = response["choices"][0]["message"]["content"].strip()
            
            # 提取數字
            import re
            match = re.search(r'[0-2]', level_str)
            if match:
                level = int(match.group())
                print(f"🎯 Mistral 選擇層級: Level {level}")
                return level
            else:
                print(f"⚠️ Mistral 回應不明確: {level_str}，預設使用 Level 1")
                return 1
                
        except Exception as e:
            print(f"❌ Mistral 層級判斷失敗: {e}，預設使用 Level 1")
            return 1     
    def find_relevant_communities(self, query: str, limit: int = None, level: int = None) -> List[dict]:
        """根據查詢找出相關社群（改進版：支援智能層級 + 分層概念）
        
        Args:
            query: 查詢問題
            limit: 返回的社群數量（None = 自動根據層級決定）
            level: 指定搜尋的層級（None = 自動選擇最佳層級）
        """
        if not self.triplet_driver:
            return []
        
        # 1. 智能選擇層級
        if level is None:
            level = self._determine_best_level_with_mistral(query)
        else:
            print(f"🎯 手動指定層級: Level {level}")
        
        # 2. 根據層級決定檢索數量
        if limit is None:
            if level >= 2:
                limit = 5   # 高層級社群較少
            elif level == 1:
                limit = 8   # 中層級需要更多
            else:
                limit = 15  # 底層級可能需要很多
        
        # 3. 提取分層概念
        concepts_dict = extract_concepts_for_global_search(query)
        core_concepts = concepts_dict.get('core', [])
        auxiliary_concepts = concepts_dict.get('auxiliary', [])
        
        if not core_concepts:
            print("⚠️ 未提取到核心概念，使用關鍵詞匹配")
            return self._find_communities_by_keywords(query, limit)
        
        # 4. 定義學校相關關鍵詞
        school_keywords = ['國立聯合大學', 'National United University', '聯大', 'NUU']
        
        # 5. 根據層級調整查詢策略
        if level >= 2:
            # 高層級：只要核心概念匹配，學校關鍵詞可選
            where_clause = "WHERE core_matches > 0"
            order_clause = "ORDER BY c.level ASC, core_matches DESC, aux_matches DESC, school_matches DESC"
        elif level == 1:
            # 中層級：核心概念必須，學校關鍵詞加分
            where_clause = "WHERE core_matches > 0"
            order_clause = "ORDER BY c.level ASC, core_matches DESC, school_matches DESC, aux_matches DESC"
        else:
            # 底層級：核心概念 + 學校關鍵詞都要匹配
            where_clause = "WHERE core_matches > 0 AND school_matches > 0"
            order_clause = "ORDER BY c.level ASC, school_matches DESC, core_matches DESC, aux_matches DESC"
        
        # 6. 構建查詢
        query_cypher = f"""
        MATCH (c:Community {{level: $level}})
        WHERE c.summary IS NOT NULL
        
        WITH c, $core_concepts as core_concepts, 
            $aux_concepts as aux_concepts, 
            $school_keywords as school_keywords
        
        // 計算各類匹配數
        WITH c, 
            SIZE([core IN core_concepts WHERE c.summary CONTAINS core]) as core_matches,
            SIZE([aux IN aux_concepts WHERE c.summary CONTAINS aux]) as aux_matches,
            SIZE([school IN school_keywords WHERE c.summary CONTAINS school]) as school_matches
        
        {where_clause}
        
        RETURN 
            c.id as community_id,
            c.level as level,
            c.summary as summary,
            c.entity_count as entity_count,
            core_matches as core_relevance,
            aux_matches as aux_relevance,
            school_matches as school_relevance
        {order_clause}, c.entity_count DESC
        LIMIT $limit
        """
        
        try:
            with self.triplet_driver.session(database="nuu-triplet") as session:
                result = session.run(
                    query_cypher,
                    core_concepts=core_concepts,
                    aux_concepts=auxiliary_concepts,
                    school_keywords=school_keywords,
                    level=level,
                    limit=limit
                )
                communities = [dict(record) for record in result]
                
                if not communities:
                    print(f"⚠️ Level {level} 無匹配結果，嘗試回退策略...")
                    return self._fallback_multi_level_search(query, level, limit)
                
                print(f"✅ 在 Level {level} 找到 {len(communities)} 個相關社群")
                for i, comm in enumerate(communities[:3], 1):
                    preview = comm['summary'][:80] + "..." if len(comm['summary']) > 80 else comm['summary']
                    print(f"  [{i}] 社群 {comm['community_id']} (核心:{comm['core_relevance']}, 輔助:{comm['aux_relevance']}, 學校:{comm['school_relevance']})")
                    print(f"      {preview}")
                
                return communities
                
        except Exception as e:
            print(f"❌ 查詢相關社群失敗: {e}")
            traceback.print_exc()
            return []
    def _fallback_multi_level_search(self, query: str, original_level: int, limit: int) -> List[dict]:
        """當指定層級無結果時，嘗試相鄰層級"""
        print(f"🔄 執行多層級回退搜尋（原層級: {original_level}）")
        
        # 定義回退順序
        if original_level == 2:
            fallback_levels = [1, 0]
        elif original_level == 1:
            fallback_levels = [2, 0]
        else:  # original_level == 0
            fallback_levels = [1, 2]
        
        all_communities = []
        
        for level in fallback_levels:
            print(f"   嘗試 Level {level}...")
            
            concepts_dict = extract_concepts_for_global_search(query)
            core_concepts = concepts_dict.get('core', [])
            auxiliary_concepts = concepts_dict.get('auxiliary', [])
            school_keywords = ['國立聯合大學', 'National United University', '聯大', 'NUU']
            
            # 放寬條件：只要核心概念匹配即可
            query_cypher = """
            MATCH (c:Community {level: $level})
            WHERE c.summary IS NOT NULL
            
            WITH c, $core_concepts as core_concepts
            WITH c, 
                SIZE([core IN core_concepts WHERE c.summary CONTAINS core]) as core_matches
            
            WHERE core_matches > 0
            
            RETURN 
                c.id as community_id,
                c.level as level,
                c.summary as summary,
                c.entity_count as entity_count,
                core_matches as core_relevance
            ORDER BY core_matches DESC, c.entity_count DESC
            LIMIT $limit
            """
            
            try:
                with self.triplet_driver.session(database="nuu-triplet") as session:
                    result = session.run(
                        query_cypher,
                        core_concepts=core_concepts,
                        level=level,
                        limit=max(3, limit // 2)  # 每層級取較少數量
                    )
                    level_communities = [dict(record) for record in result]
                    
                    if level_communities:
                        print(f"      ✅ Level {level} 找到 {len(level_communities)} 個社群")
                        all_communities.extend(level_communities)
                        
                        # 如果已經找到足夠數量，停止回退
                        if len(all_communities) >= 3:
                            break
            except Exception as e:
                print(f"      ❌ Level {level} 查詢失敗: {e}")
                continue
        
        # 去重（根據 community_id）
        seen_ids = set()
        unique_communities = []
        for comm in all_communities:
            if comm['community_id'] not in seen_ids:
                seen_ids.add(comm['community_id'])
                unique_communities.append(comm)
        
        print(f"✅ 多層級回退完成，總共找到 {len(unique_communities)} 個社群")
        return unique_communities[:limit]
    def _find_communities_without_school_filter(self, concepts: List[str], limit: int) -> List[dict]:
        """降級查詢：只匹配概念，不過濾學校"""
        query_cypher = """
        MATCH (c:Community)
        WHERE c.summary IS NOT NULL
        WITH c, $concepts as concepts
        WITH c, 
            SIZE([concept IN concepts WHERE c.summary CONTAINS concept]) as concept_matches
        WHERE concept_matches > 0
        RETURN 
            c.id as community_id,
            c.summary as summary,
            c.entity_count as entity_count,
            concept_matches as relevance
        ORDER BY relevance DESC, c.entity_count DESC
        LIMIT $limit
        """
        
        try:
            with self.triplet_driver.session(database="nuu-triplet") as session:
                result = session.run(query_cypher, concepts=concepts, limit=limit)
                communities = [dict(record) for record in result]
                print(f"⚠️ 降級查詢找到 {len(communities)} 個社群（未過濾學校）")
                return communities
        except Exception as e:
            print(f"❌ 降級查詢失敗: {e}")
            return []
    
    def _find_communities_by_keywords(self, query: str, limit: int = 5) -> List[dict]:
        """使用關鍵詞匹配找出相關社群(改進版)"""
        
        # 從問題中提取多個可能的關鍵詞
        words = query.replace('?', '').replace('?', '').split()
        keywords = [w for w in words if len(w) >= 2][:5]  # 取最多5個關鍵詞
        
        if not keywords:
            keywords = [query[:10]]  # 如果沒有關鍵詞就用前10個字
        
        print(f"🔑 使用關鍵詞: {keywords}")
        
        query_cypher = """
        MATCH (c:Community)
        WHERE c.summary IS NOT NULL
        WITH c, $keywords as keywords
        WITH c,
            SIZE([kw IN keywords WHERE c.summary CONTAINS kw]) as match_count
        WHERE match_count > 0
        RETURN 
            c.id as community_id,
            c.summary as summary,
            c.entity_count as entity_count,
            match_count as relevance
        ORDER BY relevance DESC, c.entity_count DESC
        LIMIT $limit
        """
        
        try:
            with self.triplet_driver.session(database="nuu-triplet") as session:
                result = session.run(query_cypher, keywords=keywords, limit=limit)
                communities = [dict(record) for record in result]
                
                if communities:
                    print(f"✅ 關鍵詞匹配找到 {len(communities)} 個社群")
                
                return communities
                
        except Exception as e:
            print(f"❌ 關鍵詞匹配失敗: {e}")
            return []

    def _get_communities_for_entities(self, entities: List[str], limit: int = 3) -> List[dict]:
        """根據實體獲取相關社群摘要（用於文檔增強）"""
        if not entities:
            return []
        
        query = """
        UNWIND $entities AS entity_name
        MATCH (e:Entity)
        WHERE e.name CONTAINS entity_name OR e.name = entity_name
        WITH DISTINCT e.communityId as commId
        MATCH (c:Community {id: commId})
        WHERE c.summary IS NOT NULL
        RETURN DISTINCT
            c.id as community_id,
            c.summary as summary,
            c.entity_count as entity_count
        ORDER BY c.entity_count DESC
        LIMIT $limit
        """
        
        try:
            with self.triplet_driver.session(database="nuu-triplet") as session:
                result = session.run(query, entities=entities, limit=limit)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"❌ 獲取社群摘要失敗: {e}")
            return []
            
    # 步驟1: 從Weaviate ContentSummaryEn0913搜尋，步驟2: 用neo4j_id去Neo4j查詢完整資訊
    def search(self, query: str, limit: int = 10) -> List[Document]:
        if not self.client or not self.embedder or not self.collection or not self.neo4j_driver:
            return [Document(page_content="❌ 系統連接失敗", metadata={})]
        
        try:
            # 步驟 1: Weaviate 向量搜尋
            print(f"🔍 步驟1: 在 Weaviate ContentSummaryEn0913 中進行向量搜尋...")
            query_vector = self.embedder.embed_query(query)
            weaviate_results = self._weaviate_vector_search(query_vector, limit)
            
            if not weaviate_results:
                print("❌ Weaviate 搜尋無結果")
                return [Document(page_content="搜尋無結果", metadata={})]
            
            # 步驟 2: 提取 neo4j_id 並查詢 Neo4j
            neo4j_ids = []
            weaviate_data = {}
            
            for result in weaviate_results:
                neo4j_id = result['neo4j_id']
                if neo4j_id:
                    neo4j_ids.append(neo4j_id)
                    weaviate_data[neo4j_id] = result
            
            if not neo4j_ids:
                print("⚠️ 沒有找到有效的 neo4j_id")
                return [Document(page_content="沒有找到對應的Neo4j資料", metadata={})]
            
            print(f"🔍 步驟2: 使用 {len(neo4j_ids)} 個 neo4j_id 查詢 Neo4j...")
            enhanced_documents = self._query_neo4j_by_ids(neo4j_ids, weaviate_data)
            
            print(f"✅ 總共生成 {len(enhanced_documents)} 筆增強文檔")
            return enhanced_documents
            
        except Exception as e:
            print(f"❌ 搜尋過程發生錯誤: {e}")
            traceback.print_exc()
            return [Document(page_content="搜尋失敗", metadata={})]
        
    # 從 Weaviate ContentSummaryEn0913 執行向量搜尋，返回相關度最高的結果
    def _weaviate_vector_search(self, query_vector: List[float], limit: int) -> List[dict]:
        results = []
        
        try:
            response = self.collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=wvc.query.MetadataQuery(distance=True)
            )
            
            for obj in response.objects:
                # 獲取 english_summary 內容
                content = obj.properties.get(self.text_field, '')
                if not content or content.strip() == "":
                    continue
                
                # 計算相似度分數
                distance = float(obj.metadata.distance) if obj.metadata.distance is not None else 1.0
                similarity = max(0, 1.0 - distance)
                
                # 收集結果
                result = {
                    'weaviate_uuid': str(obj.uuid),
                    'english_summary': content,
                    'neo4j_id': obj.properties.get('neo4j_id', ''),
                    'article_url': obj.properties.get('article_url', ''),
                    'language': obj.properties.get('language', 'en'),
                    'similarity': similarity,
                    'distance': distance
                }
                results.append(result)
            
            print(f"✅ Weaviate 搜尋成功，找到 {len(results)} 筆結果")
            
            # 顯示前3筆結果的預覽
            for i, result in enumerate(results[:3], 1):
                preview = result['english_summary'][:100] + "..." if len(result['english_summary']) > 100 else result['english_summary']
                print(f"  [{i}] 相似度: {result['similarity']:.3f} | Neo4j ID: {result['neo4j_id']} | {preview}")
            
            return results
            
        except Exception as e:
            print(f"❌ Weaviate 向量搜尋失敗: {e}")
            return []
    
    # 優化版本：分階段查詢，避免複雜的 collect 操作
    def _query_neo4j_by_ids(self, neo4j_ids: List[str], weaviate_data: dict) -> List[Document]:
        if not self.neo4j_driver or not neo4j_ids:
            return []
        
        enhanced_documents = []
        

        for neo4j_id in neo4j_ids:
            print(f"🔍 查詢 Neo4j ID: {neo4j_id}")
            start_time = time.time()
            
            try:
                # 分階段查詢策略
                content_data = self._get_content_info(neo4j_id)
                if not content_data:
                    print(f"⚠️ Content 資料不存在: {neo4j_id}")
                    continue
                
                article_data = self._get_article_info(neo4j_id)
                org_data = self._get_organization_info(neo4j_id)
                
                # 【修改】加入三元組圖譜增強
                triplet_data = self._enhance_with_triplet_graph(
                    neo4j_id, 
                    content_data.get('content_text', ''), 
                    article_data.get('article_title', '')
                )

                # 合併資料
                combined_data = {**content_data, **article_data, **org_data, **triplet_data}
                
                # 建立增強文檔
                weaviate_info = weaviate_data.get(neo4j_id, {})
                enhanced_doc = self._create_enhanced_document_optimized(combined_data, weaviate_info)
                enhanced_documents.append(enhanced_doc)
                
                elapsed = time.time() - start_time
                print(f"✅ 查詢完成 ({elapsed:.2f}s)")
                
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ 查詢失敗 ({elapsed:.2f}s): {e}")
                
                # 使用 Weaviate 備案
                weaviate_info = weaviate_data.get(neo4j_id, {})
                if weaviate_info:
                    fallback_doc = Document(
                        page_content=weaviate_info.get('english_summary', ''),
                        metadata={
                            'source': 'Weaviate_fallback',
                            'neo4j_id': neo4j_id,
                            'similarity': weaviate_info.get('similarity', 0),
                            'enhanced': False
                        }
                    )
                    enhanced_documents.append(fallback_doc)
        
        return enhanced_documents
    
    # 獲取 Content 基本資訊
    def _get_content_info(self, neo4j_id: str) -> dict:
        query = """
        MATCH (content:Content {neo4j_id: $neo4j_id})
        RETURN 
            content.neo4j_id as content_neo4j_id,
            content.id as content_id,
            content.text as content_text,
            content.type as content_type,
            content.order as content_order
        """
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query, neo4j_id=neo4j_id)
                record = result.single()
                return dict(record) if record else {}
        except Exception as e:
            print(f"❌ Content 查詢失敗: {e}")
            return {}
        
    # 獲取 Article 資訊
    def _get_article_info(self, neo4j_id: str) -> dict:
        query = """
        MATCH (content:Content {neo4j_id: $neo4j_id})
        OPTIONAL MATCH (article:Article)-[:HAS_CONTENT]->(content)
        RETURN 
            article.neo4j_id as article_neo4j_id,
            article.url as article_url,
            article.title as article_title,
            article.domain as article_domain,
            article.og_image as article_og_image
        """
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query, neo4j_id=neo4j_id)
                record = result.single()
                return dict(record) if record else {}
        except Exception as e:
            print(f"❌ Article 查詢失敗: {e}")
            return {}
        
    # 獲取組織資訊 - 簡化版本
    def _get_organization_info(self, neo4j_id: str) -> dict:
        query = """
        MATCH (content:Content {neo4j_id: $neo4j_id})
        OPTIONAL MATCH (article:Article)-[:HAS_CONTENT]->(content)
        OPTIONAL MATCH (article)-[:BELONGS_TO]->(org:Organization)
        RETURN 
            org.name as org_name,
            org.type as org_type,
            org.unified_number as org_unified_number
        LIMIT 1
        """
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query, neo4j_id=neo4j_id)
                record = result.single()
                return dict(record) if record else {}
        except Exception as e:
            print(f"❌ Organization 查詢失敗: {e}")
            return {}
        
    # 【新增】通過三元組圖譜增強資料
    def _enhance_with_triplet_graph(self, neo4j_id: str, content_text: str, article_title: str) -> dict:
        """通過三元組圖譜增強資料，只取10筆測試"""
        if not self.triplet_driver:
            return {'triplet_relations': [], 'entity_count': 0}
        
        try:
            # 使用 Mistral 提取實體
            text_for_extraction = ""
            if article_title:
                text_for_extraction += article_title + " "
            if content_text:
                text_for_extraction += content_text[:200]  # 只取前200字避免太長
            
            if not text_for_extraction.strip():
                return {'triplet_relations': [], 'entity_count': 0}
            
            # 提取實體
            entities = extract_entities_with_mistral(text_for_extraction)
            
            if not entities:
                return {'triplet_relations': [], 'entity_count': 0}
            
            # 在三元組圖譜中查詢關係（限制10筆）
            triplet_relations = self._query_triplet_relationships_limited(entities)
            
            
             # 【新增】查詢相關社群摘要
            community_summaries = self._get_communities_for_entities(entities)
            
            print(f"🔗 三元組增強完成: {len(entities)} 個實體，{len(triplet_relations)} 個關係")
            
            return {
                'triplet_relations': triplet_relations,
                'entity_count': len(entities),
                'extracted_entities': entities,
                'community_summaries': community_summaries  
            }
            
        except Exception as e:
            print(f"❌ 三元組圖譜增強失敗: {e}")
            return {'triplet_relations': [], 'entity_count': 0}

# 【新增】限制查詢三元組關係（只取10筆）
    def _query_triplet_relationships_limited(self, entities: List[str]) -> List[dict]:
        """在三元組圖譜中查詢實體關係，限制10筆"""
        if not entities:
            return []
        
        query = """
        UNWIND $entities AS entity_name
        MATCH (s)-[r]->(o)
        WHERE s.name CONTAINS entity_name 
        OR o.name CONTAINS entity_name
        OR s.name = entity_name 
        OR o.name = entity_name
        WITH s, r, o, entity_name
        WHERE type(r) IN ['ORGANIZATIONAL', 'SPATIAL', 'CONTACT', 'ACADEMIC', 
                        'FACILITY', 'SERVICE', 'PERSONNEL', 'TEMPORAL',
                        'COLLABORATION', 'INFORMATION', 'ADMINISTRATIVE', 'ATTRIBUTE']
        RETURN 
            s.name as subject,
            type(r) as relation_type,
            r.original_relation as original_relation,
            o.name as object,
            entity_name as matched_entity
        LIMIT 10
        """
        
        try:
            with self.triplet_driver.session() as session:
                result = session.run(query, entities=entities)
                relations = []
                
                for record in result:
                    relations.append({
                        'subject': record['subject'],
                        'relation_type': record['relation_type'], 
                        'original_relation': record['original_relation'],
                        'object': record['object'],
                        'matched_entity': record['matched_entity']
                    })
                
                return relations
                
        except Exception as e:
            print(f"❌ 查詢三元組關係失敗: {e}")
            return []
        
    # 建立優化版增強文檔
    def _create_enhanced_document_optimized(self, neo4j_data: dict, weaviate_info: dict) -> Document:
        try:
            content_parts = []
            
            # Weaviate 英文摘要
            base_content = weaviate_info.get('english_summary', '')
            if base_content:
                content_parts.append(base_content)
            
            # Neo4j Content 原始文本
            content_text = neo4j_data.get('content_text')
            if content_text and content_text != base_content:
                content_parts.append(f"【原始內容】\n{content_text}")
            
            # Article 資訊
            article_title = neo4j_data.get('article_title')
            if article_title:
                content_parts.append(f"【文章標題】\n{article_title}")
                
            article_url = neo4j_data.get('article_url')
            if article_url:
                content_parts.append(f"【文章網址】\n{article_url}")
            
            # 組織資訊
            org_name = neo4j_data.get('org_name')
            if org_name:
                org_info = org_name
                org_type = neo4j_data.get('org_type')
                if org_type:
                    org_info += f" ({org_type})"
                content_parts.append(f"【所屬組織】\n{org_info}")
            
              # 【新增】三元組圖譜關係資訊
            triplet_relations = neo4j_data.get('triplet_relations', [])
            if triplet_relations:
                relation_text = "【知識圖譜關係】\n"
                for i, rel in enumerate(triplet_relations, 1):
                    subject = rel.get('subject', '')
                    relation_type = rel.get('relation_type', '')
                    original_relation = rel.get('original_relation', '')
                    obj = rel.get('object', '')
                    
                    relation_line = f"{i}. {subject} --[{relation_type}]--> {obj}"
                    if original_relation and original_relation != relation_type:
                        relation_line += f" (原始關係: {original_relation})"
                    
                    relation_text += relation_line + "\n"
                
                content_parts.append(relation_text)
            # 【新增】社群摘要資訊
            community_summaries = neo4j_data.get('community_summaries', [])
            if community_summaries:
                community_text = "【相關領域背景】\n"
                for i, comm in enumerate(community_summaries, 1):
                    summary = comm.get('summary', '')
                    entity_count = comm.get('entity_count', 0)
                    community_text += f"{i}. {summary} (涵蓋 {entity_count} 個相關實體)\n\n"
                content_parts.append(community_text)
            
            # 組合內容
            enhanced_content = "\n\n".join(content_parts) if content_parts else "無內容"
            
            # 建立 metadata
            metadata = {
                'source': 'Weaviate+Neo4j+GraphRAG',
                'neo4j_id': neo4j_data.get('content_neo4j_id', ''),
                'similarity': weaviate_info.get('similarity', 0),
                'enhanced': True,
                'graphrag_enhanced': len(triplet_relations) > 0,  # 【新增】
                'relation_count': len(triplet_relations),  # 【新增】
                'entity_count': neo4j_data.get('entity_count', 0),  # 【新增】
                 'community_count': len(community_summaries),
                'article_title': article_title or '',
                'article_url': neo4j_data.get('article_url', ''),
                'content_type': neo4j_data.get('content_type', ''),
                'organization': org_name or ''
            }
            
            return Document(page_content=enhanced_content, metadata=metadata)
            
        except Exception as e:
            print(f"❌ 文檔建立失敗: {e}")
            # 返回基礎 Weaviate 文檔
            return Document(
                page_content=weaviate_info.get('english_summary', 'Error'),
                metadata={
                    'source': 'Error_fallback',
                    'neo4j_id': weaviate_info.get('neo4j_id', ''),
                    'enhanced': False,
                    'graphrag_enhanced': False
                }
            )
    
    # 根據 Neo4j 查詢結果和 Weaviate 資訊建立增強文檔（保留原始版本作為備用）
    def _create_enhanced_document(self, neo4j_record, weaviate_info: dict) -> Document:
        try:
            # 基礎內容從 Weaviate 的 english_summary 開始
            base_content = weaviate_info.get('english_summary', '')
            content_parts = [base_content] if base_content else []
            
            # 添加 Neo4j Content 的原始文本
            content_text = neo4j_record.get('content_text')
            if content_text and content_text != base_content:
                content_parts.append(f"【原始內容】\n{content_text}")
            
            # 添加 Article 資訊
            article_title = neo4j_record.get('article_title')
            if article_title:
                content_parts.append(f"【文章標題】\n{article_title}")
            
            article_domain = neo4j_record.get('article_domain')
            if article_domain:
                content_parts.append(f"【網站來源】\n{article_domain}")
            
            # 添加文章摘要
            article_summary = neo4j_record.get('article_summary')
            if article_summary:
                content_parts.append(f"【文章摘要】\n{article_summary}")
            
            # 添加圖片資訊
            article_og_image = neo4j_record.get('article_og_image')
            if article_og_image:
                content_parts.append(f"【文章圖片】\n{article_og_image}")
            
            # 添加組織資訊
            organizations = neo4j_record.get('organizations', [])
            valid_organizations = [o for o in organizations if o.get('name')]
            if valid_organizations:
                org_lines = []
                for org in valid_organizations:
                    org_name = org.get('name', '')
                    org_type = org.get('type', '')
                    unified_number = org.get('unified_number', '')
                    
                    org_line = org_name
                    if org_type:
                        org_line += f" ({org_type})"
                    if unified_number:
                        org_line += f" [統編: {unified_number}]"
                    org_lines.append(org_line)
                
                if org_lines:
                    content_parts.append(f"【所屬組織】\n" + "\n".join(org_lines))
            
            # 添加部門資訊
            departments = neo4j_record.get('departments', [])
            valid_departments = [d for d in departments if d.get('name')]
            if valid_departments:
                dept_lines = []
                for dept in valid_departments:
                    dept_name = dept.get('name', '')
                    dept_type = dept.get('type', '')
                    dept_description = dept.get('description', '')
                    
                    dept_line = dept_name
                    if dept_type:
                        dept_line += f" ({dept_type})"
                    if dept_description:
                        dept_line += f" - {dept_description}"
                    dept_lines.append(dept_line)
                
                if dept_lines:
                    content_parts.append(f"【所屬部門】\n" + "\n".join(dept_lines))
            
            # 添加聯絡資訊
            contacts = neo4j_record.get('contacts', [])
            valid_contacts = [c for c in contacts if c.get('value')]
            if valid_contacts:
                contact_lines = []
                for contact in valid_contacts:
                    contact_type = contact.get('type', '聯絡方式')
                    contact_value = contact.get('value', '')
                    contact_description = contact.get('description', '')
                    
                    contact_line = f"{contact_type}: {contact_value}"
                    if contact_description:
                        contact_line += f" ({contact_description})"
                    contact_lines.append(contact_line)
                
                if contact_lines:
                    content_parts.append(f"【聯絡資訊】\n" + "\n".join(contact_lines))
            
            # 添加地址資訊
            addresses = neo4j_record.get('addresses', [])
            valid_addresses = [a for a in addresses if a.get('full_address')]
            if valid_addresses:
                address_lines = []
                for addr in valid_addresses:
                    full_addr = addr.get('full_address', '')
                    campus_name = addr.get('campus_name', '')
                    city = addr.get('city', '')
                    district = addr.get('district', '')
                    postal_code = addr.get('postal_code', '')
                    
                    addr_line = full_addr
                    
                    # 添加校區名稱
                    if campus_name:
                        addr_line = f"{campus_name} - {addr_line}"
                    
                    # 添加城市區域資訊
                    location_info = []
                    if city and district:
                        location_info.append(f"{city}{district}")
                    elif city:
                        location_info.append(city)
                    
                    if postal_code:
                        location_info.append(f"郵遞區號: {postal_code}")
                    
                    if location_info:
                        addr_line += f" ({', '.join(location_info)})"
                    
                    address_lines.append(addr_line)
                
                if address_lines:
                    content_parts.append(f"【地址資訊】\n" + "\n".join(address_lines))
            
            # 添加相關內容
            related_contents = neo4j_record.get('related_contents', [])
            valid_related = [r for r in related_contents if r.get('text')]
            if valid_related:
                related_lines = []
                for related in valid_related[:3]:  # 只顯示前3個相關內容
                    text = related.get('text', '')
                    content_type = related.get('type', '未知類型')
                    order = related.get('order', '')
                    
                    # 限制相關內容長度
                    preview = text[:150] + "..." if len(text) > 150 else text
                    related_line = f"[{content_type}]"
                    if order:
                        related_line += f" (順序: {order})"
                    related_line += f" {preview}"
                    
                    related_lines.append(related_line)
                
                if related_lines:
                    content_parts.append(f"【相關內容】\n" + "\n".join(related_lines))
            
            # 組合完整內容
            enhanced_content = "\n\n".join(content_parts)
            
            # 建立完整的 metadata
            metadata = {
                'source': 'Weaviate+Neo4j',
                'weaviate_uuid': weaviate_info.get('weaviate_uuid', ''),
                'neo4j_id': neo4j_record.get('content_neo4j_id', ''),
                'similarity': weaviate_info.get('similarity', 0),
                'distance': weaviate_info.get('distance', 1.0),
                'enhanced': True,
                
                # Content 資訊
                'content_id': neo4j_record.get('content_id', ''),
                'content_type': neo4j_record.get('content_type', ''),
                'content_order': neo4j_record.get('content_order', ''),
                
                # Article 資訊
                'article_title': article_title or '',
                'article_url': neo4j_record.get('article_url', ''),
                'article_domain': article_domain or '',
                'article_summary': article_summary or '',
                'article_og_image': article_og_image or '',
                'article_updated_at': neo4j_record.get('article_updated_at', ''),
                
                # 組織資訊
                'organizations': [org.get('name', '') for org in valid_organizations],
                'organization_types': [org.get('type', '') for org in valid_organizations],
                'unified_numbers': [org.get('unified_number', '') for org in valid_organizations],
                
                # 部門資訊
                'departments': [dept.get('name', '') for dept in valid_departments],
                'department_types': [dept.get('type', '') for dept in valid_departments],
                'department_descriptions': [dept.get('description', '') for dept in valid_departments],
                
                # 聯絡資訊
                'contact_types': [c.get('type', '') for c in valid_contacts],
                'contact_values': [c.get('value', '') for c in valid_contacts],
                
                # 地址資訊
                'campus_names': [a.get('campus_name', '') for a in valid_addresses],
                'cities': [a.get('city', '') for a in valid_addresses],
                'districts': [a.get('district', '') for a in valid_addresses],
                'postal_codes': [a.get('postal_code', '') for a in valid_addresses],
                
                # 統計資訊
                'organization_count': len(valid_organizations),
                'contact_count': len(valid_contacts),
                'address_count': len(valid_addresses),
                'department_count': len(valid_departments),
                'related_content_count': len(valid_related)
            }
            
            return Document(page_content=enhanced_content, metadata=metadata)
            
        except Exception as e:
            print(f"❌ 建立增強文檔失敗: {e}")
            # 返回基礎文檔
            return Document(
                page_content=weaviate_info.get('english_summary', '建立文檔失敗'),
                metadata={
                    'source': 'Fallback',
                    'neo4j_id': weaviate_info.get('neo4j_id', ''),
                    'similarity': weaviate_info.get('similarity', 0),
                    'enhanced': False
                }
            )

# RAG 系統狀態定義
class State(TypedDict):
    question: str
    original_question: str
    translated_question: str
    is_chinese_query: bool
    strategy: str  # 新增：記錄搜尋策略
    context: List[Document]
    answer: str
    related_links: List[Dict] 

# 初始化 ContentSummaryEn0913 RAG
content_summary_rag = ContentSummaryEn0913RAG()

# 處理問題 - 檢查是否為中文並進行翻
def process_question(state: State):
    question = state["question"]
    original_question = question
    
    # 檢查是否包含中文
    is_chinese_query = is_chinese(question)
    
    if is_chinese_query:
        print(f"🈶 檢測到中文查詢，準備翻譯...")
        try:
            # 使用 Mistral 翻譯中文為英文
            translated_question = translate_with_mistral(question)
            print(f"🔄 翻譯結果: {translated_question}")
        except Exception as e:
            print(f"⚠️ 翻譯失敗，使用原始問題: {e}")
            translated_question = question
            is_chinese_query = False
    else:
        print(f"🔤 檢測到英文查詢，直接使用")
        translated_question = question
    
    return {
        "question": translated_question,  # 用於檢索的問題（英文）
        "original_question": original_question,  # 原始問題
        "translated_question": translated_question,  # 翻譯後的問題
        "is_chinese_query": is_chinese_query  # 是否為中文查詢
    }

# 檢索相關文檔 - Weaviate 向量搜尋 + Neo4j 資料增強
def retrieve(state: State):
    question = state["question"]
    
    # 使用 Mistral 判斷策略
    print(f"\n🤔 使用 Mistral 判斷搜尋策略...")
    strategy = choose_search_strategy_with_mistral(state["original_question"])
    print(f"🎯 Mistral 判斷結果: {strategy.upper()}")
    
    if strategy == "global":
        # === Global Search: 使用智能層級選擇 ===
        print("🌍 執行 Global Search（智能層級選擇）")
        
        # 自動選擇最佳層級並檢索
        communities = content_summary_rag.find_relevant_communities(
            query=question,
            limit=None,   # 自動決定
            level=None    # 自動選擇
        )
        
        if not communities:
            print("⚠️ Global Search 無結果，自動切換到 Local Search")
            strategy = "local"
        else:
            print(f"✅ 找到 {len(communities)} 個相關社群")
            community_docs = []
            for comm in communities:
                doc = Document(
                    page_content=comm['summary'],
                    metadata={
                        'source': 'Community',
                        'strategy': 'global',
                        'community_id': comm['community_id'],
                        'level': comm['level'],
                        'entity_count': comm['entity_count'],
                        'core_relevance': comm.get('core_relevance', 0),
                        'aux_relevance': comm.get('aux_relevance', 0)
                    }
                )
                community_docs.append(doc)
            
            return {"context": community_docs, "strategy": strategy}
    
    # === Local Search: 使用向量檢索 + 圖譜增強 ===
    print("📍 執行 Local Search (向量檢索 + 圖譜增強)")
    documents = content_summary_rag.search(question, limit=10)
    
    # 標記策略
    for doc in documents:
        doc.metadata['strategy'] = 'local'
    
    print(f"\n📊 檢索結果統計:")
    print(f"🔍 總共檢索到 {len(documents)} 筆相關資料")
    
    enhanced_count = sum(1 for doc in documents if doc.metadata.get('enhanced', False))
    print(f"✨ 其中 {enhanced_count} 筆已通過 Neo4j 增強")
    
    return {"context": documents, "strategy": strategy}

import time

def generate_global(state: State):
    """改進版 Global Search - 包含社群摘要的展示與連結收集"""
    communities = state["context"]
    
    # === 第一階段：展示社群摘要 ===
    print(f"\n📚 第一階段：展示 {len(communities)} 個相關社群摘要...")
    
    summaries_text = ""
    for i, comm_doc in enumerate(communities, 1):
        comm_id = comm_doc.metadata.get('community_id')
        entity_count = comm_doc.metadata.get('entity_count', 0)
        core_relevance = comm_doc.metadata.get('core_relevance', 0)
        
        summaries_text += f"\n【社群 {i} - ID: {comm_id}】(涵蓋 {entity_count} 個實體，核心匹配度: {core_relevance})\n"
        summaries_text += f"{comm_doc.page_content}\n"
        summaries_text += "-" * 40 + "\n"
    
    print(f"✅ 社群摘要收集完成")
    
    # === 第二階段：Map - 每個社群獨立生成部分答案 ===
    print(f"\n🗺️  Map 階段：為 {len(communities)} 個社群生成部分答案...")
    partial_answers = []
    
    for i, comm_doc in enumerate(communities, 1):
        comm_id = comm_doc.metadata.get('community_id')
        print(f"   處理社群 {i}/{len(communities)} (ID: {comm_id})")
        
        if state["is_chinese_query"]:
            map_prompt = f"""請根據以下社群摘要，針對用戶問題提供相關資訊。

社群摘要：
{comm_doc.page_content}

用戶問題：{state["original_question"]}

請只使用這個社群的資訊回答。如果這個社群與問題無關，請回答「此社群無相關資訊」。
請使用條列式或段落式清晰組織答案。

部分答案："""
        else:
            map_prompt = f"""Please provide relevant information based on the following community summary to answer the user's question.

Community Summary:
{comm_doc.page_content}

User Question: {state["original_question"]}

Only use information from this community. If not relevant, answer "No relevant information."
Please organize answer clearly with bullet points or paragraphs.

Partial Answer:"""
        
        try:
            response = call_mistral(map_prompt)
            partial_answer = response["choices"][0]["message"]["content"].strip()
            
            partial_answers.append({
                'community_id': comm_id,
                'entity_count': comm_doc.metadata.get('entity_count'),
                'core_relevance': comm_doc.metadata.get('core_relevance', 0),
                'answer': partial_answer
            })
            print(f"      ✅ 完成")
            
            if i < len(communities):
                time.sleep(0.5)
                
        except Exception as e:
            print(f"      ❌ 失敗: {e}")
            continue
    
    if not partial_answers:
        return {
            "answer": "抱歉，無法從社群資料中生成答案。請嘗試更具體的問題。",
            "related_links": []
        }
    
    # === 第三階段：Reduce - 合併答案 ===
    print(f"\n🔀 Reduce 階段：綜合 {len(partial_answers)} 個社群的答案...")
    
    combined_text = "各社群提供的資訊：\n\n"
    for i, pa in enumerate(partial_answers, 1):
        combined_text += f"【社群 {pa['community_id']} (涵蓋 {pa['entity_count']} 個實體)】\n"
        combined_text += f"{pa['answer']}\n\n"
    
    if state["is_chinese_query"]:
        reduce_prompt = f"""你是國立聯合大學的智能助手。以下是從不同知識社群獲得的資訊，請綜合這些資訊提供完整答案。

用戶問題：{state["original_question"]}

原始社群摘要：
{summaries_text}

各社群提供的部分答案：
{combined_text}

請根據以上資訊提供一個完整、準確的答案：
1. 整合所有相關資訊
2. 去除重複內容
3. 用清晰的結構組織答案
4. 忽略標註「無相關資訊」的部分
5. 如果所有社群都無相關資訊，請誠實告知
6. 在最後加上「資訊來源」段落，簡述涉及的相關社群
7. 直接回答用戶問題
8. 如果問題與資料無關 ，請不要亂回答，誠實告知

最終答案："""
    else:
        reduce_prompt = f"""You are an intelligent assistant for National United University. Below are community summaries and partial answers.

User Question: {state["original_question"]}

Original Community Summaries:
{summaries_text}

Partial Answers from Communities:
{combined_text}

Please provide a complete answer:
1. Integrate all relevant information
2. Remove duplicate content
3. Organize with clear structure
4. Ignore "No relevant information" parts
5. If all communities lack info, be honest
6. Add "Information Sources" section at the end mentioning relevant communities
7. Answer the user's question directly

Final Answer:"""
    
    try:
        response = call_mistral(reduce_prompt)
        final_answer = response["choices"][0]["message"]["content"].strip()
        
        if final_answer.startswith('"') and final_answer.endswith('"'):
            final_answer = final_answer[1:-1]
        
        # ===== 新增：Global Search 也收集連結（從社群摘要中） =====
        related_links = []
        seen_urls = set()
        
        for comm_doc in communities:
            # 如果社群摘要中包含 URL（視資料結構而定）
            article_url = comm_doc.metadata.get('article_url', '')
            article_title = comm_doc.metadata.get('article_title', '')
            
            if article_url and article_url not in seen_urls:
                seen_urls.add(article_url)
                related_links.append({
                    'url': article_url,
                    'title': article_title or f"社群 {comm_doc.metadata.get('community_id')}",
                    'source': 'Community',
                    'community_id': comm_doc.metadata.get('community_id'),
                    'entity_count': comm_doc.metadata.get('entity_count', 0)
                })
        
        related_links = related_links[:10]
        # ==========================================================
        
        return {
            "answer": final_answer,
            "related_links": related_links
        }
        
    except Exception as e:
        print(f"❌ Reduce 階段失敗: {e}")
        fallback_answer = "根據多個相關社群的資訊：\n\n" + combined_text
        return {
            "answer": fallback_answer,
            "related_links": []
        }


def generate_local(state: State):
    """Local Search 的直接生成（使用原本的邏輯）"""  
    # ===== 新增：收集 related_links =====
    related_links = []
    seen_urls = set()
    
    for doc in state["context"]:
        article_url = doc.metadata.get('article_url', '')
        article_title = doc.metadata.get('article_title', '')
        
        if article_url and article_url not in seen_urls:
            seen_urls.add(article_url)
            related_links.append({
                'url': article_url,
                'title': article_title or '相關文章',
                'source': doc.metadata.get('source', 'GraphRAG'),
                'similarity': doc.metadata.get('similarity', 0),
                'enhanced': doc.metadata.get('enhanced', False),
                'graphrag_enhanced': doc.metadata.get('graphrag_enhanced', False)
            })
    
    related_links = related_links[:10]
    # =====================================
    # 整理檢索內容
    context_text = ""
    for i, doc in enumerate(state["context"], 1):
        similarity = doc.metadata.get('similarity', 0)
        enhanced = doc.metadata.get('enhanced', False)
        content = doc.page_content
        article_url = doc.metadata.get('article_url', '')
        article_title = doc.metadata.get('article_title', '')
        
        context_text += f"[資料 {i}] 來源: {'Weaviate+Neo4j增強' if enhanced else 'Weaviate基礎'} (相似度: {similarity:.3f})\n"
        if article_title:
            context_text += f"文章標題: {article_title}\n"
        if article_url:
            context_text += f"文章網址: {article_url}\n"
        context_text += f"{content}\n\n"
    
    # 根據是否為中文查詢選擇不同的提示模板
    if state["is_chinese_query"]:
        template = """你是國立聯合大學的智能助手，請根據以下檢索資料回答用戶問題。

        ## 檢索與增強資料：
        {context}

        ## 用戶問題：
        {original_question}

        ## 回答指引：
        1. 請用中文回答
        2. 優先使用增強後的完整資料
        3. 如果有聯絡資訊、地址等，請明確列出
        4. 條理清晰，重要資訊用項目符號
        5. 如有相關網址請提供
        6. 不要在回答中加入資料編號引用
        7. 直接使用資料內容，不需標註來源編號
        8. 如果資料跟問題無關，請不要亂編答案，誠實告知

        回答："""
    else:
         template = """You are an intelligent assistant for National United University. Please answer based on the retrieved data.

        ## Retrieved and Enhanced Data:
        {context}

        ## User Question:
        {original_question}

        ## Guidelines:
        1. Answer in English
        2. Prioritize enhanced comprehensive data
        3. Provide specific contact information and addresses if available
        4. Well-structured with bullet points for important information
        5. Include relevant URLs if available
        6. Do not include data source numbers
        7. Use data content directly without source citations

        Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    messages = prompt.invoke({
        "context": context_text,
        "original_question": state["original_question"]
    })
    
    response = llm.invoke(messages)
    return {"answer": response.content}

# 生成回答
def generate(state: State):
    """根據策略選擇對應的生成方法"""
    strategy = state.get("strategy", "local")
    print(f"\n💡 使用 {strategy.upper()} 策略生成答案...")
    if strategy == "global":
        return generate_global(state)
    else:
        return generate_local(state)
    
# 建立 RAG 流程
graph_builder = StateGraph(State).add_sequence([process_question, retrieve, generate])
graph_builder.add_edge(START, "process_question")
graph = graph_builder.compile()

# RAG 問答主函數
def ask_question(question: str) -> tuple:  # 修改回傳類型
    """
    回傳：(answer: str, related_links: List[Dict])
    """
    start_time = time.time()
    result = graph.invoke({"question": question})
    end_time = time.time()
    
    print(f"⏱️ 總處理時間：{end_time - start_time:.2f} 秒")
    
    # 回傳答案和連結
    return result["answer"], result.get("related_links", [])

# 主程式執行
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🎓 國立聯合大學 ContentSummaryEn0913 + Neo4j 圖譜增強 RAG 問答系統")
    print(f"{'='*60}")
    print(f"📊 系統狀態: {'✅ 正常運行' if content_summary_rag.client and content_summary_rag.neo4j_driver else '❌ 連接失敗'}")
    print(f"🔧 架構: 問題處理(翻譯) → 向量檢索 → Neo4j圖譜增強 → 生成")
    print(f"📚 資料來源: ContentSummaryEn0913 集合 (英文摘要) + Neo4j 圖譜資料")
    print(f"🧠 嵌入模型: Gemini embedding-001")
    print(f"🔍 搜尋方法: 向量相似度搜尋 + 圖譜關係增強")
    print(f"🌐 翻譯功能: 中文問題自動翻譯為英文進行檢索")
    print(f"💬 回答語言: 根據問題語言自動判斷（中文問題用中文回答，英文問題用英文回答）")
    print(f"{'='*60}\n")
    

    
    # 檢查 Mistral API Key
    if not MISTRAL_API_KEY:
        print("⚠️  警告: 未設定 MISTRAL_API_KEY，中文翻譯功能將無法使用")
    else:
        print("✅ Mistral API 已設定，支援中文翻譯功能")
    
    # 檢查翻譯映射檔案
    try:
        mapping_count = len(translation_manager.translation_dict)
        print(f"✅ 翻譯映射檔案已載入，包含 {mapping_count} 條翻譯對應")
    except Exception as e:
        print(f"⚠️  警告: 翻譯映射檔案載入失敗: {e}")
    
    # 檢查系統連接狀態
    weaviate_status = "✅ 正常" if content_summary_rag.client else "❌ 失敗"
    neo4j_status = "✅ 正常" if content_summary_rag.neo4j_driver else "❌ 失敗"
    print(f"🔗 Weaviate 連接: {weaviate_status}")
    print(f"🔗 Neo4j 連接: {neo4j_status}")
    
    # 互動式問答
    try:
        while True:
            print("-" * 50)
            question = input("請輸入您的問題 (中文或英文，輸入 'exit' 退出): ")
            
            if question.lower() in ['exit', 'quit', '退出', '離開']:
                print("謝謝使用，再見！")
                break
            
            if not question.strip():
                continue
                
            print(f"\n🤖 正在處理...")
            try:
                answer = ask_question(question)
                print(f"\n💡 回答:\n{answer}\n")
                        
            except Exception as e:
                print(f"處理問題時發生錯誤: {e}")
                traceback.print_exc()
                
    except KeyboardInterrupt:
        print(f"\n\n程式中斷，謝謝使用！")