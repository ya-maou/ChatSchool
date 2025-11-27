#新data0804 - ContentSummaryEn0913 集合 (英文摘要) + Neo4j 關鍵字搜尋 新data0804
#還沒更新 新的文字轉向量
from typing import List, TypedDict, Dict, Optional
from langchain_core.documents.base import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
import weaviate
from weaviate import connect_to_local
import weaviate.classes as wvc
import google.generativeai as genai
from neo4j import GraphDatabase
from pydantic import BaseModel
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

# 設置 LM Studio 資訊
LMSTUDIO_MODEL = "google/gemma-3-4B" 
LMSTUDIO_URL = "http://192.168.98.34:1234/v1" 

genai.configure(api_key=GOOGLE_API_KEY)

# Setup LLM
from langchain.chat_models import init_chat_model

def init_LMStudio(model: str, base_url: str, api_key: str = ".", configurable_fields: None = None, config_prefix: str | None = None, **kwargs) -> BaseChatModel:
    """使用 LangChain 連接至 LM Studio 的 OpenAI 相容 API"""
    return init_chat_model(model=model, base_url=base_url, configurable_fields=configurable_fields, config_prefix=config_prefix, model_provider="openai", api_key=api_key, **kwargs)

try:
    # 嘗試連接 LM Studio 作為主要模型
    llm = init_LMStudio(model=LMSTUDIO_MODEL, base_url=LMSTUDIO_URL)
    print("✅ 成功連接 LM Studio (google/gemma-3-4B)")
except Exception as e:
    print(f"[LMStudio 初始化失敗] 無法連線至 {LMSTUDIO_URL}。錯誤: {e}")
    # 若失敗，回退到 Gemini 作為備用模型
    llm = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")
    print("⚠️ 使用 Gemini 作為備用模型")

#翻譯管理器 - 支援外部翻譯檔案
class TranslationManager:
    
    def __init__(self, translation_file_path=r"C:\Users\User\Studio\ChatSchool\CS_Project\CS_App\translate\translation_mapping.json"):
        self.translation_file = translation_file_path
        self.translation_dict = self._load_translation_dict()
    
    def _load_translation_dict(self) -> dict:
        try:
            with open(self.translation_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            print(f"✅ 成功載入翻譯映射檔案，條目數：{len(mapping)}")
            return mapping
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到翻譯映射檔案：{self.translation_file}，請先建立檔案。")
        except Exception as e:
            print(f"❌ 載入翻譯映射失敗: {e}")
            return {}
        
    def get_translation(self, chinese_text: str) -> Optional[str]:
        return self.translation_dict.get(chinese_text)
    
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
            # 部分命中，組合翻譯
            final_translation = remaining_text
            for i, (chinese_term, english_term) in enumerate(translated_parts):
                final_translation = final_translation.replace(f"__TRANSLATED_{i}__", english_term)
            
            # 如果還有剩餘中文，用 LMStudio 翻譯
            if is_chinese(final_translation):
                final_translation = self._lmstudio_translate(final_translation)
            
            print(f"🔄 混合翻譯: {text} → {final_translation}")
            return final_translation
        
        # 完全未命中，使用 LMStudio
        return self._lmstudio_translate(text)
    
    def _lmstudio_translate(self, text: str) -> str:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            translation_prompt = f"請將以下中文翻譯成英文，若有『聯合大學』的字樣，請一律翻成『National United University』。只回傳翻譯結果，不要加上任何解釋或前綴。\n\n原文：{text}"
            
            response = llm.invoke([
                SystemMessage(content='You are a professional translator. Translate Chinese to English only. Output only the translation without any explanation.'),
                HumanMessage(content=translation_prompt)
            ])
            
            translated = response.content.strip()
            
            # 移除可能的引號或前綴
            if translated.startswith('"') and translated.endswith('"'):
                translated = translated[1:-1]
            
            # 移除常見的前綴詞
            prefixes_to_remove = ["翻譯：", "Translation:", "英文：", "English:", "答：", "Answer:"]
            for prefix in prefixes_to_remove:
                if translated.startswith(prefix):
                    translated = translated[len(prefix):].strip()
            
            print(f"🤖 LMStudio翻譯: {text} → {translated}")
            return translated
            
        except Exception as e:
            print(f"❌ LMStudio翻譯失敗: {e}")
            return text

# 檢查文本是否包含中文字符
def is_chinese(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(chinese_pattern.search(text))

# 分割關鍵字函數
def split_keywords(s: str):
    return [kw.strip() for kw in re.split(r"[、,，]", s) if len(kw.strip()) > 1]

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

# 關鍵字搜尋結構
class Search(BaseModel):
    keywords: List[str]


#混合RAG系統：Weaviate語義檢索 + Neo4j關鍵字檢索 + 分類分析
class HybridRAG:
    
    # 初始化連接
    def __init__(self):
        self.weaviate_client = None
        self.neo4j_driver = None
        self.embedder = None
        self.weaviate_collection = None
        
        # 初始化翻譯管理器
        self.translator = TranslationManager()
        
        # 嘗試連接 Weaviate
        try:
            self.weaviate_client = connect_to_local(
                skip_init_checks=True,
                additional_config=wvc.init.AdditionalConfig(
                    timeout=wvc.init.Timeout(init=60, query=30, insert=30)
                )
            )
            print("✅ 成功連接到 Weaviate")
            
            # 載入 Weaviate 集合
            try:
                self.weaviate_collection = self.weaviate_client.collections.get("ContentSummaryEn0913")
                self.text_field = "english_summary"
                print(f"📚 已載入 ContentSummaryEn0913 集合")
            except Exception as e:
                print(f"⚠️ 無法載入 ContentSummaryEn0913 集合: {e}")
                self.weaviate_collection = None
                
        except Exception as e:
            print(f"❌ Weaviate 連接失敗: {e}")
            self.weaviate_client = None
        
        # 嘗試連接 Neo4j
        try:
            self.neo4j_driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "neo4j0804"),
                database="nuu-data-0804",
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
                connection_timeout=20,
                resolver=None
            )
            
            # 測試連接
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            print("✅ 成功連接到 Neo4j")
            
        except Exception as e:
            print(f"❌ Neo4j 連接失敗: {e}")
            self.neo4j_driver = None
        
        # 初始化嵌入器
        try:
            self.embedder = GeminiEmbedder()
            print("✅ 初始化 Gemini 嵌入器")
        except Exception as e:
            print(f"❌ 嵌入器初始化失敗: {e}")
            self.embedder = None
            
        
    # 基於分類的Neo4j關鍵字檢索
    def neo4j_keyword_search(self, keywords: List[str], limit: int = 10) -> List[Document]:
        if not self.neo4j_driver or not keywords:
            print("⚠️ Neo4j路徑不可用：缺少連接或關鍵字")
            return []
        
        try:
            print(f"🔍 [Neo4j關鍵字路徑] 搜尋關鍵字: {keywords}")
            return self._neo4j_global_search(keywords, limit)
            
        except Exception as e:
            print(f"❌ Neo4j 關鍵字搜尋失敗: {e}")
            return []
    def neo4j_keyword_search_enhanced(self, keywords: List[str], limit: int = 10) -> List[Document]:
        """
        Neo4j 增強檢索：關鍵字精確匹配 + 圖譜關聯分析
        """
        if not self.neo4j_driver or not keywords:
            print("⚠️ Neo4j路徑不可用：缺少連接或關鍵字")
            return []
        
        try:
            print(f"🔍 [Neo4j增強] 搜尋關鍵字: {keywords}")
            
            cypher = """
            // 1. 找到匹配的 Keyword 節點
            MATCH (k:Keyword)
            WHERE any(kw IN $keywords WHERE 
                toLower(k.name) CONTAINS toLower(kw) OR 
                toLower(kw) CONTAINS toLower(k.name))
            
            // 2. 透過圖譜找到相關文章
            MATCH (a:Article)-[:HAS_KEYWORD]->(k)
            
            // 3. 獲取文章的其他關鍵字（用於計算關聯度）
            WITH a, k, 
                [(a)-[:HAS_KEYWORD]->(ak:Keyword) | ak.name] as article_keywords
            
            // 4. 計算關鍵字匹配分數
            WITH a, k, article_keywords,
                CASE 
                    WHEN k.name IN $keywords THEN 2.0  // 完全匹配
                    ELSE 1.0  // 部分匹配
                END as keyword_score,
                // 計算關鍵字重疊度（文章包含多少個查詢關鍵字）
                size([kw IN $keywords WHERE any(article_k IN article_keywords WHERE 
                    toLower(article_k) CONTAINS toLower(kw))]) as keyword_overlap
            
            // 5. 獲取關鍵字的分類資訊
            OPTIONAL MATCH (k)-[:BELONGS_TO_CATEGORY]->(cat:Category)
            
            // 6. 獲取內容
            OPTIONAL MATCH (a)-[:HAS_CONTENT]->(c:Content)
            
            // 7. 計算最終分數
            WITH DISTINCT a, c, 
                collect(DISTINCT k.name) AS keywords_found,
                collect(DISTINCT cat.name) AS categories_found,
                article_keywords,
                max(keyword_score) as max_keyword_score,
                max(keyword_overlap) as overlap_count,
                // 綜合分數 = 匹配分數 + 重疊度加成
                (max(keyword_score) + max(keyword_overlap) * 0.5) as match_score
            
            RETURN 
                a.url AS article_url, 
                a.title AS article_title,
                a.domain AS article_domain,
                collect(DISTINCT {
                    id: c.id, 
                    text: c.text, 
                    type: c.type, 
                    order: c.order
                }) AS contents,
                keywords_found,
                categories_found,
                article_keywords,
                overlap_count,
                match_score
            ORDER BY match_score DESC, overlap_count DESC
            LIMIT $limit
            """
            
            with self.neo4j_driver.session() as session:
                result = session.run(cypher, keywords=keywords, limit=limit)
                records = result.data()
            
            print(f"✅ [Neo4j增強] 找到 {len(records)} 筆結果")
            
            # 顯示匹配詳情
            for i, rec in enumerate(records[:3], 1):
                print(f"  [{i}] 分數: {rec.get('match_score', 0):.2f} | "
                    f"重疊: {rec.get('overlap_count', 0)} | "
                    f"關鍵字: {rec.get('keywords_found', [])} | "
                    f"標題: {rec.get('article_title', '')[:40]}")
            
            return self._convert_neo4j_to_documents(records, "Neo4j_Enhanced")
            
        except Exception as e:
            print(f"❌ Neo4j 增強搜尋失敗: {e}")
            traceback.print_exc()
            return []
        
    # Neo4j 全域關鍵字搜尋
    def _neo4j_global_search(self, keywords: List[str], limit: int = 10) -> List[Document]:
        try:
            cypher = """
            MATCH (k:Keyword)
            WHERE any(kw IN $keywords WHERE 
                toLower(k.name) CONTAINS toLower(kw) OR 
                toLower(kw) CONTAINS toLower(k.name))
            
            MATCH (a:Article)-[:HAS_KEYWORD]->(k)
            OPTIONAL MATCH (a)-[:HAS_CONTENT]->(c:Content)
            OPTIONAL MATCH (k)-[:BELONGS_TO_CATEGORY]->(cat:Category)
            
            WITH DISTINCT a, c, k, cat,
                CASE 
                WHEN k.name IN $keywords THEN 1.0
                ELSE 0.8
                END as keyword_score
            
            RETURN a.url AS article_url, 
                a.title AS article_title,
                a.domain AS article_domain,
                collect(DISTINCT {
                    id: c.id, 
                    text: c.text, 
                    type: c.type, 
                    order: c.order
                }) AS contents,
                collect(DISTINCT k.name) AS keywords_found,
                collect(DISTINCT cat.name) AS categories_found,
                max(keyword_score) as match_score
            ORDER BY match_score DESC
            LIMIT $limit
            """
            
            with self.neo4j_driver.session() as session:
                result = session.run(cypher, keywords=keywords, limit=limit)
                records = result.data()
            
            print(f"✅ [全域搜尋] 找到 {len(records)} 筆結果")
            return self._convert_neo4j_to_documents(records, "Neo4j_Global")
            
        except Exception as e:
            print(f"❌ Neo4j 全域搜尋失敗: {e}")
            return []

    # 將Neo4j記錄轉換為Document格式
    def _convert_neo4j_to_documents(self, records: List[Dict], source_type: str) -> List[Document]:
        documents = []
        
        for rec in records:
            text_parts = []
            if rec.get('article_title'):
                text_parts.append(f"標題: {rec['article_title']}")
            if rec.get('article_url'):
                text_parts.append(f"網址: {rec['article_url']}")
            if rec.get('contents'):
                content_texts = []
                for content in rec['contents']:
                    if content and content.get('text'):
                        content_text = content['text']
                        content_type = content.get('type', '')
                        if content_type:
                            content_texts.append(f"[{content_type}] {content_text}")
                        else:
                            content_texts.append(content_text)
            if content_texts:
                text_parts.append(f"內容: {' '.join(content_texts)}")
            if rec.get('keywords_found'):
                text_parts.append(f"關鍵字: {', '.join(rec['keywords_found'])}")
            if rec.get('categories_found'):
                text_parts.append(f"分類: {', '.join(rec['categories_found'])}")
            if rec.get('contacts'):
                for contact in rec['contacts']:
                    if contact and contact.get('value'):
                        text_parts.append(f"聯絡方式({contact.get('type', '')}): {contact['value']} {contact.get('department', '')}")
            if rec.get('addresses'):
                for address in rec['addresses']:
                    if address and address.get('full_address'):
                        text_parts.append(f"地址: {address['full_address']} (城市: {address.get('city', '')}, 區域: {address.get('district', '')})")
            if rec.get('departments_found'):
                text_parts.append(f"部門: {', '.join(rec['departments_found'])}")
            
            content = "\n".join(text_parts)
            match_score = rec.get('match_score', 0.5)
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': source_type,
                    'article_url': rec.get('article_url', ''),
                    'article_title': rec.get('article_title', ''),
                    'match_score': match_score,
                    'enhanced': True
                }
            )
            documents.append(doc)
        
        return documents
    
    # Weaviate 語義檢索路徑
    def weaviate_search(self, query: str, limit: int = 10) -> List[Document]:
        if not self.weaviate_client or not self.embedder or not self.weaviate_collection or not self.neo4j_driver:
            print("⚠️ Weaviate路徑不可用：缺少必要組件")
            return []
        
        try:
            print(f"🔍 [Weaviate路徑] 進行語義檢索...")
            query_vector = self.embedder.embed_query(query)
            weaviate_results = self._weaviate_vector_search(query_vector, limit)
            
            if not weaviate_results:
                print("❌ Weaviate 搜尋無結果")
                return []
            
            # 提取 neo4j_id 並查詢 Neo4j 獲取完整資訊
            neo4j_ids = []
            weaviate_data = {}
            
            for result in weaviate_results:
                neo4j_id = result['neo4j_id']
                if neo4j_id:
                    neo4j_ids.append(neo4j_id)
                    weaviate_data[neo4j_id] = result
            
            if not neo4j_ids:
                print("⚠️ 沒有找到有效的 neo4j_id")
                return []
            
            print(f"🔍 [Weaviate路徑] 使用 {len(neo4j_ids)} 個 ID 查詢 Neo4j...")
            enhanced_documents = self._query_neo4j_by_ids(neo4j_ids, weaviate_data, source="Weaviate+Neo4j")
            
            print(f"✅ [Weaviate路徑] 生成 {len(enhanced_documents)} 筆增強文檔")
            return enhanced_documents
            
        except Exception as e:
            print(f"❌ Weaviate 搜尋失敗: {e}")
            return []
    def weaviate_search_with_prf_adaptive(self, query: str, limit: int = 10, use_prf: bool = True) -> List[Document]:
        """
        自適應 PRF：根據查詢複雜度動態調整參數
        """
        if not self.weaviate_client or not self.embedder or not self.weaviate_collection or not self.neo4j_driver:
            print("⚠️ Weaviate+PRF路徑不可用：缺少必要組件")
            return []
        
        try:
            # 階段0：評估查詢複雜度
            query_complexity = self._assess_query_complexity(query)
            print(f"📊 [PRF] 查詢複雜度評估: {query_complexity}")
            
            # 階段1：初始語義檢索
            print(f"🔍 [Weaviate+PRF] 階段1：初始語義檢索...")
            query_vector = self.embedder.embed_query(query)
            initial_results = self._weaviate_vector_search(query_vector, limit=10)
            
            if not initial_results:
                print("❌ Weaviate 初始搜尋無結果")
                return []
            
            print(f"✅ 獲得 {len(initial_results)} 筆初始結果")
            
            if not use_prf:
                return self._build_documents_from_weaviate(initial_results)
            
            #動態調整 PRF 參數
            prf_config = self._get_prf_config(query_complexity)
            print(f"⚙️ [PRF] 配置: Top-{prf_config['top_n']} 文章, "
                f"最多 {prf_config['max_keywords']} 個關鍵字, "
                f"過濾到 {prf_config['target_filtered']} 個")
            
            # 階段2：PRF 關鍵字擴展
            print(f"📊 [PRF] 階段2：從 Top-{prf_config['top_n']} 文檔提取關鍵字...")
            
            top_n_urls = []
            for result in initial_results[:prf_config['top_n']]:
                article_url = result.get('article_url')
                if article_url:
                    top_n_urls.append(article_url)
            
            if not top_n_urls:
                print("⚠️ 無法提取 URLs，返回初始結果")
                return self._build_documents_from_weaviate(initial_results)
            
            # 使用自適應圖譜擴展
            expanded_keywords = self._extract_and_expand_keywords_from_graph_adaptive(
                top_n_urls, 
                max_total_keywords=prf_config['max_keywords']
            )
            
            if not expanded_keywords:
                print("⚠️ PRF 無法擴展關鍵字，返回初始結果")
                return self._build_documents_from_weaviate(initial_results)
            
            # 階段3：Mistral 過濾
            print(f"🤖 [PRF] 階段3：用 Mistral 過濾關鍵字...")
            filtered_keywords = self._filter_keywords_with_lmstudio_adaptive(
                query, 
                expanded_keywords,
                target_count=prf_config['target_filtered']
            )
            
            if not filtered_keywords:
                print("⚠️ PRF 關鍵字過濾後無結果，返回初始結果")
                return self._build_documents_from_weaviate(initial_results)
            
            print(f"✅ [PRF] 過濾後的擴展關鍵字: {filtered_keywords}")
            
            # 階段4：擴展查詢重新檢索
            print(f"🔍 [PRF] 階段4：用擴展查詢重新檢索...")
            
            #  智能組合查詢（限制長度）
            max_expand_kw = min(3, len(filtered_keywords))
            expanded_query = f"{query} {' '.join(filtered_keywords[:max_expand_kw])}"
            
            # 防止查詢過長
            if len(expanded_query) > 200:
                print(f"⚠️ 擴展查詢過長 ({len(expanded_query)} 字元)，截斷到 200")
                expanded_query = expanded_query[:200]
            
            print(f"   擴展查詢: {expanded_query}")
            
            expanded_vector = self.embedder.embed_query(expanded_query)
            expanded_results = self._weaviate_vector_search(expanded_vector, limit=10)
            
            # 階段5：智能合併
            print(f"🔗 [PRF] 階段5：合併結果...")
            merged_results = self._merge_prf_results_smart(initial_results, expanded_results)
            
            print(f"✅ [PRF] 最終結果: {len(merged_results)} 筆")
            
            return self._build_documents_from_weaviate(merged_results)
            
        except Exception as e:
            print(f"❌ Weaviate+PRF 搜尋失敗: {e}")
            traceback.print_exc()
            return []
    def _assess_query_complexity(self, query: str) -> str:
        """
        評估查詢複雜度
        
        Returns:
            'simple' | 'medium' | 'complex'
        """
        word_count = len(query.split())
        has_multiple_entities = ',' in query or 'and' in query.lower() or '和' in query
        
        if word_count <= 5 and not has_multiple_entities:
            return 'simple'
        elif word_count <= 10:
            return 'medium'
        else:
            return 'complex'

    def _get_prf_config(self, complexity: str) -> dict:
        """
        根據複雜度返回 PRF 配置
        """
        configs = {
            'simple': {
                'top_n': 2,              # 只用 Top-2 文章
                'max_keywords': 30,      # 最多 30 個候選
                'target_filtered': 3     # 過濾到 3 個
            },
            'medium': {
                'top_n': 3,              # Top-3 文章
                'max_keywords': 50,      # 最多 50 個候選
                'target_filtered': 5     # 過濾到 5 個
            },
            'complex': {
                'top_n': 5,              # Top-5 文章
                'max_keywords': 80,      # 最多 80 個候選
                'target_filtered': 8     # 過濾到 8 個
            }
        }
        return configs.get(complexity, configs['medium'])
    # Weaviate 向量搜尋
    def _weaviate_vector_search(self, query_vector: List[float], limit: int) -> List[dict]:
        results = []
        
        try:
            response = self.weaviate_collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=wvc.query.MetadataQuery(distance=True)
            )
            
            for obj in response.objects:
                content = obj.properties.get(self.text_field, '')
                if not content or content.strip() == "":
                    continue
                
                distance = float(obj.metadata.distance) if obj.metadata.distance is not None else 1.0
                similarity = max(0, 1.0 - distance)
                
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
            
            return results
            
        except Exception as e:
            print(f"❌ Weaviate 向量搜尋失敗: {e}")
            return []
    
    # 根據 neo4j_id 列表查詢 Neo4j 中的完整資料
    def _query_neo4j_by_ids(self, neo4j_ids: List[str], weaviate_data: dict, source: str = "Enhanced") -> List[Document]:
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
                
                # 合併資料
                combined_data = {**content_data, **article_data, **org_data}
                
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
    def _extract_and_expand_keywords_from_graph_adaptive(self, article_urls: List[str], max_total_keywords: int = 50) -> List[str]:
        """
        動態自適應的圖譜關鍵字擴展
        
        Args:
            article_urls: Top-N 文章 URLs
            max_total_keywords: 最大關鍵字總數（預設 50）
        
        Returns:
            精選的關鍵字列表
        """
        
        query = """
        // 1. 找到 Top-N 文章
        UNWIND $article_urls AS url
        MATCH (a:Article {url: url})
        
        // 2. 提取這些文章的關鍵字
        MATCH (a)-[:HAS_KEYWORD]->(k:Keyword)
        
        // 3. 找到這些關鍵字的分類
        OPTIONAL MATCH (k)-[:BELONGS_TO_CATEGORY]->(cat:Category)
        
        //  4. 透過分類找相關關鍵字
        OPTIONAL MATCH (cat)<-[:BELONGS_TO_CATEGORY]-(related_k:Keyword)
        WHERE related_k <> k
        
        //  核心改進：在圖譜層面就限制每個原始關鍵字的擴展數量
        WITH k, cat, 
            k.name as original_keyword,
            cat.name as category,
            // 每個原始關鍵字最多擴展 10 個相關詞
            [related IN collect(DISTINCT related_k.name)[0..10] | related] as related_keywords,
            size(collect(DISTINCT related_k)) as total_related_count
        
        // 5. 計算擴展質量分數
        WITH original_keyword, 
            category,
            related_keywords,
            total_related_count,
            // 擴展質量 = 有限的擴展數（10個）和總擴展能力的平衡
            (size(related_keywords) * 1.0 / (1.0 + log(total_related_count))) as expansion_quality
        
        // 6. 按質量排序，限制總數
        ORDER BY expansion_quality DESC
        LIMIT $max_original_keywords  // 限制原始關鍵字數量（預設 10）
        RETURN original_keyword, category, related_keywords, total_related_count, expansion_quality
        """
        
        try:
            # 動態計算參數
            max_original_keywords = min(10, len(article_urls) * 3)  # 每篇文章最多 3 個原始關鍵字
            
            with self.neo4j_driver.session() as session:
                result = session.run(
                    query, 
                    article_urls=article_urls,
                    max_original_keywords=max_original_keywords
                )
                records = result.data()
            
            if not records:
                print("⚠️ 圖譜中未找到關鍵字")
                return []
            
            # 收集關鍵字：原始 + 限量擴展
            all_keywords = []
            keyword_sources = {}
            
            for record in records:
                # 原始關鍵字（高優先級）
                original = record.get('original_keyword')
                category = record.get('category')
                
                if original:
                    all_keywords.append(original)
                    keyword_sources[original] = 'original'
                
                # 擴展關鍵字（每個原始詞最多 10 個）
                related = record.get('related_keywords', [])
                for kw in related[:10]:  # 雙重保險
                    if kw:
                        all_keywords.append(kw)
                        keyword_sources[kw] = f'expanded_from_{category}'
            
            # 去重保持順序
            unique_keywords = []
            seen = set()
            for kw in all_keywords:
                if kw not in seen:
                    unique_keywords.append(kw)
                    seen.add(kw)
            
            # ⭐ 如果還是太多，按來源優先級截斷
            if len(unique_keywords) > max_total_keywords:
                print(f"⚠️ 關鍵字數量 {len(unique_keywords)} 超過限制 {max_total_keywords}，進行截斷")
                
                # 優先保留原始關鍵字
                original_kws = [kw for kw in unique_keywords if keyword_sources.get(kw) == 'original']
                expanded_kws = [kw for kw in unique_keywords if keyword_sources.get(kw) != 'original']
                
                remaining_slots = max_total_keywords - len(original_kws)
                unique_keywords = original_kws + expanded_kws[:remaining_slots]
            
            print(f"📊 [圖譜擴展] 從 {len(article_urls)} 篇文章提取:")
            print(f"   - 原始關鍵字: {sum(1 for v in keyword_sources.values() if v == 'original')} 個")
            print(f"   - 擴展關鍵字: {sum(1 for v in keyword_sources.values() if v.startswith('expanded'))} 個")
            print(f"   - 總計: {len(unique_keywords)} 個（限制 {max_total_keywords}）")
            
            return unique_keywords
            
        except Exception as e:
            print(f"❌ 圖譜關鍵字提取失敗: {e}")
            traceback.print_exc()
            return []
    def _filter_keywords_with_lmstudio_adaptive(self, original_query: str, candidate_keywords: List[str], target_count: int = None) -> List[str]:
        """
        動態自適應的關鍵字過濾（使用 LMStudio）
        """
        try:
            # 動態計算候選數量上限
            if len(candidate_keywords) > 100:
                print(f"⚠️ 候選關鍵字過多 ({len(candidate_keywords)} 個)，智能採樣到 50 個")
                
                step = len(candidate_keywords) // 20
                sampled = (
                    candidate_keywords[:20] +
                    candidate_keywords[20:-10:max(1, step)] +
                    candidate_keywords[-10:]
                )
                candidates = list(dict.fromkeys(sampled))[:50]
            else:
                candidates = candidate_keywords[:30]
            
            if not candidates:
                return []
            
            # 動態計算目標數量
            if target_count is None:
                query_words = len(original_query.split())
                if query_words <= 5:
                    target_count = "3-5"
                elif query_words <= 10:
                    target_count = "5-8"
                else:
                    target_count = "8-12"
            else:
                target_count = str(target_count)
            
            prompt = f"""你是一個關鍵字過濾專家。給定原始查詢和候選關鍵字列表，請選出最相關的 {target_count} 個關鍵字用於擴展查詢。

    原始查詢：{original_query}

    候選關鍵字列表（共 {len(candidates)} 個）：
    {', '.join(candidates)}

    任務：
    1. 選出與原始查詢語義最相關的關鍵字
    2. 優先選擇能幫助找到更多相關文檔的關鍵字
    3. 排除不相關或過於寬泛的關鍵字
    4. ⭐ 嚴格控制數量在 {target_count} 個以內

    請只回傳選中的關鍵字，用逗號分隔，不要有其他文字。
    例如：關鍵字1, 關鍵字2, 關鍵字3"""

            from langchain_core.messages import SystemMessage, HumanMessage
            
            response = llm.invoke([
                SystemMessage(content='You are a keyword filter expert. Select most relevant keywords only.'),
                HumanMessage(content=prompt)
            ])
            
            filtered_text = response.content.strip()
            
            # 解析回應
            filtered_keywords = [kw.strip() for kw in filtered_text.split(',') if kw.strip()]
            
            # 強制限制數量
            max_count = int(target_count.split('-')[-1]) if '-' in target_count else int(target_count)
            filtered_keywords = filtered_keywords[:max_count]
            
            print(f"🤖 [LMStudio過濾] {len(candidates)} 個候選 → {len(filtered_keywords)} 個保留")
            if filtered_keywords:
                print(f"   保留的關鍵字: {filtered_keywords}")
            
            return filtered_keywords
            
        except Exception as e:
            print(f"❌ LMStudio 過濾失敗: {e}")
            fallback_count = 5 if len(candidate_keywords) > 10 else 3
            fallback = candidate_keywords[:fallback_count]
            print(f"⚠️ 使用備案關鍵字: {fallback}")
            return fallback
    def _merge_prf_results_smart(self, initial_results: List[dict], expanded_results: List[dict]) -> List[dict]:
        """
        智能合併：保留初始 Top-10（主要），補充 PRF 新發現的文檔
        研究建議：原始查詢保留主要權重，擴展為輔
        """
        seen_ids = set()
        merged = []
        
        # 第一優先：保留初始 Top-10（相似度不變，這些是最相關的）
        print(f"🔗 [合併] 階段1：保留初始 Top-10")
        for i, result in enumerate(initial_results[:10], 1):
            result_id = result.get('weaviate_uuid') or result.get('neo4j_id')
            if result_id:
                result_copy = result.copy()
                result_copy['prf_source'] = 'initial_top10'
                result_copy['rank'] = i
                merged.append(result_copy)
                seen_ids.add(result_id)
        
        print(f"   已加入 {len(merged)} 筆初始結果")
        
        # 第二優先：補充 PRF 擴展發現的新文檔（權重降低 0.5）
        print(f"🔗 [合併] 階段2：補充 PRF 新發現的文檔")
        prf_added = 0
        for result in expanded_results:
            result_id = result.get('weaviate_uuid') or result.get('neo4j_id')
            if result_id and result_id not in seen_ids:
                result_copy = result.copy()
                # 降低 PRF 補充文檔的權重（因為是擴展查詢找到的）
                result_copy['similarity'] = result.get('similarity', 0) * 0.5
                result_copy['prf_source'] = 'prf_supplement'
                merged.append(result_copy)
                seen_ids.add(result_id)
                prf_added += 1
                
                # 最多補充 5 筆
                if prf_added >= 5:
                    break
        
        print(f"   PRF 補充了 {prf_added} 筆新文檔")
        
        # 排序：初始結果優先，然後按相似度
        merged.sort(key=lambda x: (
            0 if x.get('prf_source') == 'initial_top10' else 1,  # 初始結果排前面
            -x.get('similarity', 0)  # 相似度降序
        ))
        
        final_count = min(len(merged), 15)  # 最多返回 15 筆
        print(f"🔗 [合併] 最終: {final_count} 筆文檔（初始 {len(initial_results)} + PRF補充 {prf_added}）")
        
        return merged[:15]
    def _build_documents_from_weaviate(self, weaviate_results: List[dict]) -> List[Document]:
        """
        從 Weaviate 結果建立文檔（需要查詢 Neo4j 獲取完整資訊）
        """
        if not weaviate_results:
            return []
        
        neo4j_ids = []
        weaviate_data = {}
        
        for result in weaviate_results:
            neo4j_id = result.get('neo4j_id')
            if neo4j_id:
                neo4j_ids.append(neo4j_id)
                weaviate_data[neo4j_id] = result
        
        if not neo4j_ids:
            print("⚠️ 沒有找到有效的 neo4j_id")
            return []
        
        print(f"🔍 使用 {len(neo4j_ids)} 個 ID 查詢 Neo4j 獲取完整資訊...")
        enhanced_documents = self._query_neo4j_by_ids(neo4j_ids, weaviate_data, source="Weaviate+Neo4j+PRF")
    
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
        
    # 獲取 Organization 資訊
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
        
    # 優化版增強文檔建立
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
            
            # 組合內容
            enhanced_content = "\n\n".join(content_parts) if content_parts else "無內容"
            
            # 建立 metadata
            metadata = {
                'source': 'Weaviate+Neo4j_optimized',
                'neo4j_id': neo4j_data.get('content_neo4j_id', ''),
                'similarity': weaviate_info.get('similarity', 0),
                'enhanced': True,
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
                    'enhanced': False
                }
            )
            
    # 建立增強文檔 (保留備用)
    def _create_enhanced_document(self, neo4j_record, weaviate_info: dict, source: str) -> Document:
        try:
            base_content = weaviate_info.get('english_summary', '')
            content_parts = [base_content] if base_content else []
            
            content_text = neo4j_record.get('content_text')
            if content_text and content_text != base_content:
                content_parts.append(f"【原始內容】\n{content_text}")
            
            article_title = neo4j_record.get('article_title')
            if article_title:
                content_parts.append(f"【文章標題】\n{article_title}")
            
            article_domain = neo4j_record.get('article_domain')
            if article_domain:
                content_parts.append(f"【網站來源】\n{article_domain}")
            
            # 添加聯絡資訊
            contacts = neo4j_record.get('contacts', [])
            valid_contacts = [c for c in contacts if c.get('value')]
            if valid_contacts:
                contact_lines = []
                for contact in valid_contacts:
                    contact_type = contact.get('type', '聯絡方式')
                    contact_value = contact.get('value', '')
                    contact_dept = contact.get('department', '')
                    
                    contact_line = f"{contact_type}: {contact_value}"
                    if contact_dept:
                        contact_line += f" ({contact_dept})"
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
                    city = addr.get('city', '')
                    district = addr.get('district', '')
                    
                    addr_line = full_addr
                    if city and district:
                        addr_line += f" ({city}{district})"
                    elif city:
                        addr_line += f" ({city})"
                    
                    address_lines.append(addr_line)
                
                if address_lines:
                    content_parts.append(f"【地址資訊】\n" + "\n".join(address_lines))
            
            # 添加部門資訊
            departments = neo4j_record.get('departments', [])
            valid_departments = [d for d in departments if d.get('name')]
            if valid_departments:
                dept_lines = []
                for dept in valid_departments:
                    dept_name = dept.get('name', '')
                    dept_type = dept.get('type', '')
                    
                    dept_line = dept_name
                    if dept_type:
                        dept_line += f" ({dept_type})"
                    dept_lines.append(dept_line)
                
                if dept_lines:
                    content_parts.append(f"【所屬部門】\n" + "\n".join(dept_lines))
            
            enhanced_content = "\n\n".join(content_parts)
            
            metadata = {
                'source': source,
                'weaviate_uuid': weaviate_info.get('weaviate_uuid', ''),
                'neo4j_id': neo4j_record.get('content_neo4j_id', ''),
                'similarity': weaviate_info.get('similarity', 0),
                'distance': weaviate_info.get('distance', 1.0),
                'enhanced': True,
                'article_title': article_title or '',
                'article_url': neo4j_record.get('article_url', ''),
                'article_domain': article_domain or '',
                'contact_count': len(valid_contacts),
                'address_count': len(valid_addresses),
                'department_count': len(valid_departments),
            }
            
            return Document(page_content=enhanced_content, metadata=metadata)
            
        except Exception as e:
            print(f"❌ 建立增強文檔失敗: {e}")
            return Document(
                page_content=weaviate_info.get('english_summary', '建立文檔失敗'),
                metadata={
                    'source': f'{source}_error',
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
    keywords: List[str]
    weaviate_context: List[Document]
    neo4j_context: List[Document]
    merged_context: List[Document]
    answer: str
    related_links: List[Dict]

# 初始化混合RAG系統
hybrid_rag = HybridRAG()

# 意圖推理 - 使用 Mistral 分析並澄清用戶問題意圖
def infer_intent(state: State):
    original_question = state["question"]
    
    print(f"🧠 開始意圖推理: {original_question}")
    
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # 構建意圖推理提示詞
        intent_prompt = f"""你是一個智能問題分析助手，專門分析用戶在大學資訊系統中的查詢意圖。

用戶問題：「{original_question}」

請分析這個問題並以 JSON 格式回答：

{{
  "問題類型": "查詢類/諮詢類/比較類/其他",
  "核心意圖": "用戶真正想要的核心需求",
  "關鍵實體": ["涉及的系所", "部門", "服務", "人員等"],
  "時空限制": ["時間限制", "地點限制", "條件限制等"],
  "澄清問題": ["可能的明確解釋1", "可能的明確解釋2"],
  "建議關鍵字": ["最重要的關鍵字1", "關鍵字2", "關鍵字3"],
  "檢索策略": "需要的資訊類型（聯絡方式/地址/政策文件等）",
  "最佳查詢": "重新表述的明確問題"
}}

請確保回傳有效的 JSON 格式。"""

        # 調用 LMStudio 進行意圖推理
        response = llm.invoke([
            SystemMessage(content='You are an intent analysis expert. Analyze queries and return valid JSON only.'),
            HumanMessage(content=intent_prompt)
        ])
        
        intent_result = response.content.strip()
        
        # 清理 JSON 格式
        if intent_result.startswith('```json'):
            intent_result = intent_result.replace('```json', '').replace('```', '').strip()
        elif intent_result.startswith('```'):
            intent_result = intent_result.replace('```', '').strip()
        
        # 嘗試解析 JSON 回應
        try:
            intent_analysis = json.loads(intent_result)
            
            print(f"✅ 意圖推理完成:")
            print(f"   📋 問題類型: {intent_analysis.get('問題類型', 'N/A')}")
            print(f"   🎯 核心意圖: {intent_analysis.get('核心意圖', 'N/A')}")
            print(f"   🏷️  關鍵實體: {intent_analysis.get('關鍵實體', [])}")
            print(f"   ⏰ 時空限制: {intent_analysis.get('時空限制', [])}")
            print(f"   🔑 建議關鍵字: {intent_analysis.get('建議關鍵字', [])}")
            print(f"   🔍 檢索策略: {intent_analysis.get('檢索策略', 'N/A')}")
            print(f"   📝 最佳查詢: {intent_analysis.get('最佳查詢', 'N/A')}")
            
            # 更新問題為最佳查詢版本（如果有）
            best_query = intent_analysis.get('最佳查詢', '')
            if best_query and best_query.strip() and best_query != original_question:
                refined_question = best_query.strip()
                print(f"📝 問題重新表述: {original_question} → {refined_question}")
            else:
                refined_question = original_question
            
            return {
                "question": refined_question,
                "original_question": original_question,
                "intent_analysis": intent_analysis,
                "refined_question": refined_question
            }
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失敗，使用原始問題: {e}")
            print(f"原始回應: {intent_result}")
            return {
                "question": original_question,
                "original_question": original_question,
                "intent_analysis": {},
                "refined_question": original_question
            }
            
    except Exception as e:
        print(f"❌ 意圖推理失敗: {e}")
        traceback.print_exc()
        return {
            "question": original_question,
            "original_question": original_question,
            "intent_analysis": {},
            "refined_question": original_question
        }
    
# 增強的問題處理 - 整合意圖推理結果
def enhanced_process_question(state: State):
    question = state["question"]
    intent_analysis = state.get("intent_analysis", {})
    original_question = state.get("original_question", question)
    
    # 檢查是否包含中文
    is_chinese_query = is_chinese(original_question)
    
    if is_chinese_query:
        print(f"🈶 檢測到中文查詢，準備翻譯...")
        try:
            translated_question = hybrid_rag.translator.translate_with_mapping(question)
            print(f"🔄 翻譯結果: {translated_question}")
        except Exception as e:
            print(f"⚠️ 翻譯失敗，使用原始問題: {e}")
            translated_question = question
            is_chinese_query = False
    else:
        print(f"🔤 檢測到英文查詢，直接使用")
        translated_question = question
    
    # 擷取關鍵字 - 整合意圖推理的建議
    try:
        # 先嘗試使用意圖推理的建議關鍵字
        suggested_keywords = intent_analysis.get("建議關鍵字", [])
        key_entities = intent_analysis.get("關鍵實體", [])
        
        if suggested_keywords or key_entities:
            # 結合建議關鍵字和關鍵實體
            combined_suggested = list(set(suggested_keywords + key_entities))
            print(f"💡 使用意圖推理建議的關鍵字: {combined_suggested}")
            keywords = [kw.strip() for kw in combined_suggested if len(kw.strip()) > 1]
        else:
            # 後備：使用 LLM 擷取關鍵字
            structured_llm = llm.with_structured_output(Search)
            query = structured_llm.invoke(
                f"""請從使用者的問題中擷取有意義的關鍵詞：
                - 優先保留完整詞組，例如科系名稱「資訊工程」、機構名、地點、活動名稱等。
                - 若遇到人名，請只擷取純人名，不要包含職稱或身份稱呼詞。
                - 其餘關鍵詞可兩字為一單位。
                - 排除虛詞。

                使用者問題：{original_question}"""
            )
            
            keywords = [kw.strip() for kw in query.keywords if len(kw.strip()) > 1]
            print(f"📝 LLM擷取關鍵字: {keywords}")
        
    except Exception as e:
        print(f"❌ 關鍵字擷取失敗: {e}")
        keywords = []
    
    return {
        "question": translated_question,
        "original_question": original_question,
        "translated_question": translated_question,
        "is_chinese_query": is_chinese_query,
        "keywords": keywords,
        "intent_analysis": intent_analysis
    }

#並行檢索：Weaviate語義檢索 + Neo4j分類關鍵字檢索
def parallel_retrieve(state: State):
    translated_question = state["translated_question"]
    keywords = state["keywords"]
    
    print(f"\n🔄 開始並行檢索（Weaviate含PRF + Neo4j增強）...")
    print(f"📝 Weaviate 查詢: {translated_question}")
    print(f"🔑 Neo4j 關鍵字: {keywords}")
    
    import concurrent.futures
    
    weaviate_results = []
    neo4j_results = []
    
    def weaviate_search_task():
        return hybrid_rag.weaviate_search_with_prf_adaptive( 
            translated_question, limit=10, use_prf=True
        )
    
    def neo4j_search_task():
        # 使用增強的 Neo4j 檢索（關鍵字匹配 + 圖譜關聯分析）
        return hybrid_rag.neo4j_keyword_search_enhanced(keywords, limit=10)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        weaviate_future = executor.submit(weaviate_search_task)
        neo4j_future = executor.submit(neo4j_search_task)
        
        try:
            weaviate_results = weaviate_future.result(timeout=45)  # PRF 需要更多時間
        except Exception as e:
            print(f"⚠️ Weaviate+PRF 檢索失敗: {e}")
            traceback.print_exc()
            weaviate_results = []
        
        try:
            neo4j_results = neo4j_future.result(timeout=30)
        except Exception as e:
            print(f"⚠️ Neo4j 檢索失敗: {e}")
            traceback.print_exc()
            neo4j_results = []
    
    print(f"\n📊 並行檢索結果:")
    print(f"🔍 Weaviate+PRF 路徑: {len(weaviate_results)} 筆結果")
    print(f"🔍 Neo4j增強 路徑: {len(neo4j_results)} 筆結果")
    
    return {
        "weaviate_context": weaviate_results,
        "neo4j_context": neo4j_results
    }

#合併和去重兩個檢索路徑的結果
def merge_results(state: State):
    weaviate_docs = state["weaviate_context"]
    neo4j_docs = state["neo4j_context"]
    
    print(f"\n🔗 開始合併結果...")
    
    # 使用URL進行去重
    seen_urls = set()
    merged_docs = []
    
    # 優先加入 Weaviate 結果（語義相似度較高）
    for doc in weaviate_docs:
        url = doc.metadata.get('article_url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            merged_docs.append(doc)
            print(f"✅ 加入Weaviate結果: {doc.metadata.get('article_title', 'No Title')[:50]}")
        elif not url:  # 沒有URL的也加入
            merged_docs.append(doc)
    
    # 加入 Neo4j 結果（避免重複）
    for doc in neo4j_docs:
        url = doc.metadata.get('article_url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            merged_docs.append(doc)
            print(f"✅ 加入Neo4j結果: {doc.metadata.get('article_title', 'No Title')[:50]}")
        elif not url:  # 沒有URL的也加入
            merged_docs.append(doc)
    
    # 按相似度和來源進行排序
    def sort_key(doc):
        similarity = doc.metadata.get('similarity', 0)
        match_score = doc.metadata.get('match_score', 0)
        is_enhanced = doc.metadata.get('enhanced', False)
        source_priority = 0
        
        if 'Weaviate+Neo4j' in doc.metadata.get('source', ''):
            source_priority = 3  # 最高優先級
        elif 'Neo4j_Category' in doc.metadata.get('source', ''):
            source_priority = 2.5  # 分類搜尋優先於全域搜尋
        elif 'Neo4j_Global' in doc.metadata.get('source', ''):
            source_priority = 2
        else:
            source_priority = 1
        
        # 綜合分數：來源優先級 + 語義相似度 + 關鍵字匹配分數
        combined_score = source_priority + similarity + match_score
        return (combined_score, is_enhanced)
    
    merged_docs.sort(key=sort_key, reverse=True)
    
    # 限制最終結果數量
    final_docs = merged_docs[:15]
    
    print(f"🎯 最終合併結果: {len(final_docs)} 筆文檔")
    
    # 顯示前5筆結果概述
    for i, doc in enumerate(final_docs[:5], 1):
        source = doc.metadata.get('source', 'Unknown')
        similarity = doc.metadata.get('similarity', 0)
        match_score = doc.metadata.get('match_score', 0)
        title = doc.metadata.get('article_title', 'No Title')
        enhanced = doc.metadata.get('enhanced', False)
        
        print(f"  [{i}] {source} | 相似度: {similarity:.3f} | 匹配: {match_score:.3f} | {'增強' if enhanced else '基礎'} | {title[:30]}")
    
    return {"merged_context": final_docs}

# 混合RAG生成回答
def generate(state: State):
    docs = state["merged_context"]
     # ===== 新增：收集 related_links =====
    related_links = []
    seen_urls = set()
    
    for doc in docs:
        article_url = doc.metadata.get('article_url', '')
        article_title = doc.metadata.get('article_title', '')
        
        # 去重並加入連結
        if article_url and article_url not in seen_urls:
            seen_urls.add(article_url)
            related_links.append({
                'url': article_url,
                'title': article_title or '相關文章',
                'source': doc.metadata.get('source', 'KGRAG'),
                'similarity': doc.metadata.get('similarity', 0),
                'match_score': doc.metadata.get('match_score', 0)
            })
    
    # 限制連結數量（例如最多 10 個）
    related_links = related_links[:10]
    # =====================================
    # 整理檢索內容
    context_text = ""
    for i, doc in enumerate(docs, 1):
        similarity = doc.metadata.get('similarity', 0)
        match_score = doc.metadata.get('match_score', 0)
        enhanced = doc.metadata.get('enhanced', False)
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content
        article_url = doc.metadata.get('article_url', '')
        article_title = doc.metadata.get('article_title', '')
        
        context_text += f"[資料 {i}] 來源: {source} (相似度: {similarity:.3f}, 匹配: {match_score:.3f})\n"
        if article_title:
            context_text += f"文章標題: {article_title}\n"
        if article_url:
            context_text += f"文章網址: {article_url}\n"
        context_text += f"{content}\n\n"
    
    # 根據是否為中文查詢選擇不同的提示模板
    if state["is_chinese_query"]:
        # 中文查詢 - 用中文回答
        template = """你是國立聯合大學的智能助手，請根據以下混合檢索資料回答用戶問題。
        這些資料來自兩個檢索路徑：
        1. Weaviate ContentSummaryEn0913 向量語義搜尋 + Neo4j 圖譜增強
        2. Neo4j 分類關鍵字檢索（優先在預測分類中搜尋）+ 全域後備搜尋

        ## 混合檢索與增強資料：
        {context}

        ## 用戶原始問題（中文）：
        {original_question}

        ## 翻譯後的查詢問題（英文）：
        {translated_question}

        ## 關鍵字分析結果：
        擷取關鍵字: {keywords}

        ## 回答指引：
        1. **請用中文回答用戶的問題**
        2. 優先使用來源標註為 "Weaviate+Neo4j" 的增強資料
        3. 其次使用 "Neo4j_Category" 的分類搜尋結果
        4. 如有相關網址請提供
        5. 如果有聯絡資訊，請提供具體的聯絡方式（電話、信箱等）
        6. 如果有地址資訊，請提供詳細的地址位置
        7. 如果有部門資訊，請說明相關的部門和職責
        8. 優先引用相似度和匹配分數較高的資料
        9. 如果資料不足，請誠實說明，但盡可能提供相關資訊
        10. 回答要條理清晰，重要資訊用項目符號列出
        11. **重要：請不要在回答中加入任何資料編號引用**
        12. **直接使用資料內容回答問題，保持自然流暢**
        13.不要輸出(資料4）這種內容

        ## 回答："""
    else:
        # 英文查詢 - 用英文回答
        template = """You are an intelligent assistant for National United University. Please answer the user's question based on the following hybrid retrieval data.
        This data comes from two retrieval paths:
        1. Weaviate ContentSummaryEn0913 vector semantic search + Neo4j graph enhancement
        2. Neo4j category keyword search (prioritizing predicted categories) + global fallback search

        ## Hybrid Retrieval and Enhanced Data:
        {context}

        ## User Question:
        {original_question}

        ## Keywords Analysis:
        Extracted Keywords: {keywords}

        ## Answer Guidelines:
        1. **Please answer in English**
        2. Prioritize using enhanced data marked as "Weaviate+Neo4j" source
        3. Then use "Neo4j_Category" category search results
        4. Provide contact information, addresses, and department details when available
        5. Prioritize data with higher similarity and match scores
        6. If data is insufficient, explain honestly but provide relevant information
        7. Structure the answer clearly with bullet points for important information
        8. **Do not include any data source numbers in your answer**
        9. **Keep the answer natural and fluent**

        ## Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    messages = prompt.invoke({
        "context": context_text,
        "original_question": state["original_question"],
        "translated_question": state.get("translated_question", state["original_question"]),
        "keywords": ", ".join(state.get("keywords", [])),
    })
    
    response = llm.invoke(messages)
    return {
        "answer": response.content,
        "related_links": related_links  # 新增
    }


# 建立增強的混合RAG流程 - 包含意圖推理
def create_enhanced_graph():
    enhanced_graph_builder = StateGraph(State).add_sequence([
        infer_intent,         
        enhanced_process_question,    
        parallel_retrieve, 
        merge_results,
        generate
    ])
    enhanced_graph_builder.add_edge(START, "infer_intent")
    return enhanced_graph_builder.compile()


# 增強混合RAG問答主函數 - 包含意圖推理
def ask_enhanced_hybrid_rag(question: str) -> tuple:  # 修改回傳類型
    """
    回傳：(answer: str, related_links: List[Dict])
    """
    start_time = time.time()
    
    enhanced_graph = create_enhanced_graph()
    result = enhanced_graph.invoke({
        "question": question, 
        "original_question": question
    })
    
    end_time = time.time()
    
    print(f"⏱️ 總處理時間：{end_time - start_time:.2f} 秒")
    
    # 回傳答案和連結
    return result["answer"], result.get("related_links", [])



if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"🎓 國立聯合大學 混合RAG問答系統 (LMStudio版)")
    print(f"{'='*70}")
    print(f"📊 系統架構: 雙路並行檢索 + 分類分析 + 結果融合")
    print(f"🤖 主要模型: LM Studio (google/gemma-3-4B)")
    print(f"🛤️  路徑1: Weaviate語義檢索 + PRF → Neo4j圖譜增強")
    print(f"🛤️  路徑2: Neo4j分類關鍵字檢索 → 全域後備搜尋")
    print(f"🔗 結果融合: URL去重 + 多重分數排序")
    print(f"{'='*70}\n")
    
    # 檢查系統狀態
    weaviate_status = "✅ 正常" if hybrid_rag.weaviate_client else "❌ 失敗"
    neo4j_status = "✅ 正常" if hybrid_rag.neo4j_driver else "❌ 失敗"
    translation_file_status = "✅ 已載入" if hybrid_rag.translator.translation_dict else "❌ 載入失敗"
    
    print(f"🔗 Weaviate 連接: {weaviate_status}")
    print(f"🔗 Neo4j 連接: {neo4j_status}")
    print(f"🔗 LM Studio: {'✅ 正常' if isinstance(llm, BaseChatModel) else '⚠️ 使用備用模型'}")
    print(f"📄 翻譯映射檔案: {translation_file_status}")
    
    
    if not hybrid_rag.weaviate_client and not hybrid_rag.neo4j_driver:
        print("\n⚠️ 注意：所有數據庫連接失敗，系統將運行在演示模式")
    elif not hybrid_rag.weaviate_client:
        print("\n⚠️ Weaviate 連接失敗，將僅使用 Neo4j 關鍵字檢索")
    elif not hybrid_rag.neo4j_driver:
        print("\n⚠️ Neo4j 連接失敗，將僅使用 Weaviate 語義檢索")
    
    # 互動式問答
    try:
        while True:
            print("-" * 60)
            question = input("請輸入您的問題 (中文或英文，輸入 'exit' 退出): ")
            
            if question.lower() in ['exit', 'quit', '退出', '離開']:
                print("謝謝使用，再見！")
                break
            
            if not question.strip():
                continue
                
            print(f"\n🤖 正在處理...")
            try:
                answer = ask_enhanced_hybrid_rag(question)
                print(f"\n💡 回答:\n{answer}\n")
                        
            except Exception as e:
                print(f"處理問題時發生錯誤: {e}")
                traceback.print_exc()
                
    except KeyboardInterrupt:
        print(f"\n\n程式中斷，謝謝使用！")