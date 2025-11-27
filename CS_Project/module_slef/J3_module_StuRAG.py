#新0804data - ContentSummaryEn0913 集合 (英文摘要) + Neo4j 圖譜資料
#已改新的向量
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
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Mistral API 錯誤：{response.status_code}, {response.text}")
    return response.json()

# 翻譯管理器 - 支援外部翻譯檔案
class TranslationManager:

    def __init__(self, translation_file_path=r"C:\Users\User\Studio\ChatSchool\CS_Project\CS_App\translate\translation_mapping.json"):
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
            
            # 初始化嵌入器
            self.embedder = GeminiEmbedder()
            print("✅ 初始化 Gemini 嵌入器")
            
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
    original_question: str  # 保存原始中文問題
    translated_question: str  # 保存翻譯後的英文問題
    is_chinese_query: bool  # 標記是否為中文查詢
    context: List[Document]
    answer: str
    related_links: List[Dict]  # 新增

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
    # 使用翻譯後的英文問題進行檢索
    question = state["question"]
    documents = content_summary_rag.search(question, limit=10)
    
    print(f"\n📊 檢索結果統計:")
    print(f"🔍 總共檢索到 {len(documents)} 筆相關資料")
    
    enhanced_count = sum(1 for doc in documents if doc.metadata.get('enhanced', False))
    print(f"✨ 其中 {enhanced_count} 筆已通過 Neo4j 增強")
    
    # 顯示前3筆結果的詳細資訊
    for i, doc in enumerate(documents[:3], 1):
        similarity = doc.metadata.get('similarity', 0)
        enhanced = doc.metadata.get('enhanced', False)
        article_title = doc.metadata.get('article_title', '')
        content_type = doc.metadata.get('content_type', '')
        
        print(f"\n  [{i}] {'✅增強' if enhanced else '⚪基礎'} | 相似度: {similarity:.3f}")
        if article_title:
            print(f"      標題: {article_title}")
        if content_type:
            print(f"      類型: {content_type}")
        
        if enhanced:
            contact_count = doc.metadata.get('contact_count', 0)
            address_count = doc.metadata.get('address_count', 0)
            dept_count = doc.metadata.get('department_count', 0)
            related_count = doc.metadata.get('related_content_count', 0)
            print(f"      📈 增強資訊: 聯絡{contact_count}筆 | 地址{address_count}筆 | 部門{dept_count}筆 | 相關內容{related_count}筆")
        
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"      內容預覽: {content_preview}")
    
    return {"context": documents}

# 生成回答
def generate(state: State):
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
        # 中文查詢 - 用中文回答
        template = """你是國立聯合大學的智能助手，請根據以下檢索資料回答用戶問題。
        這些資料來自 Weaviate ContentSummaryEn0913 集合的向量搜尋，並通过 Neo4j 圖譜資料庫進行了增強，包含了完整的聯絡資訊、地址、部門等詳細資料。

        ## 檢索與增強資料：
        {context}

        ## 用戶原始問題（中文）：
        {original_question}

        ## 翻譯後的查詢問題（英文）：
        {translated_question}

        ## 回答指引：
        1. **請用中文回答用戶的問題**
        2. 優先使用增強後的完整資料回答問題
        3. 如果有聯絡資訊，請提供具體的聯絡方式（電話、信箱等）
        4. 如果有地址資訊，請提供詳細的地址位置
        5. 如果有部門資訊，請說明相關的部門和職責
        6. 如果有相關內容，可以適當引用來豐富回答
        7. 優先引用相似度較高的資料
        8. 如果資料不足，請誠實說明，但盡可能提供相關資訊
        9. 回答要條理清晰，重要資訊用項目符號列出
        10. 特別注意提供實用的聯絡方式和地址等用戶關心的具體資訊
        11. 如有相關網址請提出
        12. **重要：請不要在回答中加入任何資料編號引用，如 (資料1)、(資料2) 等標記**
        13. **直接使用資料內容回答問題，不需要標註資料來源編號**
        14. **回答應該自然流暢，不包含任何括號內的引用標記**

        ## 回答："""
    else:
        # 英文查詢 - 用英文回答
        template = """You are an intelligent assistant for National United University. Please answer the user's question based on the following retrieved and enhanced data.
        This data comes from Weaviate ContentSummaryEn0913 collection vector search and has been enhanced through Neo4j graph database, including comprehensive contact information, addresses, department details, etc.

        ## Retrieved and Enhanced Data:
        {context}

        ## User Question:
        {original_question}

        ## Answer Guidelines:
        1. **Please answer in English**
        2. Prioritize using the enhanced comprehensive data to answer the question
        3. If there is contact information, please provide specific contact methods (phone, email, etc.)
        4. If there is address information, please provide detailed location information
        5. If there is department information, please mention relevant departments and responsibilities
        6. If there are related contents, you can appropriately reference them to enrich your answer
        7. Prioritize citing data with higher similarity scores
        8. If the data is insufficient, please honestly explain but provide relevant information as much as possible
        9. Answer should be well-structured, with important information listed in bullet points
        10. Pay special attention to providing practical contact methods and addresses that users care about
        11. **Important: Do not include any data source numbers or citations like (Data 1), (Data 2) in your answer**
        12. **Use the data content directly without marking the source numbers**
        13. **The answer should be natural and fluent without any citation marks in parentheses**

        ## Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    messages = prompt.invoke({
        "context": context_text,
        "original_question": state["original_question"],
        "translated_question": state.get("translated_question", state["original_question"])
    })
    
    response = llm.invoke(messages)
    # === 新增：收集 related_links ===
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
                'source': doc.metadata.get('source', 'StuRAG'),
                'similarity': doc.metadata.get('similarity', 0),
                'enhanced': doc.metadata.get('enhanced', False)
            })
    
    related_links = related_links[:10]
    # =====================================
    return {
        "answer": response.content,
        "related_links": related_links
    }

# 建立 RAG 流程
graph_builder = StateGraph(State).add_sequence([process_question, retrieve, generate])
graph_builder.add_edge(START, "process_question")
graph = graph_builder.compile()

# RAG 問答主函數
def ask_question(question: str) -> tuple:
    """
    回傳：(answer: str, related_links: List[Dict])
    """
    start_time = time.time()  
    result = graph.invoke({"question": question})
    end_time = time.time()   
    
    print(f"⏱️ 總處理時間：{end_time - start_time:.2f} 秒")
    
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