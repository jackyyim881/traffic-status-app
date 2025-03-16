from transformers import pipeline, AutoModelForCausalLM, AutoModel, BertForQuestionAnswering
from datasets import load_dataset
import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time
from typing import List, Dict, Tuple, Any, Optional
import os
import json

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("carpark_qa_system")


class CarparkQASystem:
    """香港停車場問答系統"""

    def __init__(self,
                 dataset_name: str = "hkdata/hongkong_carpark",
                 embedding_model_name: str = "bert-base-chinese",
                 qa_model_name: str = "bert-base-chinese",
                 cache_dir: str = "./cache"):
        """
        初始化問答系統

        Args:
            dataset_name: 資料集名稱
            embedding_model_name: 用於語義嵌入的模型
            qa_model_name: 用於問答的模型
            cache_dir: 用於緩存的目錄
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 載入資料集
        start_time = time.time()
        logger.info(f"正在載入資料集: {dataset_name}")
        try:
            self.dataset = load_dataset(dataset_name)
            self.df = self.dataset['train'].to_pandas()
            logger.info(
                f"已載入資料集，共 {len(self.df)} 筆資料，耗時 {time.time() - start_time:.2f}秒")
        except Exception as e:
            logger.error(f"載入資料集失敗: {e}")
            raise

        # 載入模型 - 使用正確的模型加載方式
        logger.info(f"正在載入嵌入模型: {embedding_model_name}")
        try:
            # 修正：使用正確的模型類別和方法
            self.embedding_tokenizer = AutoModelForCausalLM.from_pretrained(
                embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model_name)

            # 建立嵌入管道
            self.embedding_pipeline = pipeline(
                "feature-extraction",
                model=self.embedding_model,
                tokenizer=self.embedding_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("嵌入模型載入成功")
        except Exception as e:
            logger.error(f"載入嵌入模型失敗: {e}")
            raise

        logger.info(f"正在載入問答模型: {qa_model_name}")
        try:
            # 使用正確的QA模型類別
            self.qa_tokenizer = AutoModelForCausalLM.from_pretrained(
                qa_model_name)
            self.qa_model = BertForQuestionAnswering.from_pretrained(
                qa_model_name)

            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.qa_model,
                tokenizer=self.qa_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("問答模型載入成功")
        except Exception as e:
            logger.error(f"載入問答模型失敗: {e}")
            raise

        # 處理並索引資料集
        self._process_dataset()

    def _process_dataset(self):
        """處理資料集並為搜尋創建嵌入"""
        cache_file = os.path.join(self.cache_dir, "carpark_embeddings.npy")
        search_texts_file = os.path.join(self.cache_dir, "search_texts.json")

        # 檢查緩存的嵌入和搜尋文本
        if os.path.exists(cache_file) and os.path.exists(search_texts_file):
            logger.info("正在載入緩存的嵌入和搜尋文本")
            self.embeddings = np.load(cache_file)

            # 從JSON文件加載搜尋文本
            with open(search_texts_file, 'r', encoding='utf-8') as f:
                self.search_texts = json.load(f)

            logger.info(f"已加載 {len(self.search_texts)} 筆搜尋文本和嵌入")
        else:
            logger.info("正在創建搜尋索引和嵌入")
            # 創建豐富的文本表示
            self.search_texts = []
            for _, row in self.df.iterrows():
                # 創建包含所有相關字段的全面文本表示
                text = (f"區域: {row['region']} | 地址: {row['address']} | "
                        f"地區: {row['area']} | carpark name: {row['carpark name']}")

                self.search_texts.append(text)

            # 生成嵌入
            embeddings_list = []
            batch_size = 16
            for i in range(0, len(self.search_texts), batch_size):
                batch = self.search_texts[i:i+batch_size]
                logger.info(
                    f"正在處理批次 {i//batch_size + 1}/{len(self.search_texts)//batch_size + 1}")
                batch_embeddings = self.embedding_pipeline(batch)
                # 平均池化以獲取句子嵌入
                for emb in batch_embeddings:
                    embeddings_list.append(np.mean(emb, axis=0))

            self.embeddings = np.vstack(embeddings_list)

            # 保存嵌入和搜尋文本到緩存
            np.save(cache_file, self.embeddings)
            with open(search_texts_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_texts, f, ensure_ascii=False, indent=2)

            logger.info(f"嵌入已創建並緩存，形狀為 {self.embeddings.shape}")

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        在停車場資料集上執行語義搜尋

        Args:
            query: 搜尋查詢
            top_k: 要返回的頂部結果數量

        Returns:
            包含 (索引, 相似度得分) 的元組列表
        """
        # 獲取查詢嵌入
        query_embedding = self.embedding_pipeline(query)
        query_embedding = np.mean(query_embedding[0], axis=0).reshape(1, -1)

        # 計算相似度
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # 獲取前k個結果
        # 確保我們不會請求超過我們擁有的結果數量
        top_k = min(top_k, len(similarities))
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]

    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        執行關鍵字搜尋作為備用

        Args:
            query: 搜尋查詢
            top_k: 要返回的頂部結果數量

        Returns:
            包含 (索引, 得分) 的元組列表
        """
        results = []
        query_keywords = query.lower().split()

        for i, text in enumerate(self.search_texts):
            text_lower = text.lower()
            score = sum(
                1 for keyword in query_keywords if keyword in text_lower)
            if score > 0:
                results.append((i, score))

        results.sort(key=lambda x: x[1], reverse=True)

        # 確保我們不會請求超過我們擁有的結果數量
        top_k = min(top_k, len(results))
        return results[:top_k]

    def search_carparks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        根據查詢使用語義和關鍵字搜尋來搜尋停車場

        Args:
            query: 搜尋查詢
            top_k: 要返回的頂部結果數量

        Returns:
            停車場信息字典列表
        """
        # 確保 top_k 不會超過資料集大小
        top_k = min(top_k, len(self.df))

        try:
            semantic_results = self.semantic_search(query, top_k)

            # 如果語義搜尋沒有產生好結果，回退到關鍵字搜尋
            if not semantic_results or (semantic_results and semantic_results[0][1] < 0.5):
                logger.info("語義搜尋結果不足，回退到關鍵字搜尋")
                search_results = self.keyword_search(query, top_k)
            else:
                search_results = semantic_results
        except Exception as e:
            logger.error(f"搜尋過程中出錯: {e}")
            search_results = self.keyword_search(query, top_k)

        # 如果仍然沒有結果，返回空列表
        if not search_results:
            return []

        # 轉換為完整的停車場信息
        results = []
        for idx, score in search_results:
            # 確保索引在有效範圍內
            if 0 <= idx < len(self.df):
                carpark_info = self.df.iloc[idx].to_dict()
                carpark_info['similarity_score'] = float(score)
                results.append(carpark_info)
            else:
                logger.warning(f"索引 {idx} 超出了範圍 (0-{len(self.df)-1})")

        return results

    def format_carpark_info(self, carpark: Dict[str, Any]) -> str:
        """格式化停車場信息以供顯示"""
        info = f"停車場資訊 (相關度: {carpark.get('similarity_score', 0):.2f}):\n"
        info += f"區域: {carpark.get('region', 'N/A')}\n"
        info += f"地址: {carpark.get('address', 'N/A')}\n"
        info += f"地區: {carpark.get('area', 'N/A')}\n"
        info += f"carpark name: {carpark.get('carpark name', 'N/A')}\n"

        # 添加任何可用的額外信息
        for key, value in carpark.items():
            if key not in ['region', 'address', 'area', 'carpark name', 'similarity_score'] and value and isinstance(value, (str, int, float)):
                info += f"{key}: {value}\n"

        return info

    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        回答關於香港停車場的查詢

        Args:
            query: 中文的用戶查詢

        Returns:
            包含答案、上下文和匹配停車場的字典
        """
        start_time = time.time()
        logger.info(f"處理查詢: {query}")

        try:
            carpark_matches = self.search_carparks(query, top_k=3)
            processing_time = time.time() - start_time
        except Exception as e:
            logger.error(f"搜尋停車場時出錯: {e}")
            return {
                "answer": f"處理查詢時發生錯誤: {str(e)}",
                "context": "",
                "carparks": [],
                "processing_time": time.time() - start_time
            }

        if not carpark_matches:
            logger.info("找不到匹配的停車場")
            return {
                "answer": "找不到符合條件的停車場。請嘗試提供更多細節或使用不同的關鍵詞。",
                "context": "",
                "carparks": [],
                "processing_time": processing_time
            }

        # 格式化停車場信息
        context = "以下是找到的停車場資訊：\n\n"
        for carpark in carpark_matches:
            context += self.format_carpark_info(carpark) + "\n"

        logger.info(f"找到 {len(carpark_matches)} 個匹配的停車場")

        try:
            # 從問答模型獲取答案
            qa_result = self.qa_pipeline(question=query, context=context)
            answer = qa_result['answer']
            confidence = qa_result['score']

            # 如果置信度低，提供更一般的回應
            if confidence < 0.5:
                answer = context

            logger.info(f"查詢處理完成，耗時 {processing_time:.2f}秒")
            return {
                "answer": answer,
                "context": context,
                "carparks": carpark_matches,
                "confidence": float(confidence),
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"問答管道中出錯: {e}")
            return {
                "answer": context,
                "context": context,
                "carparks": carpark_matches,
                "error": str(e),
                "processing_time": processing_time
            }


# 為系統創建一個簡單的API
class CarparkAPI:
    """停車場問答系統的簡單API包裝器"""

    def __init__(self):
        """使用問答系統初始化API"""
        logger.info("初始化停車場問答API")
        self.qa_system = CarparkQASystem()
        logger.info("API準備好處理請求")

    def process_query(self, query: str) -> Dict[str, Any]:
        """處理用戶查詢並返回格式化結果"""
        if not query.strip():
            return {"status": "error", "message": "查詢不能為空"}

        try:
            result = self.qa_system.answer_query(query)
            return {
                "status": "success",
                "query": query,
                "answer": result["answer"],
                "carparks": [
                    {
                        "region": cp.get("region", "N/A"),
                        "address": cp.get("address", "N/A"),
                        "area": cp.get("area", "N/A"),
                        "similarity": cp.get("similarity_score", 0)
                    }
                    for cp in result.get("carparks", [])
                ],
                "processing_time": result["processing_time"]
            }
        except Exception as e:
            logger.error(f"API錯誤: {e}")
            return {"status": "error", "message": f"處理查詢時發生錯誤: {str(e)}"}
