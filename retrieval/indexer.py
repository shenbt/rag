import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import os
# 避免循环导入，使用延迟导入
# from models.evidence_cards import EvidenceCard, EvidenceCardCollection

class HaystackRetriever:
    """
    Haystack规模化检索器：混合检索 + 候选重排，适配大规模文档库
    """
    def __init__(self, 
                 sparse_model_name="BM25",
                 dense_model_name="./all-MiniLM-L6-v2",
                 cross_encoder_name="./ms-marco-MiniLM-L-6-v2",
                 max_docs=1000):
        self.sparse_model_name = sparse_model_name
        self.dense_model_name = dense_model_name
        self.cross_encoder_name = cross_encoder_name
        self.max_docs = max_docs
        
        # 文档存储
        self.documents = []
        self.doc_embeddings = []
        self.doc_metadata = []
        
        # 初始化模型
        self._init_models()
        
        # 检索参数
        self.sparse_top_k = 50
        self.dense_top_k = 50
        self.final_top_k = 10

    def _init_models(self):
        """初始化检索模型"""
        # 稀疏检索模型（BM25）- 直接使用简化版避免Haystack兼容性问题
        print("🔄 使用简化版BM25（避免Haystack兼容性问题）")
        self.sparse_retriever = SimpleBM25Retriever()
        self.document_store = None
        
        try:
            # 稠密检索模型
            print(f"🔄 正在加载稠密检索模型: {self.dense_model_name}")
            self.dense_tokenizer = AutoTokenizer.from_pretrained(
                self.dense_model_name,
                trust_remote_code=True,
                cache_dir="./model_cache"
            )
            self.dense_model = AutoModel.from_pretrained(
                self.dense_model_name,
                trust_remote_code=True,
                cache_dir="./model_cache"
            )
            print("✅ 稠密检索模型初始化成功")
            
        except Exception as e:
            print(f"⚠️ 稠密检索模型初始化失败: {e}")
            print("🔄 尝试使用备用模型...")
            try:
                # 尝试使用更小的模型
                backup_model = "./paraphrase-MiniLM-L3-v2"
                print(f"🔄 尝试加载备用模型: {backup_model}")
                self.dense_tokenizer = AutoTokenizer.from_pretrained(
                    backup_model,
                    trust_remote_code=True,
                    cache_dir="./model_cache"
                )
                self.dense_model = AutoModel.from_pretrained(
                    backup_model,
                    trust_remote_code=True,
                    cache_dir="./model_cache"
                )
                print("✅ 备用稠密检索模型初始化成功")
            except Exception as e2:
                print(f"⚠️ 备用模型也失败: {e2}")
                self.dense_tokenizer = None
                self.dense_model = None
        
        try:
            # 交叉编码器（重排模型）
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(self.cross_encoder_name)
            print("✅ 交叉编码器初始化成功")
            
        except Exception as e:
            print(f"⚠️ 交叉编码器初始化失败: {e}")
            self.cross_encoder = None

    def add_documents(self, documents: List[Dict[str, Any]]):
        """添加文档到索引"""
        for doc in documents:
            self.documents.append(doc)
            self.doc_metadata.append({
                'doc_id': doc.get('id', len(self.documents)),
                'content': doc.get('content', ''),
                'title': doc.get('title', ''),
                'metadata': doc.get('metadata', {})
            })
        
        # Haystack 不可用时，给 SimpleBM25 构建索引（需要有 id 字段）
        if not hasattr(self, 'document_store'):
            simple_docs = []
            for i, doc in enumerate(documents):
                simple_docs.append({
                    'id': doc.get('id', i),
                    'content': doc.get('content', '')
                })
            self.sparse_retriever.add_documents(simple_docs)
        
        # 计算稠密嵌入
        if self.dense_model is not None:
            self._compute_dense_embeddings()

    def _compute_dense_embeddings(self):
        """计算文档的稠密嵌入"""
        if not self.documents:
            return
        
        embeddings = []
        for doc in self.documents:
            content = doc.get('content', '')
            if content:
                embedding = self._encode_text(content)
                embeddings.append(embedding)
            else:
                embeddings.append(np.zeros(384))  # 默认维度
        
        self.doc_embeddings = np.array(embeddings)

    def _encode_text(self, text: str) -> np.ndarray:
        """编码文本"""
        if self.dense_tokenizer is None or self.dense_model is None:
            return np.random.randn(384)
        
        try:
            inputs = self.dense_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.dense_model(**inputs)
                # 使用[CLS] token的表示
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                return embedding.flatten()
                
        except Exception as e:
            print(f"文本编码失败: {e}")
            return np.random.randn(384)

    def retrieve(self, query: str):
        """混合检索"""
        # 延迟导入避免循环导入
        from models.evidence_cards import EvidenceCard, EvidenceCardCollection
        
        evidence_cards = EvidenceCardCollection()
        
        # 1. 稀疏检索
        sparse_results = self._sparse_retrieval(query)
        
        # 2. 稠密检索
        dense_results = self._dense_retrieval(query)
        
        # 3. 结果融合
        combined_results = self._combine_results(sparse_results, dense_results)
        
        # 4. 交叉编码器重排
        if self.cross_encoder is not None and combined_results:
            try:
                reranked_results = self._cross_encoder_rerank(query, combined_results)
            except Exception as e:
                print(f"⚠️ 交叉编码器重排失败，使用原始结果: {e}")
                reranked_results = combined_results
        else:
            reranked_results = combined_results
        
        # 5. 转换为证据卡
        for i, result in enumerate(reranked_results[:self.final_top_k]):
            evidence_card = EvidenceCard(
                ocr_text=result['content'],
                bbox=result.get('bbox', [0, 0, 0, 0]),
                page_id=result.get('doc_id', f'doc_{i}'),
                confidence=result.get('score', 0.5),
                source_type=result.get('type', 'text')
            )
            
            evidence_card.set_importance_score(result.get('score', 0.5))
            evidence_card.set_relevance_score(result.get('relevance', 0.5))
            
            evidence_card.add_metadata('rank', i + 1)
            evidence_card.add_metadata('retrieval_method', result.get('method', 'hybrid'))
            
            evidence_cards.add_card(evidence_card)
        
        return evidence_cards

    def _sparse_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """稀疏检索（BM25）"""
        try:
            if hasattr(self, 'document_store') and self.document_store is not None:
                # 使用Haystack的BM25
                try:
                    results = self.sparse_retriever.retrieve(
                        query=query,
                        top_k=self.sparse_top_k
                    )
                    
                    sparse_results = []
                    for result in results:
                        sparse_results.append({
                            'content': result.content,
                            'score': result.score,
                            'doc_id': result.meta.get('doc_id', 'unknown'),
                            'method': 'sparse'
                        })
                    
                    return sparse_results
                except Exception as e:
                    print(f"⚠️ Haystack BM25检索失败: {e}")
                    # 回退到简化版BM25
                    results = self.sparse_retriever.retrieve(query, self.sparse_top_k)
                    # SimpleBM25 返回 {'id','content','score','method'}，下面统一成 'doc_id'
                    for r in results:
                        r['doc_id'] = r.get('doc_id', r.get('id', 'unknown'))
                    return results
            else:
                # 使用简化版BM25
                results = self.sparse_retriever.retrieve(query, self.sparse_top_k)
                # SimpleBM25 返回 {'id','content','score','method'}，下面统一成 'doc_id'
                for r in results:
                    r['doc_id'] = r.get('doc_id', r.get('id', 'unknown'))
                return results
                
        except Exception as e:
            print(f"稀疏检索失败: {e}")
            return []

    def _dense_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """稠密检索"""
        if self.dense_model is None or not self.doc_embeddings:
            return []
        
        try:
            # 编码查询
            query_embedding = self._encode_text(query)
            
            # 计算相似度
            similarities = []
            for doc_embedding in self.doc_embeddings:
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append(similarity)
            
            # 排序
            sorted_indices = np.argsort(similarities)[::-1]
            
            dense_results = []
            for i in sorted_indices[:self.dense_top_k]:
                if i < len(self.doc_metadata):
                    dense_results.append({
                        'content': self.doc_metadata[i]['content'],
                        'score': similarities[i],
                        'doc_id': self.doc_metadata[i]['doc_id'],
                        'method': 'dense'
                    })
            
            return dense_results
            
        except Exception as e:
            print(f"稠密检索失败: {e}")
            return []

    def _combine_results(self, sparse_results: List[Dict], 
                        dense_results: List[Dict]) -> List[Dict]:
        """融合稀疏和稠密检索结果"""
        # 创建文档ID到结果的映射
        doc_scores = {}
        
        # 处理稀疏检索结果
        for result in sparse_results:
            doc_id = result['doc_id']
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'content': result['content'],
                    'doc_id': doc_id,
                    'sparse_score': result['score'],
                    'dense_score': 0.0,
                    'combined_score': 0.0
                }
            else:
                doc_scores[doc_id]['sparse_score'] = result['score']
        
        # 处理稠密检索结果
        for result in dense_results:
            doc_id = result['doc_id']
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'content': result['content'],
                    'doc_id': doc_id,
                    'sparse_score': 0.0,
                    'dense_score': result['score'],
                    'combined_score': 0.0
                }
            else:
                doc_scores[doc_id]['dense_score'] = result['score']
        
        # 计算综合评分
        for doc_id, scores in doc_scores.items():
            # 加权融合
            sparse_weight = 0.3
            dense_weight = 0.7
            scores['combined_score'] = (
                sparse_weight * scores['sparse_score'] +
                dense_weight * scores['dense_score']
            )
        
        # 排序
        combined_results = list(doc_scores.values())
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined_results

    def _cross_encoder_rerank(self, query: str, 
                             candidates: List[Dict]) -> List[Dict]:
        """交叉编码器重排"""
        if self.cross_encoder is None or not candidates:
            print(f"⚠️ 交叉编码器为空或候选列表为空: cross_encoder={self.cross_encoder is not None}, candidates={len(candidates) if candidates else 0}")
            return candidates
        
        print(f"🔄 开始交叉编码器重排: 查询='{query[:50]}...', 候选数量={len(candidates)}")
        
        try:
            # 准备输入
            pairs = []
            valid_indices = []  # 记录有效的候选索引
            
            for i, candidate in enumerate(candidates):
                content = candidate.get('content', '')
                if content and len(content.strip()) > 0:  # 确保内容不为空且非空字符串
                    pairs.append([query, content])
                    valid_indices.append(i)
                else:
                    print(f"⚠️ 候选 {i} 内容为空或无效: {repr(content[:50]) if content else 'None'}")
            
            print(f"📝 准备了 {len(pairs)} 个有效输入对, 有效索引: {valid_indices}")
            
            if not pairs:  # 如果没有有效的输入对
                print(f"⚠️ 没有有效的候选内容进行重排，返回原始候选")
                return candidates
            
            # 计算重排分数
            try:
                rerank_scores = self.cross_encoder.predict(pairs)
                if not isinstance(rerank_scores, (list, np.ndarray)):
                    print(f"⚠️ 交叉编码器返回的分数格式异常: {type(rerank_scores)}")
                    return candidates
            except Exception as e:
                print(f"⚠️ 交叉编码器预测失败: {e}")
                return candidates
            
            # 确保rerank_scores是列表
            if isinstance(rerank_scores, np.ndarray):
                rerank_scores = rerank_scores.tolist()
            
            # 验证分数数量
            if len(rerank_scores) != len(pairs):
                print(f"⚠️ 分数数量({len(rerank_scores)})与输入对数量({len(pairs)})不匹配")
                return candidates
            
            # 更新分数
            for i, candidate in enumerate(candidates):
                if i in valid_indices:
                    # 找到对应的分数索引
                    score_idx = valid_indices.index(i)
                    if score_idx < len(rerank_scores):
                        candidate['rerank_score'] = rerank_scores[score_idx]
                        combined_score = candidate.get('combined_score', 0.0)
                        candidate['final_score'] = (
                            0.3 * combined_score +
                            0.7 * rerank_scores[score_idx]
                        )
                    else:
                        # 如果索引越界，使用默认分数
                        candidate['rerank_score'] = 0.0
                        candidate['final_score'] = candidate.get('combined_score', 0.0)
                else:
                    # 对于无效的候选，使用默认分数
                    candidate['rerank_score'] = 0.0
                    candidate['final_score'] = candidate.get('combined_score', 0.0)
            
            # 按最终分数排序
            candidates.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
            
            return candidates
            
        except Exception as e:
            print(f"交叉编码器重排失败: {e}")
            return candidates

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"计算余弦相似度失败: {e}")
            return 0.0

    def save_index(self, path: str):
        """保存索引"""
        try:
            index_data = {
                'documents': self.documents,
                'metadata': self.doc_metadata,
                'embeddings': self.doc_embeddings.tolist() if hasattr(self.doc_embeddings, 'tolist') else []
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 索引已保存到 {path}")
        except Exception as e:
            print(f"保存索引失败: {e}")

    def load_index(self, path: str):
        """加载索引"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            self.documents = index_data.get('documents', [])
            self.doc_metadata = index_data.get('metadata', [])
            self.doc_embeddings = np.array(index_data.get('embeddings', []))
            
            print(f"✅ 索引已从 {path} 加载")
        except Exception as e:
            print(f"加载索引失败: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            'total_documents': len(self.documents),
            'total_embeddings': len(self.doc_embeddings) if hasattr(self.doc_embeddings, '__len__') else 0,
            'sparse_model': self.sparse_model_name,
            'dense_model': self.dense_model_name,
            'cross_encoder': self.cross_encoder_name
        }


class SimpleBM25Retriever:
    """
    简化版BM25检索器（当Haystack不可用时使用）
    """
    def __init__(self):
        self.documents = []
        self.term_freq = {}
        self.doc_freq = {}
        self.avg_doc_length = 0.0
        self.total_docs = 0
        
        # BM25参数
        self.k1 = 1.2
        self.b = 0.75

    def add_documents(self, documents: List[Dict[str, Any]]):
        """添加文档"""
        self.documents = documents
        self._build_index()

    def _build_index(self):
        """构建索引"""
        self.total_docs = len(self.documents)
        
        # 计算词频和文档频率
        for doc in self.documents:
            content = doc.get('content', '')
            terms = content.split()
            doc_length = len(terms)
            self.avg_doc_length += doc_length
            
            # 词频统计
            for term in terms:
                if term not in self.term_freq:
                    self.term_freq[term] = {}
                if doc.get('id') not in self.term_freq[term]:
                    self.term_freq[term][doc.get('id')] = 0
                self.term_freq[term][doc.get('id')] += 1
                
                # 文档频率
                if term not in self.doc_freq:
                    self.doc_freq[term] = set()
                self.doc_freq[term].add(doc.get('id'))
        
        if self.total_docs > 0:
            self.avg_doc_length /= self.total_docs

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """检索"""
        query_terms = query.split()
        scores = {}
        
        for doc in self.documents:
            doc_id = doc.get('id')
            content = doc.get('content', '')
            doc_terms = content.split()
            doc_length = len(doc_terms)
            
            score = 0.0
            for term in query_terms:
                if term in self.term_freq and doc_id in self.term_freq[term]:
                    tf = self.term_freq[term][doc_id]
                    df = len(self.doc_freq[term])
                    
                    # BM25公式
                    idf = np.log((self.total_docs - df + 0.5) / (df + 0.5))
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                    
                    score += idf * numerator / denominator
            
            scores[doc_id] = score
        
        # 排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = next((d for d in self.documents if d.get('id') == doc_id), None)
            if doc:
                results.append({
                    'content': doc.get('content', ''),
                    'score': score,
                    'doc_id': doc_id,
                    'method': 'sparse'
                })
        
        return results
