import faiss
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from models.evidence_cards import EvidenceCard, EvidenceCardCollection
import json

class TwoStageRetriever:
    """
    两阶段检索器：Page-level粗检索 + Region-level精排
    """
    def __init__(self, page_dim=768, region_dim=768):
        self.page_dim = page_dim
        self.region_dim = region_dim
        
        # Page-level索引
        self.page_index = faiss.IndexFlatL2(page_dim)
        self.page_embeddings = []
        self.page_metadata = []
        
        # Region-level索引
        self.region_index = faiss.IndexFlatL2(region_dim)
        self.region_embeddings = []
        self.region_metadata = []
        
        # 证据卡集合
        self.evidence_collection = EvidenceCardCollection()
        
        # 检索参数
        self.page_top_k = 10
        self.region_top_k = 20
        self.rerank_top_k = 5

    def add_page(self, page_embedding: np.ndarray, page_metadata: Dict[str, Any]):
        """添加页面级文档"""
        if isinstance(page_embedding, torch.Tensor):
            page_embedding = page_embedding.detach().cpu().numpy()
        
        if page_embedding.ndim == 1:
            page_embedding = page_embedding.reshape(1, -1)
        
        self.page_embeddings.append(page_embedding)
        self.page_metadata.append(page_metadata)
        self.page_index.add(page_embedding)

    def add_region(self, region_embedding: np.ndarray, region_metadata: Dict[str, Any]):
        """添加区域级文档"""
        if isinstance(region_embedding, torch.Tensor):
            region_embedding = region_embedding.detach().cpu().numpy()
        
        if region_embedding.ndim == 1:
            region_embedding = region_embedding.reshape(1, -1)
        
        self.region_embeddings.append(region_embedding)
        self.region_metadata.append(region_metadata)
        self.region_index.add(region_embedding)

    def retrieve(self, query_embedding: np.ndarray, query_text: str = "") -> EvidenceCardCollection:
        """
        两阶段检索：Page-level粗检索 + Region-level精排
        """
        # 第一阶段：Page-level粗检索
        candidate_pages = self._page_level_retrieval(query_embedding)
        
        # 第二阶段：Region-level精排
        evidence_cards = self._region_level_rerank(query_embedding, candidate_pages, query_text)
        
        return evidence_cards

    def _page_level_retrieval(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Page-level粗检索"""
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # 搜索候选页面
        k = min(self.page_top_k, len(self.page_embeddings))
        if k == 0:
            return []
        
        distances, indices = self.page_index.search(query_embedding, k)
        
        candidate_pages = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.page_metadata):
                candidate_pages.append({
                    'page_id': self.page_metadata[idx].get('page_id', f'page_{idx}'),
                    'distance': distances[0][i],
                    'metadata': self.page_metadata[idx]
                })
        
        return candidate_pages

    def _region_level_rerank(self, query_embedding: np.ndarray, 
                           candidate_pages: List[Dict[str, Any]], 
                           query_text: str) -> EvidenceCardCollection:
        """Region-level精排"""
        evidence_cards = EvidenceCardCollection()
        
        # 获取候选页面中的区域
        candidate_regions = []
        for page_info in candidate_pages:
            page_id = page_info['page_id']
            # 找到属于该页面的所有区域
            for i, region_meta in enumerate(self.region_metadata):
                if region_meta.get('page_id') == page_id:
                    candidate_regions.append({
                        'region_idx': i,
                        'metadata': region_meta,
                        'page_distance': page_info['distance']
                    })
        
        if not candidate_regions:
            return evidence_cards
        
        # 计算区域与查询的相关性
        region_scores = []
        for region_info in candidate_regions:
            region_idx = region_info['region_idx']
            if region_idx < len(self.region_embeddings):
                region_embedding = self.region_embeddings[region_idx]
                
                # 计算语义相似度
                semantic_score = self._calculate_similarity(query_embedding, region_embedding)
                
                # 计算综合评分（结合页面距离和语义相似度）
                combined_score = self._calculate_combined_score(
                    semantic_score, region_info['page_distance']
                )
                
                region_scores.append({
                    'region_idx': region_idx,
                    'semantic_score': semantic_score,
                    'combined_score': combined_score,
                    'metadata': region_info['metadata']
                })
        
        # 按综合评分排序
        region_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # 创建证据卡
        for i, region_score in enumerate(region_scores[:self.rerank_top_k]):
            region_idx = region_score['region_idx']
            metadata = region_score['metadata']
            
            # 创建证据卡
            evidence_card = EvidenceCard(
                ocr_text=metadata.get('ocr_text', ''),
                bbox=metadata.get('bbox', [0, 0, 0, 0]),
                page_id=metadata.get('page_id', ''),
                confidence=metadata.get('confidence', 1.0),
                source_type=metadata.get('source_type', 'text')
            )
            
            # 设置评分
            evidence_card.set_importance_score(region_score['combined_score'])
            evidence_card.set_relevance_score(region_score['semantic_score'])
            
            # 添加元数据
            evidence_card.add_metadata('rank', i + 1)
            evidence_card.add_metadata('page_distance', region_score.get('page_distance', 0))
            evidence_card.add_metadata('semantic_score', region_score['semantic_score'])
            
            evidence_cards.add_card(evidence_card)
        
        return evidence_cards

    def _calculate_similarity(self, query_embedding: np.ndarray, 
                             region_embedding: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            # 归一化向量
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            region_norm = region_embedding / (np.linalg.norm(region_embedding) + 1e-8)
            
            # 计算余弦相似度
            similarity = np.dot(query_norm.flatten(), region_norm.flatten())
            return max(0, similarity)  # 确保非负
        except Exception as e:
            print(f"计算相似度时出错: {e}")
            return 0.0

    def _calculate_combined_score(self, semantic_score: float, page_distance: float) -> float:
        """计算综合评分"""
        # 将页面距离转换为相似度（距离越小，相似度越高）
        page_similarity = 1.0 / (1.0 + page_distance)
        
        # 加权组合
        semantic_weight = 0.7
        page_weight = 0.3
        
        combined_score = semantic_weight * semantic_score + page_weight * page_similarity
        return combined_score

    def save_index(self, page_index_path: str = "./page_index", 
                  region_index_path: str = "./region_index"):
        """保存索引"""
        try:
            faiss.write_index(self.page_index, page_index_path)
            faiss.write_index(self.region_index, region_index_path)
            
            # 保存元数据
            with open(f"{page_index_path}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.page_metadata, f, ensure_ascii=False, indent=2)
            
            with open(f"{region_index_path}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.region_metadata, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 索引已保存到 {page_index_path} 和 {region_index_path}")
        except Exception as e:
            print(f"保存索引时出错: {e}")

    def load_index(self, page_index_path: str = "./page_index", 
                  region_index_path: str = "./region_index"):
        """加载索引"""
        try:
            self.page_index = faiss.read_index(page_index_path)
            self.region_index = faiss.read_index(region_index_path)
            
            # 加载元数据
            with open(f"{page_index_path}_metadata.json", 'r', encoding='utf-8') as f:
                self.page_metadata = json.load(f)
            
            with open(f"{region_index_path}_metadata.json", 'r', encoding='utf-8') as f:
                self.region_metadata = json.load(f)
            
            print(f"✅ 索引已从 {page_index_path} 和 {region_index_path} 加载")
        except Exception as e:
            print(f"加载索引时出错: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            'page_count': len(self.page_metadata),
            'region_count': len(self.region_metadata),
            'page_index_size': self.page_index.ntotal,
            'region_index_size': self.region_index.ntotal
        }


class LayoutPlanner:
    """
    版式规划器：预测答案的结构和布局
    """
    def __init__(self):
        self.layout_patterns = {
            'table': ['表格', '表', '列表', '清单', '统计', '数据'],
            'list': ['列表', '清单', '项目', '条目', '要点'],
            'paragraph': ['段落', '描述', '说明', '介绍', '详情'],
            'chart': ['图表', '图', '统计图', '柱状图', '折线图'],
            'form': ['表单', '表格', '申请', '登记']
        }

    def predict_layout(self, query: str, evidence_cards: EvidenceCardCollection) -> Dict[str, Any]:
        """预测答案的版式结构"""
        layout_prediction = {
            'primary_layout': 'paragraph',
            'confidence': 0.5,
            'suggested_structure': [],
            'evidence_distribution': {}
        }
        
        # 基于查询内容预测
        query_layout_score = self._analyze_query_layout(query)
        
        # 基于证据内容预测
        evidence_layout_score = self._analyze_evidence_layout(evidence_cards)
        
        # 综合预测
        combined_score = {}
        for layout_type in self.layout_patterns.keys():
            combined_score[layout_type] = (
                query_layout_score.get(layout_type, 0) * 0.4 +
                evidence_layout_score.get(layout_type, 0) * 0.6
            )
        
        # 选择最高分的版式
        if combined_score:
            primary_layout = max(combined_score, key=combined_score.get)
            layout_prediction['primary_layout'] = primary_layout
            layout_prediction['confidence'] = combined_score[primary_layout]
        
        # 生成建议结构
        layout_prediction['suggested_structure'] = self._generate_structure(
            primary_layout, evidence_cards
        )
        
        # 分析证据分布
        layout_prediction['evidence_distribution'] = self._analyze_evidence_distribution(
            evidence_cards
        )
        
        return layout_prediction

    def _analyze_query_layout(self, query: str) -> Dict[str, float]:
        """分析查询的版式倾向"""
        scores = {layout: 0.0 for layout in self.layout_patterns.keys()}
        
        for layout_type, keywords in self.layout_patterns.items():
            for keyword in keywords:
                if keyword in query:
                    scores[layout_type] += 0.3
        
        # 归一化
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            for layout_type in scores:
                scores[layout_type] /= max_score
        
        return scores

    def _analyze_evidence_layout(self, evidence_cards: EvidenceCardCollection) -> Dict[str, float]:
        """分析证据的版式倾向"""
        scores = {layout: 0.0 for layout in self.layout_patterns.keys()}
        
        for card in evidence_cards:
            text = card.ocr_text
            source_type = card.source_type
            
            # 基于来源类型
            if source_type == 'table':
                scores['table'] += 0.5
            elif source_type == 'chart':
                scores['chart'] += 0.5
            
            # 基于文本内容
            for layout_type, keywords in self.layout_patterns.items():
                for keyword in keywords:
                    if keyword in text:
                        scores[layout_type] += 0.2
        
        # 归一化
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            for layout_type in scores:
                scores[layout_type] /= max_score
        
        return scores

    def _generate_structure(self, layout_type: str, 
                          evidence_cards: EvidenceCardCollection) -> List[str]:
        """生成建议的结构"""
        structures = {
            'table': ['标题', '表头', '数据行', '总计'],
            'list': ['标题', '列表项', '总结'],
            'paragraph': ['引言', '主体内容', '结论'],
            'chart': ['图表标题', '数据说明', '趋势分析'],
            'form': ['表单标题', '字段', '提交说明']
        }
        
        return structures.get(layout_type, ['内容'])

    def _analyze_evidence_distribution(self, evidence_cards: EvidenceCardCollection) -> Dict[str, Any]:
        """分析证据分布"""
        distribution = {
            'total_evidence': len(evidence_cards),
            'by_type': {},
            'by_page': {},
            'by_importance': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        for card in evidence_cards:
            # 按类型统计
            source_type = card.source_type
            distribution['by_type'][source_type] = distribution['by_type'].get(source_type, 0) + 1
            
            # 按页面统计
            page_id = card.page_id
            distribution['by_page'][page_id] = distribution['by_page'].get(page_id, 0) + 1
            
            # 按重要性统计
            if card.importance_score >= 0.7:
                distribution['by_importance']['high'] += 1
            elif card.importance_score >= 0.4:
                distribution['by_importance']['medium'] += 1
            else:
                distribution['by_importance']['low'] += 1
        
        return distribution
