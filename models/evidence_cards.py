import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib

class EvidenceCard:
    """
    增强版证据卡：保存检索到的证据，包括图像、OCR文本、位置信息和重要性评分
    支持证据溯源、重要性评估和可视化展示
    """
    def __init__(self, 
                 image=None, 
                 ocr_text: str = "", 
                 bbox: List[float] = None, 
                 page_id: str = "",
                 confidence: float = 1.0,
                 source_type: str = "text"):
        self.image = image
        self.ocr_text = ocr_text
        self.bbox = bbox or [0, 0, 0, 0]  # [x1, y1, x2, y2]
        self.page_id = page_id
        self.confidence = confidence
        self.source_type = source_type  # text, table, image, chart
        
        # 新增字段
        self.evidence_id = self._generate_id()
        self.created_at = datetime.now().isoformat()
        self.importance_score = 0.0
        self.relevance_score = 0.0
        self.entities = {}
        self.numbers = []
        self.keywords = []
        self.metadata = {}
        
        # 初始化证据分析
        self._analyze_evidence()

    def _generate_id(self) -> str:
        """生成唯一的证据ID"""
        content = f"{self.ocr_text}{self.page_id}{self.bbox}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def _analyze_evidence(self):
        """分析证据内容，提取实体、数值和关键词"""
        if not self.ocr_text:
            return
        
        # 提取数值
        import re
        number_patterns = [
            r'\d+\.?\d*%?',  # 百分比
            r'\d+\.?\d*',    # 普通数字
            r'[一二三四五六七八九十百千万亿]+',  # 中文数字
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, self.ocr_text)
            self.numbers.extend(matches)
        
        self.numbers = list(set(self.numbers))
        
        # 提取关键词（简单实现）
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '而', '了', '着', '过'}
        words = [word for word in self.ocr_text.split() 
                if word not in stop_words and len(word) > 1]
        self.keywords = list(set(words))[:10]  # 限制关键词数量
        
        # 提取实体（简单实现）
        entity_keywords = {
            'person': ['人', '员', '者', '师', '家', '长', '经理', '主任', '总监'],
            'organization': ['公司', '企业', '集团', '机构', '部门', '单位', '学校', '医院'],
            'location': ['省', '市', '县', '区', '街道', '路', '号', '地址'],
            'time': ['年', '月', '日', '时', '分', '秒', '天', '周', '月', '季度'],
            'money': ['元', '万元', '亿元', '美元', '欧元', '人民币', '￥', '$'],
        }
        
        for entity_type, keywords in entity_keywords.items():
            found_entities = []
            for keyword in keywords:
                if keyword in self.ocr_text:
                    pattern = rf'[^。！？]*{keyword}[^。！？]*'
                    matches = re.findall(pattern, self.ocr_text)
                    found_entities.extend(matches)
            
            if found_entities:
                self.entities[entity_type] = list(set(found_entities))

    def set_importance_score(self, score: float):
        """设置重要性评分"""
        self.importance_score = max(0.0, min(1.0, score))

    def set_relevance_score(self, score: float):
        """设置相关性评分"""
        self.relevance_score = max(0.0, min(1.0, score))

    def add_metadata(self, key: str, value: Any):
        """添加元数据"""
        self.metadata[key] = value
    
    def set_source(self, source: str):
        """设置证据来源"""
        self.metadata['source'] = source
    
    def set_field_map(self, field_map: Dict[str, Any]):
        """设置字段映射"""
        self.metadata['field_map'] = field_map
    
    def is_tool_result(self) -> bool:
        """判断是否为工具结果"""
        return self.source_type in ['ocr', 'table', 'formula'] or \
               self.metadata.get('source', '').startswith('tool:')
    
    def get_tool_name(self) -> str:
        """获取工具名称"""
        if self.is_tool_result():
            source = self.metadata.get('source', '')
            if source.startswith('tool:'):
                return source.replace('tool:', '')
            return self.source_type
        return 'retrieval'

    def get_bbox_area(self) -> float:
        """计算边界框面积"""
        if len(self.bbox) >= 4:
            width = self.bbox[2] - self.bbox[0]
            height = self.bbox[3] - self.bbox[1]
            return width * height
        return 0.0

    def get_center_point(self) -> Tuple[float, float]:
        """获取边界框中心点"""
        if len(self.bbox) >= 4:
            x = (self.bbox[0] + self.bbox[2]) / 2
            y = (self.bbox[1] + self.bbox[3]) / 2
            return (x, y)
        return (0.0, 0.0)

    def is_overlapping(self, other_card) -> bool:
        """检查是否与其他证据卡重叠"""
        if not isinstance(other_card, EvidenceCard):
            return False
        
        # 简单的重叠检测
        x1, y1, x2, y2 = self.bbox
        ox1, oy1, ox2, oy2 = other_card.bbox
        
        return not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "evidence_id": self.evidence_id,
            "ocr_text": self.ocr_text,
            "bbox": self.bbox,
            "page_id": self.page_id,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "created_at": self.created_at,
            "importance_score": self.importance_score,
            "relevance_score": self.relevance_score,
            "entities": self.entities,
            "numbers": self.numbers,
            "keywords": self.keywords,
            "metadata": self.metadata,
            "bbox_area": self.get_bbox_area(),
            "center_point": self.get_center_point()
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __str__(self) -> str:
        """字符串表示"""
        return f"EvidenceCard(id={self.evidence_id}, text='{self.ocr_text[:50]}...', score={self.importance_score:.3f})"

    def __repr__(self) -> str:
        return self.__str__()


class EvidenceCardCollection:
    """
    证据卡集合：管理多个证据卡，支持排序、过滤和聚合
    """
    def __init__(self):
        self.cards = []
        self.card_dict = {}  # 用于快速查找

    def add_card(self, card: EvidenceCard):
        """添加证据卡"""
        self.cards.append(card)
        self.card_dict[card.evidence_id] = card

    def remove_card(self, card_id: str):
        """移除证据卡"""
        if card_id in self.card_dict:
            card = self.card_dict.pop(card_id)
            self.cards.remove(card)

    def get_card(self, card_id: str) -> Optional[EvidenceCard]:
        """获取指定ID的证据卡"""
        return self.card_dict.get(card_id)

    def sort_by_importance(self, reverse: bool = True):
        """按重要性评分排序"""
        self.cards.sort(key=lambda x: x.importance_score, reverse=reverse)

    def sort_by_relevance(self, reverse: bool = True):
        """按相关性评分排序"""
        self.cards.sort(key=lambda x: x.relevance_score, reverse=reverse)

    def filter_by_type(self, source_type: str) -> List[EvidenceCard]:
        """按来源类型过滤"""
        return [card for card in self.cards if card.source_type == source_type]

    def filter_by_page(self, page_id: str) -> List[EvidenceCard]:
        """按页面ID过滤"""
        return [card for card in self.cards if card.page_id == page_id]

    def filter_by_confidence(self, min_confidence: float) -> List[EvidenceCard]:
        """按置信度过滤"""
        return [card for card in self.cards if card.confidence >= min_confidence]

    def get_top_cards(self, n: int = 5, sort_by: str = "importance") -> List[EvidenceCard]:
        """获取前N个证据卡"""
        if sort_by == "importance":
            self.sort_by_importance()
        elif sort_by == "relevance":
            self.sort_by_relevance()
        
        return self.cards[:n]

    def get_entities_summary(self) -> Dict[str, List[str]]:
        """获取所有实体的汇总"""
        all_entities = {}
        for card in self.cards:
            for entity_type, entities in card.entities.items():
                if entity_type not in all_entities:
                    all_entities[entity_type] = []
                all_entities[entity_type].extend(entities)
        
        # 去重
        for entity_type in all_entities:
            all_entities[entity_type] = list(set(all_entities[entity_type]))
        
        return all_entities

    def get_numbers_summary(self) -> List[str]:
        """获取所有数值的汇总"""
        all_numbers = []
        for card in self.cards:
            all_numbers.extend(card.numbers)
        return list(set(all_numbers))

    def get_keywords_summary(self) -> List[str]:
        """获取所有关键词的汇总"""
        all_keywords = []
        for card in self.cards:
            all_keywords.extend(card.keywords)
        return list(set(all_keywords))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_cards": len(self.cards),
            "cards": [card.to_dict() for card in self.cards],
            "entities_summary": self.get_entities_summary(),
            "numbers_summary": self.get_numbers_summary(),
            "keywords_summary": self.get_keywords_summary()
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self.cards)

    def get_all_cards(self) -> List[EvidenceCard]:
        """获取所有证据卡"""
        return self.cards
    
    def extend(self, other_collection):
        """扩展集合，添加另一个集合的所有卡片"""
        if hasattr(other_collection, 'get_all_cards'):
            self.cards.extend(other_collection.get_all_cards())
        elif hasattr(other_collection, 'cards'):
            self.cards.extend(other_collection.cards)
        else:
            self.cards.extend(other_collection)
    
    def __iter__(self):
        return iter(self.cards)
