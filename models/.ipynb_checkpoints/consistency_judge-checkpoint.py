import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re
import numpy as np
from typing import Dict, List, Tuple, Optional

class ConsistencyJudge:
    """
    增强版一致性判别器：检查生成答案与检索证据是否一致
    支持实体对齐、数值验证和动态阈值调整
    """
    def __init__(self, model_name="bert-base-chinese", threshold=0.4):
        self.threshold = threshold
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # 数值模式匹配
        self.number_patterns = [
            r'\d+\.?\d*%?',  # 百分比
            r'\d+\.?\d*',    # 普通数字
            r'[一二三四五六七八九十百千万亿]+',  # 中文数字
        ]
        
        # 实体类型关键词
        self.entity_keywords = {
            'person': ['人', '员', '者', '师', '家', '长', '经理', '主任', '总监'],
            'organization': ['公司', '企业', '集团', '机构', '部门', '单位', '学校', '医院'],
            'location': ['省', '市', '县', '区', '街道', '路', '号', '地址'],
            'time': ['年', '月', '日', '时', '分', '秒', '天', '周', '月', '季度'],
            'money': ['元', '万元', '亿元', '美元', '欧元', '人民币', '￥', '$'],
        }

    def _load_model(self):
        """加载BERT模型用于语义相似度计算"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            print(f"✅ 成功加载一致性判别模型: {self.model_name}")
        except Exception as e:
            print(f"⚠️ 无法加载BERT模型，将使用基础方法: {e}")
            self.tokenizer = None
            self.model = None

    def check(self, answer: str, evidence) -> Dict[str, any]:
        """
        综合一致性检查
        支持EvidenceCardCollection和字符串两种输入格式
        """
        result = {
            'overall_score': 0.0,
            'semantic_similarity': 0.0,
            'entity_alignment': 0.0,
            'number_consistency': 0.0,
            'keyword_overlap': 0.0,
            'tool_agreement': 0.0,  # 新增：工具一致性分
            'is_consistent': False,
            'issues': [],
            'suggestions': []
        }
        
        # 处理输入格式
        if hasattr(evidence, 'get_all_cards'):  # EvidenceCardCollection
            evidence_cards = evidence
            evidence_text = self._extract_evidence_text(evidence_cards)
            tool_cards = self._extract_tool_cards(evidence_cards)
        else:  # 字符串
            evidence_text = str(evidence)
            tool_cards = []
        
        # 1. 语义相似度检查
        result['semantic_similarity'] = self._semantic_similarity(answer, evidence_text)
        
        # 2. 实体对齐检查
        result['entity_alignment'] = self._entity_alignment_check(answer, evidence_text)
        
        # 3. 数值一致性检查
        result['number_consistency'] = self._number_consistency_check(answer, evidence_text)
        
        # 4. 关键词重叠检查
        result['keyword_overlap'] = self._keyword_overlap(answer, evidence_text)
        
        # 5. 工具一致性检查（新增）
        result['tool_agreement'] = self._tool_agreement_check(answer, tool_cards)
        
        # 6. 综合评分
        result['overall_score'] = self._calculate_overall_score(result)
        result['is_consistent'] = result['overall_score'] >= self.threshold
        
        # 7. 问题诊断和建议
        result['issues'], result['suggestions'] = self._diagnose_issues(result)
        
        # 8. 添加调试信息
        print(f"🔍 一致性检查详情:")
        print(f"  答案: {answer[:100]}...")
        print(f"  证据长度: {len(evidence_text)} 字符")
        print(f"  总体分数: {result['overall_score']:.3f} (阈值: {self.threshold})")
        print(f"  各组件分数: 语义={result['semantic_similarity']:.3f}, 实体={result['entity_alignment']:.3f}, 数值={result['number_consistency']:.3f}, 关键词={result['keyword_overlap']:.3f}, 工具={result['tool_agreement']:.3f}")
        print(f"  是否一致: {'✅' if result['is_consistent'] else '❌'}")
        
        return result

    def _semantic_similarity(self, answer: str, evidence: str) -> float:
        """计算语义相似度"""
        if self.model is None or self.tokenizer is None:
            # 回退到基础方法
            return self._basic_similarity(answer, evidence)
        
        try:
            # 编码文本 - 修复输入格式
            inputs = self.tokenizer(
                text=[answer, evidence],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                
                # 计算余弦相似度
                similarity = F.cosine_similarity(embeddings[0:1], embeddings[1:2])
                return similarity.item()
                
        except Exception as e:
            print(f"语义相似度计算失败: {e}")
            return self._basic_similarity(answer, evidence)

    def _basic_similarity(self, answer: str, evidence) -> float:
        """基础相似度计算（Jaccard相似度）"""
        # 确保evidence是字符串
        if not isinstance(evidence, str):
            evidence = str(evidence)
        
        answer_tokens = set(answer.split())
        evidence_tokens = set(evidence.split())
        intersection = len(answer_tokens & evidence_tokens)
        union = len(answer_tokens | evidence_tokens)
        return intersection / union if union > 0 else 0.0

    def _entity_alignment_check(self, answer: str, evidence: str) -> float:
        """实体对齐检查"""
        answer_entities = self._extract_entities(answer)
        evidence_entities = self._extract_entities(evidence)
        
        if not answer_entities or not evidence_entities:
            return 0.5  # 中性评分
        
        # 计算实体匹配度
        matched_entities = 0
        total_entities = len(answer_entities)
        
        for entity_type, entities in answer_entities.items():
            if entity_type in evidence_entities:
                for entity in entities:
                    if any(self._fuzzy_match(entity, ev_entity) for ev_entity in evidence_entities[entity_type]):
                        matched_entities += 1
        
        return matched_entities / total_entities if total_entities > 0 else 0.0

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取文本中的实体"""
        entities = {}
        
        for entity_type, keywords in self.entity_keywords.items():
            found_entities = []
            for keyword in keywords:
                # 简单的关键词匹配
                if keyword in text:
                    # 提取包含关键词的短语
                    pattern = rf'[^。！？]*{keyword}[^。！？]*'
                    matches = re.findall(pattern, text)
                    found_entities.extend(matches)
            
            if found_entities:
                entities[entity_type] = list(set(found_entities))
        
        return entities

    def _fuzzy_match(self, entity1: str, entity2: str) -> bool:
        """模糊匹配两个实体"""
        # 简单的包含关系检查
        return entity1 in entity2 or entity2 in entity1

    def _number_consistency_check(self, answer: str, evidence: str) -> float:
        """数值一致性检查"""
        answer_numbers = self._extract_numbers(answer)
        evidence_numbers = self._extract_numbers(evidence)
        
        if not answer_numbers or not evidence_numbers:
            return 0.5  # 中性评分
        
        # 检查数值是否匹配
        matched_numbers = 0
        total_numbers = len(answer_numbers)
        
        for ans_num in answer_numbers:
            for ev_num in evidence_numbers:
                if self._number_match(ans_num, ev_num):
                    matched_numbers += 1
                    break
        
        return matched_numbers / total_numbers if total_numbers > 0 else 0.0

    def _extract_numbers(self, text: str) -> List[str]:
        """提取文本中的数值"""
        numbers = []
        for pattern in self.number_patterns:
            matches = re.findall(pattern, text)
            numbers.extend(matches)
        return list(set(numbers))

    def _number_match(self, num1: str, num2: str) -> bool:
        """检查两个数值是否匹配"""
        try:
            # 尝试转换为数字进行比较
            val1 = float(num1.replace('%', ''))
            val2 = float(num2.replace('%', ''))
            return abs(val1 - val2) < 0.01  # 允许小的误差
        except:
            # 如果无法转换，进行字符串比较
            return num1 == num2

    def _keyword_overlap(self, answer: str, evidence) -> float:
        """关键词重叠度检查"""
        # 确保evidence是字符串
        if not isinstance(evidence, str):
            evidence = str(evidence)
        
        # 如果答案或证据为空，返回0
        if not answer.strip() or not evidence.strip():
            return 0.0
        
        # 提取重要关键词（去除停用词）
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '而', '了', '着', '过', '这', '那', '你', '我', '他', '她', '它', '们', '个', '只', '条', '张', '本', '台', '部', '件', '项', '次', '回', '遍', '趟', '下', '上', '中', '外', '内', '前', '后', '左', '右', '东', '西', '南', '北'}
        
        # 分词并过滤
        answer_words = set(word for word in answer.split() if word not in stop_words and len(word) > 1)
        evidence_words = set(word for word in evidence.split() if word not in stop_words and len(word) > 1)
        
        # 如果两个集合都为空，返回0.5（中性评分）
        if not answer_words and not evidence_words:
            return 0.5
        
        # 计算Jaccard相似度
        intersection = len(answer_words & evidence_words)
        union = len(answer_words | evidence_words)
        
        # 如果交集不为空，给予奖励分数
        if intersection > 0:
            base_score = intersection / union
            # 给予额外奖励
            bonus = min(0.2, intersection * 0.1)
            return min(1.0, base_score + bonus)
        
        # 如果没有关键词重叠，检查是否有部分匹配
        partial_matches = 0
        for answer_word in answer_words:
            for evidence_word in evidence_words:
                if len(answer_word) > 2 and len(evidence_word) > 2:
                    # 检查部分匹配（包含关系）
                    if answer_word in evidence_word or evidence_word in answer_word:
                        partial_matches += 1
        
        if partial_matches > 0:
            return min(0.3, partial_matches * 0.1)
        
        return 0.0

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """计算综合评分"""
        weights = {
            'semantic_similarity': 0.3,
            'entity_alignment': 0.2,
            'number_consistency': 0.2,
            'keyword_overlap': 0.1,
            'tool_agreement': 0.2  # 新增工具一致性权重
        }
        
        overall_score = 0.0
        for key, weight in weights.items():
            overall_score += scores[key] * weight
        
        return overall_score

    def _extract_evidence_text(self, evidence_cards) -> str:
        """从EvidenceCardCollection提取文本"""
        texts = []
        for card in evidence_cards.get_all_cards():
            if card.source_type != 'tool':  # 排除工具结果
                texts.append(card.ocr_text)
        return " ".join(texts)
    
    def _extract_tool_cards(self, evidence_cards) -> List:
        """从EvidenceCardCollection提取工具卡片"""
        tool_cards = []
        for card in evidence_cards.get_all_cards():
            if card.source_type in ['ocr', 'table', 'formula']:
                tool_cards.append(card)
        return tool_cards
    
    def _tool_agreement_check(self, answer: str, tool_cards: List) -> float:
        """工具一致性检查"""
        if not tool_cards:
            return 0.6  # 提高默认分数
        
        agreement_scores = []
        
        for card in tool_cards:
            tool_type = card.source_type
            tool_text = card.ocr_text
            
            # 如果工具文本为空，给予中等分数
            if not tool_text or not tool_text.strip():
                agreement_scores.append(0.6)
                continue
            
            if tool_type == 'ocr':
                # 印章文本匹配
                score = self._seal_text_match(answer, tool_text)
            elif tool_type == 'table':
                # 表格内容匹配
                score = self._table_content_match(answer, tool_text)
            elif tool_type == 'formula':
                # 公式匹配
                score = self._formula_match(answer, tool_text)
            else:
                # 对于未知工具类型，使用语义相似度
                score = self._semantic_similarity(answer, tool_text)
            
            # 确保分数不为0
            if score < 0.1:
                score = 0.1
            
            agreement_scores.append(score)
        
        # 计算平均分数，但给予最低保证
        avg_score = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.6
        return max(0.3, avg_score)  # 最低保证0.3分
    
    def _seal_text_match(self, answer: str, seal_text: str) -> float:
        """印章文本匹配"""
        # 使用编辑距离计算相似度
        from difflib import SequenceMatcher
        
        # 提取答案中可能的印章文本
        answer_seal_candidates = self._extract_seal_candidates(answer)
        
        if not answer_seal_candidates:
            return 0.0
        
        # 计算最佳匹配分数
        best_score = 0.0
        for candidate in answer_seal_candidates:
            similarity = SequenceMatcher(None, candidate, seal_text).ratio()
            best_score = max(best_score, similarity)
        
        return best_score
    
    def _table_content_match(self, answer: str, table_text) -> float:
        """表格内容匹配"""
        # 确保table_text是字符串
        if not isinstance(table_text, str):
            table_text = str(table_text)
        
        # 简单的关键词匹配
        table_keywords = set(table_text.split())
        answer_keywords = set(answer.split())
        
        intersection = len(table_keywords & answer_keywords)
        union = len(table_keywords | answer_keywords)
        
        return intersection / union if union > 0 else 0.0
    
    def _formula_match(self, answer: str, formula_text: str) -> float:
        """公式匹配"""
        # 提取LaTeX公式
        import re
        
        # 简单的LaTeX模式匹配
        latex_pattern = r'\\[a-zA-Z]+|\\[{}]|[a-zA-Z]+\^[a-zA-Z0-9]+|[a-zA-Z]+_[a-zA-Z0-9]+'
        
        answer_formulas = set(re.findall(latex_pattern, answer))
        tool_formulas = set(re.findall(latex_pattern, formula_text))
        
        if not answer_formulas and not tool_formulas:
            return 0.5  # 都没有公式，中性评分
        
        intersection = len(answer_formulas & tool_formulas)
        union = len(answer_formulas | tool_formulas)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_seal_candidates(self, text: str) -> List[str]:
        """提取可能的印章文本候选"""
        # 简单的启发式规则
        candidates = []
        
        # 查找可能的印章文本模式
        import re
        
        # 查找短文本（可能是印章）
        short_texts = re.findall(r'[^\s]{2,8}', text)
        candidates.extend(short_texts)
        
        # 查找包含特定关键词的文本
        seal_keywords = ['章', '印', '签', '名', '公司', '机构']
        for keyword in seal_keywords:
            if keyword in text:
                # 提取包含关键词的短语
                pattern = rf'[^。！？]*{keyword}[^。！？]*'
                matches = re.findall(pattern, text)
                candidates.extend(matches)
        
        return list(set(candidates))
    
    def _diagnose_issues(self, scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """诊断问题并提供建议"""
        issues = []
        suggestions = []
        
        if scores['semantic_similarity'] < 0.5:
            issues.append("语义相似度较低")
            suggestions.append("建议重新检索相关证据或调整答案表述")
        
        if scores['entity_alignment'] < 0.3:
            issues.append("实体对齐度不足")
            suggestions.append("检查答案中的实体是否与证据一致")
        
        if scores['number_consistency'] < 0.3:
            issues.append("数值一致性较差")
            suggestions.append("验证答案中的数值是否准确")
        
        if scores['keyword_overlap'] < 0.2:
            issues.append("关键词重叠度低")
            suggestions.append("答案可能偏离了证据的核心内容")
        
        if scores['tool_agreement'] < 0.3:
            issues.append("与专家工具结果不一致")
            suggestions.append("检查答案是否与专家工具的输出一致")
        
        return issues, suggestions

    def adjust_threshold(self, new_threshold: float):
        """动态调整一致性阈值"""
        self.threshold = max(0.1, min(0.9, new_threshold))
        print(f"一致性阈值已调整为: {self.threshold}")

    def get_consistency_report(self, answer: str, evidence: str) -> str:
        """生成一致性报告"""
        result = self.check(answer, evidence)
        
        report = f"""
一致性检查报告
================
总体评分: {result['overall_score']:.3f} ({'通过' if result['is_consistent'] else '未通过'})
语义相似度: {result['semantic_similarity']:.3f}
实体对齐度: {result['entity_alignment']:.3f}
数值一致性: {result['number_consistency']:.3f}
关键词重叠: {result['keyword_overlap']:.3f}

问题诊断:
"""
        
        if result['issues']:
            for issue in result['issues']:
                report += f"- {issue}\n"
        else:
            report += "- 无明显问题\n"
        
        if result['suggestions']:
            report += "\n改进建议:\n"
            for suggestion in result['suggestions']:
                report += f"- {suggestion}\n"
        
        return report
