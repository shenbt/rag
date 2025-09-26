import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

class ConsistencyLoss(torch.nn.Module):
    """
    答案-证据一致性校验损失：确保生成的答案与所选证据一致
    """
    def __init__(self, threshold=0.5, use_tfidf=True):
        super(ConsistencyLoss, self).__init__()
        self.threshold = threshold
        self.use_tfidf = use_tfidf
        if use_tfidf:
            # 配置TF-IDF向量器，支持中文
            self.vectorizer = TfidfVectorizer(
                max_features=1000, 
                stop_words=None,  # 不使用英文停用词
                token_pattern=r'(?u)\b\w+\b'  # 支持Unicode字符
            )

    def forward(self, generated_answer, evidence):
        """
        计算一致性损失
        :param generated_answer: 生成的答案
        :param evidence: 相关证据（例如，OCR文本）
        :return: 一致性损失
        """
        # 确保输入是字符串
        if not isinstance(generated_answer, str):
            generated_answer = str(generated_answer)
        if not isinstance(evidence, str):
            evidence = str(evidence)
        
        try:
            # 清理文本
            generated_answer = self._clean_text(generated_answer)
            evidence = self._clean_text(evidence)
            
            if self.use_tfidf and len(generated_answer.strip()) > 0 and len(evidence.strip()) > 0:
                # 使用TF-IDF计算语义相似度
                try:
                    # 准备文本
                    texts = [generated_answer, evidence]
                    
                    # 计算TF-IDF向量
                    tfidf_matrix = self.vectorizer.fit_transform(texts)
                    
                    # 计算余弦相似度
                    similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
                    
                    # 转换为一致性得分
                    consistency_score = max(0, similarity)
                    
                except Exception:
                    # 如果TF-IDF失败，回退到词级匹配
                    consistency_score = self._word_level_similarity(generated_answer, evidence)
            else:
                # 使用词级匹配
                consistency_score = self._word_level_similarity(generated_answer, evidence)
            
            # 损失为 1 - 一致性得分，确保模型生成的答案和证据尽可能一致
            loss_value = 1 - consistency_score
            
            # 使用relu并确保返回的是张量
            loss = F.relu(torch.tensor(loss_value, dtype=torch.float32))
            
            # 确保损失在正确的设备上
            if hasattr(self, 'device'):
                loss = loss.to(self.device)
            
            return loss
            
        except Exception as e:
            print(f"计算一致性损失时出错: {e}")
            # 返回默认损失
            return torch.tensor(0.5, dtype=torch.float32)

    def _word_level_similarity(self, text1, text2):
        """
        计算词级相似度（支持中文）
        """
        # 中文分词（简单按字符分割）
        def tokenize_chinese(text):
            # 移除标点符号，按字符分割
            text = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
            return list(text)
        
        # 获取tokens
        tokens1 = set(tokenize_chinese(text1.lower()))
        tokens2 = set(tokenize_chinese(text2.lower()))
        
        # 计算Jaccard相似度
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0

    def _clean_text(self, text):
        """
        清理文本，移除乱码和无效字符
        """
        # 移除控制字符和无效Unicode
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # 移除多余的标点符号，保留中文标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:，。！？；：]', '', text)
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
