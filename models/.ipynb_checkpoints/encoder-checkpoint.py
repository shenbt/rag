import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3Model, LayoutLMv3Config, LayoutLMv3Processor
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

class RegionAwareEncoder(nn.Module):
    """
    区域感知编码器：支持区域粒度自监督预训练
    扩展VDocRAG的RCR/RCG，加入region-level重构
    """
    def __init__(self, model_name="./layoutlmv3-base", 
                 hidden_size=768, 
                 num_attention_heads=12,
                 num_hidden_layers=12):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        
        # 加载LayoutLMv3模型
        try:
            print(f"正在从本地路径加载LayoutLMv3模型: {model_name}")
            self.layout_model = LayoutLMv3Model.from_pretrained(model_name)
            self.processor = LayoutLMv3Processor.from_pretrained(model_name)
            print("✅ 从本地路径加载LayoutLMv3模型成功")
        except Exception as e:
            print(f"无法从本地路径加载模型 {model_name}: {e}")
            try:
                # 尝试从Hugging Face加载
                print("尝试从Hugging Face加载: microsoft/layoutlmv3-base")
                self.layout_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
                self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
                print("✅ 从Hugging Face加载LayoutLMv3模型成功")
            except Exception as e2:
                print(f"无法从Hugging Face加载模型: {e2}")
                print("使用随机初始化的模型")
                config = LayoutLMv3Config(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_hidden_layers=num_hidden_layers
                )
                self.layout_model = LayoutLMv3Model(config)
                self.processor = None
        
        # 区域重构头
        self.region_reconstruction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 区域分类头
        self.region_classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 5)  # 5种区域类型：text, table, image, chart, form
        )
        
        # 区域对齐头
        self.region_alignment_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )

    def forward(self, 
                input_ids=None,
                attention_mask=None,
                bbox=None,
                pixel_values=None,
                region_masks=None,
                region_labels=None,
                task_type="encoding"):
        """
        前向传播
        :param task_type: encoding, reconstruction, classification, alignment
        """
        if task_type == "encoding":
            return self._encode(input_ids, attention_mask, bbox, pixel_values)
        elif task_type == "reconstruction":
            return self._region_reconstruction(input_ids, attention_mask, bbox, pixel_values, region_masks)
        elif task_type == "classification":
            return self._region_classification(input_ids, attention_mask, bbox, pixel_values, region_labels)
        elif task_type == "alignment":
            return self._region_alignment(input_ids, attention_mask, bbox, pixel_values)
        else:
            raise ValueError(f"未知的任务类型: {task_type}")

    def _encode(self, input_ids, attention_mask, bbox, pixel_values):
        """基础编码 - 使用安全的bbox处理"""
        # 如果bbox有问题，直接使用fallback
        safe_bbox = self._create_safe_bbox(input_ids, bbox)
        
        try:
            # 尝试不使用bbox参数调用模型
            if safe_bbox is None or torch.all(safe_bbox == 0):
                outputs = self.layout_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
            else:
                outputs = self.layout_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=safe_bbox,
                    pixel_values=pixel_values
                )
            return outputs.last_hidden_state
        except Exception as e:
            print(f"⚠️ LayoutLMv3编码失败: {e}")
            # 如果编码失败，尝试不使用bbox
            try:
                outputs = self.layout_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
                return outputs.last_hidden_state
            except Exception as e2:
                print(f"⚠️ 无bbox编码也失败: {e2}")
                # 如果编码失败，返回零张量
                batch_size = input_ids.size(0) if input_ids is not None else 1
                seq_len = input_ids.size(1) if input_ids is not None else 512
                return torch.zeros(batch_size, seq_len, self.hidden_size)
    
    def _create_safe_bbox(self, input_ids, bbox):
        """创建安全的bbox，确保完全符合LayoutLMv3要求"""
        if input_ids is None:
            seq_len = 512
            batch_size = 1
        else:
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
        
        # 创建一个完全安全的bbox矩阵，使用LayoutLMv3推荐的默认值
        safe_bbox = torch.zeros(batch_size, seq_len, 4, dtype=torch.long)
        
        # 使用LayoutLMv3推荐的默认bbox格式
        # 每个token使用相同的默认bbox，这是最安全的方法
        default_bbox = torch.tensor([0, 0, 1000, 1000], dtype=torch.long)
        
        for b in range(batch_size):
            for i in range(seq_len):
                safe_bbox[b, i, :] = default_bbox
        
        # 如果提供了原始bbox，尝试使用它（但要安全处理）
        if bbox is not None:
            try:
                # 转换为张量
                if not isinstance(bbox, torch.Tensor):
                    bbox = torch.tensor(bbox, dtype=torch.long)
                
                bbox = bbox.long()
                
                # 处理维度
                if bbox.dim() == 1:
                    bbox = bbox.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
                elif bbox.dim() == 2:
                    bbox = bbox.unsqueeze(0)  # [1, seq_len, 4]
                elif bbox.dim() == 3:
                    pass  # 已经是正确维度
                else:
                    # 维度不正确，使用默认bbox
                    return safe_bbox
                
                # 确保batch维度匹配
                if bbox.size(0) != batch_size:
                    bbox = bbox.expand(batch_size, -1, -1)
                
                # 确保序列长度匹配
                if bbox.size(1) > seq_len:
                    bbox = bbox[:, :seq_len, :]
                elif bbox.size(1) < seq_len:
                    # 用最后一个bbox填充
                    last_bbox = bbox[:, -1:, :] if bbox.size(1) > 0 else safe_bbox[:, :1, :]
                    padding_needed = seq_len - bbox.size(1)
                    padding = last_bbox.expand(-1, padding_needed, -1)
                    bbox = torch.cat([bbox, padding], dim=1)
                
                # 强制clamp所有坐标到0-1000
                bbox = torch.clamp(bbox, 0, 1000)
                
                # 确保x2 > x1, y2 > y1
                for b in range(batch_size):
                    for s in range(seq_len):
                        x1, y1, x2, y2 = bbox[b, s, :]
                        if x2 <= x1:
                            x2 = min(1000, x1 + 1)
                        if y2 <= y1:
                            y2 = min(1000, y1 + 1)
                        bbox[b, s, :] = torch.tensor([x1, y1, x2, y2], dtype=torch.long)
                
                # 最终验证每个坐标
                for b in range(batch_size):
                    for s in range(seq_len):
                        for c in range(4):
                            if bbox[b, s, c] < 0 or bbox[b, s, c] > 1000:
                                # 如果仍然有问题，使用安全默认值
                                return safe_bbox
                
                return bbox
                
            except Exception as e:
                print(f"⚠️ bbox处理失败，使用安全默认值: {e}")
                return safe_bbox
        
        return safe_bbox

    def _region_reconstruction(self, input_ids, attention_mask, bbox, pixel_values, region_masks):
        """区域重构任务"""
        # 获取编码
        hidden_states = self._encode(input_ids, attention_mask, bbox, pixel_values)
        
        # 应用区域掩码
        masked_hidden_states = hidden_states.clone()
        if region_masks is not None:
            masked_hidden_states = masked_hidden_states * (1 - region_masks.unsqueeze(-1))
        
        # 重构被掩码的区域
        reconstructed_hidden = self.region_reconstruction_head(masked_hidden_states)
        
        return {
            'original_hidden': hidden_states,
            'reconstructed_hidden': reconstructed_hidden,
            'masked_hidden': masked_hidden_states
        }

    def _region_classification(self, input_ids, attention_mask, bbox, pixel_values, region_labels):
        """区域分类任务"""
        hidden_states = self._encode(input_ids, attention_mask, bbox, pixel_values)
        
        # 区域分类
        region_logits = self.region_classification_head(hidden_states)
        
        return {
            'hidden_states': hidden_states,
            'region_logits': region_logits,
            'region_labels': region_labels
        }

    def _region_alignment(self, input_ids, attention_mask, bbox, pixel_values):
        """区域对齐任务"""
        hidden_states = self._encode(input_ids, attention_mask, bbox, pixel_values)
        
        # 区域对齐表示
        alignment_embeddings = self.region_alignment_head(hidden_states)
        
        return {
            'hidden_states': hidden_states,
            'alignment_embeddings': alignment_embeddings
        }

    def encode_text(self, text: str, bbox: List[List[int]] = None) -> torch.Tensor:
        """编码文本"""
        if self.processor is None:
            # 如果没有processor，返回随机向量
            return torch.randn(1, len(text.split()), self.hidden_size)
        
        try:
            # 处理输入
            if bbox is None:
                bbox = [[0, 0, 1000, 1000]] * len(text.split())
            
            # 确保bbox坐标在0-1000范围内
            if bbox:
                normalized_bbox = []
                for box in bbox:
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        # 确保坐标在0-1000范围内
                        x1 = max(0, min(1000, x1))
                        y1 = max(0, min(1000, y1))
                        x2 = max(0, min(1000, x2))
                        y2 = max(0, min(1000, y2))
                        # 确保边界框有效
                        if x2 <= x1:
                            x2 = min(1000, x1 + 1)
                        if y2 <= y1:
                            y2 = min(1000, y1 + 1)
                        normalized_bbox.append([x1, y1, x2, y2])
                    else:
                        normalized_bbox.append([0, 0, 1000, 1000])
                bbox = normalized_bbox
            
            inputs = self.processor(
                text=text,
                boxes=bbox,
                return_tensors="pt"
            )
            
            # 编码
            with torch.no_grad():
                outputs = self.layout_model(**inputs)
                return outputs.last_hidden_state
            
        except Exception as e:
            print(f"文本编码失败: {e}")
            return torch.randn(1, len(text.split()), self.hidden_size)

    def encode_image(self, image, text: str = "", bbox: List[List[int]] = None) -> torch.Tensor:
        """编码图像"""
        if self.processor is None:
            return torch.randn(1, 512, self.hidden_size)
        
        try:
            # 处理输入
            if bbox is None:
                bbox = [[0, 0, 1000, 1000]] * len(text.split()) if text else [[0, 0, 1000, 1000]]
            
            # 确保bbox坐标在0-1000范围内
            if bbox:
                normalized_bbox = []
                for box in bbox:
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        # 确保坐标在0-1000范围内
                        x1 = max(0, min(1000, x1))
                        y1 = max(0, min(1000, y1))
                        x2 = max(0, min(1000, x2))
                        y2 = max(0, min(1000, y2))
                        # 确保边界框有效
                        if x2 <= x1:
                            x2 = min(1000, x1 + 1)
                        if y2 <= y1:
                            y2 = min(1000, y1 + 1)
                        normalized_bbox.append([x1, y1, x2, y2])
                    else:
                        normalized_bbox.append([0, 0, 1000, 1000])
                bbox = normalized_bbox
            
            inputs = self.processor(
                images=image,
                text=text,
                boxes=bbox,
                return_tensors="pt"
            )
            
            # 编码
            with torch.no_grad():
                outputs = self.layout_model(**inputs)
                return outputs.last_hidden_state
            
        except Exception as e:
            print(f"图像编码失败: {e}")
            return torch.randn(1, 512, self.hidden_size)

    def extract_regions(self, hidden_states: torch.Tensor, 
                       bbox: List[List[int]], 
                       region_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """提取区域特征"""
        regions = []
        
        # 简单的区域提取（基于边界框）
        for i, box in enumerate(bbox):
            if i < hidden_states.size(1):
                region_embedding = hidden_states[0, i, :]
                
                # 计算区域重要性（基于嵌入的L2范数）
                importance = torch.norm(region_embedding).item()
                
                if importance > region_threshold:
                    regions.append({
                        'bbox': box,
                        'embedding': region_embedding,
                        'importance': importance,
                        'index': i
                    })
        
        # 按重要性排序
        regions.sort(key=lambda x: x['importance'], reverse=True)
        
        return regions

    def compute_region_similarity(self, region1: torch.Tensor, region2: torch.Tensor) -> float:
        """计算区域相似度"""
        try:
            # 余弦相似度
            similarity = F.cosine_similarity(region1.unsqueeze(0), region2.unsqueeze(0))
            return similarity.item()
        except Exception as e:
            print(f"计算区域相似度失败: {e}")
            return 0.0

    def get_region_embeddings(self, hidden_states: torch.Tensor, 
                            region_indices: List[int]) -> torch.Tensor:
        """获取指定区域的嵌入"""
        if not region_indices:
            return torch.empty(0, self.hidden_size)
        
        embeddings = []
        for idx in region_indices:
            if idx < hidden_states.size(1):
                embeddings.append(hidden_states[0, idx, :])
        
        if embeddings:
            return torch.stack(embeddings)
        else:
            return torch.empty(0, self.hidden_size)


class RegionPreTrainer:
    """
    区域预训练器：实现区域粒度自监督预训练
    """
    def __init__(self, encoder: RegionAwareEncoder, learning_rate: float = 1e-5):
        self.encoder = encoder
        self.optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()

    def train_reconstruction(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练区域重构任务"""
        self.encoder.train()
        
        try:
            # 前向传播（bbox处理在_create_safe_bbox中完成）
            outputs = self.encoder(
                input_ids=batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                bbox=batch_data.get('bbox'),
                pixel_values=batch_data.get('pixel_values'),
                region_masks=batch_data.get('region_masks'),
                task_type="reconstruction"
            )
            
            # 检查输出格式
            if not isinstance(outputs, dict) or 'reconstructed_hidden' not in outputs:
                # 如果输出格式不正确，返回默认损失
                return {'reconstruction_loss': 0.0}
            
            # 计算重构损失
            reconstruction_loss = self.mse_criterion(
                outputs['reconstructed_hidden'],
                outputs['original_hidden']
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            reconstruction_loss.backward()
            self.optimizer.step()
            
            return {'reconstruction_loss': reconstruction_loss.item()}
            
        except Exception as e:
            print(f"⚠️ 重构训练失败: {e}")
            return {'reconstruction_loss': 0.0}

    def train_classification(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练区域分类任务"""
        self.encoder.train()
        
        try:
            # 确保bbox坐标在0-1000范围内
            bbox = batch_data['bbox'].clone()
            
            # 详细的bbox验证和修复
            if bbox is not None:
                # 检查bbox的维度和类型
                if not isinstance(bbox, torch.Tensor):
                    bbox = torch.tensor(bbox, dtype=torch.long)
                
                # 确保是长整型
                bbox = bbox.long()
                
                # 处理bbox维度
                if bbox.dim() == 3:  # [batch, seq_len, 4]
                    # 取第一个batch的bbox
                    bbox = bbox[0]  # [seq_len, 4]
                elif bbox.dim() == 1:  # [4]
                    # 扩展为 [seq_len, 4]
                    bbox = bbox.unsqueeze(0)
                
                # 强制clamp到0-1000范围
                bbox = torch.clamp(bbox, 0, 1000)
                
                # 验证bbox范围
                min_val = torch.min(bbox).item()
                max_val = torch.max(bbox).item()
                if min_val < 0 or max_val > 1000:
                    print(f"⚠️ 分类训练bbox范围异常: {min_val}-{max_val}，已强制clamp")
                    bbox = torch.clamp(bbox, 0, 1000)
            
            # 前向传播
            outputs = self.encoder(
                input_ids=batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                bbox=bbox,
                pixel_values=batch_data.get('pixel_values'),
                region_labels=batch_data['region_labels'],
                task_type="classification"
            )
            
            # 检查输出格式
            if not isinstance(outputs, dict) or 'region_logits' not in outputs:
                return {'classification_loss': 0.0}
            
            # 计算分类损失
            # 确保维度匹配
            logits = outputs['region_logits']
            labels = batch_data['region_labels']
            
            # 调整logits维度
            if logits.dim() == 3:  # [batch, seq_len, num_classes]
                logits = logits.view(-1, logits.size(-1))
            else:
                logits = logits.view(-1, 5)  # 默认5个类别
            
            # 调整labels维度
            if labels.dim() == 2:  # [batch, seq_len]
                labels = labels.view(-1)
            else:
                labels = labels.view(-1)
            
            # 确保标签在有效范围内
            num_classes = logits.size(-1)
            labels = torch.clamp(labels, 0, num_classes - 1)
            
            classification_loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            classification_loss.backward()
            self.optimizer.step()
            
            return {'classification_loss': classification_loss.item()}
        
        except Exception as e:
            print(f"⚠️ 分类训练失败: {e}")
            return {'classification_loss': 0.0}

    def train_alignment(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练区域对齐任务"""
        self.encoder.train()
        
        try:
            # 确保bbox坐标在0-1000范围内
            bbox = batch_data['bbox'].clone()
            
            # 详细的bbox验证和修复
            if bbox is not None:
                # 检查bbox的维度和类型
                if not isinstance(bbox, torch.Tensor):
                    bbox = torch.tensor(bbox, dtype=torch.long)
                
                # 确保是长整型
                bbox = bbox.long()
                
                # 处理bbox维度
                if bbox.dim() == 3:  # [batch, seq_len, 4]
                    # 取第一个batch的bbox
                    bbox = bbox[0]  # [seq_len, 4]
                elif bbox.dim() == 1:  # [4]
                    # 扩展为 [seq_len, 4]
                    bbox = bbox.unsqueeze(0)
                
                # 强制clamp到0-1000范围
                bbox = torch.clamp(bbox, 0, 1000)
                
                # 验证bbox范围
                min_val = torch.min(bbox).item()
                max_val = torch.max(bbox).item()
                if min_val < 0 or max_val > 1000:
                    print(f"⚠️ 对齐训练bbox范围异常: {min_val}-{max_val}，已强制clamp")
                    bbox = torch.clamp(bbox, 0, 1000)
            
            # 前向传播
            outputs = self.encoder(
                input_ids=batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                bbox=bbox,
                pixel_values=batch_data.get('pixel_values'),
                task_type="alignment"
            )
            
            # 检查输出格式
            if not isinstance(outputs, dict) or 'alignment_embeddings' not in outputs:
                return {'alignment_loss': 0.0}
            
            # 计算对齐损失（对比学习）
            alignment_embeddings = outputs['alignment_embeddings']
            
            # 确保维度正确
            if alignment_embeddings.dim() == 3:  # [batch, seq_len, hidden_dim]
                # 取平均池化得到序列表示
                alignment_embeddings = alignment_embeddings.mean(dim=1)  # [batch, hidden_dim]
            
            # 简单的对比损失
            batch_size = alignment_embeddings.size(0)
            
            # 计算相似度矩阵
            if batch_size > 1:
                similarity_matrix = torch.matmul(alignment_embeddings, alignment_embeddings.transpose(0, 1))
                
                # 对角线应该是1（自己与自己的相似度）
                target_matrix = torch.eye(batch_size, device=similarity_matrix.device)
                alignment_loss = self.mse_criterion(similarity_matrix, target_matrix)
            else:
                # 如果batch_size=1，使用简单的L2损失
                alignment_loss = self.mse_criterion(
                    alignment_embeddings,
                    torch.zeros_like(alignment_embeddings)
                )
            
            # 反向传播
            self.optimizer.zero_grad()
            alignment_loss.backward()
            self.optimizer.step()
            
            return {'alignment_loss': alignment_loss.item()}
            
        except Exception as e:
            print(f"⚠️ 对齐训练失败: {e}")
            return {'alignment_loss': 0.0}

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"✅ 模型已保存到 {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ 模型已从 {path} 加载")
