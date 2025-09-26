import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    RCR/RCG 对比损失：用于训练视觉-语言模型
    """
    def __init__(self, margin=0.1, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, query_embedding, positive_embedding, negative_embedding=None):
        """
        计算对比损失
        :param query_embedding: 查询的嵌入
        :param positive_embedding: 正样本的嵌入
        :param negative_embedding: 负样本的嵌入（可选）
        :return: 损失值
        """
        # 确保所有输入都是张量
        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        if not isinstance(positive_embedding, torch.Tensor):
            positive_embedding = torch.tensor(positive_embedding, dtype=torch.float32)
        
        # 确保设备一致
        device = query_embedding.device
        positive_embedding = positive_embedding.to(device)
        
        if negative_embedding is None:
            # 如果没有负样本，使用硬负样本生成策略
            negative_embedding = self._generate_hard_negative(query_embedding, positive_embedding, device)
        else:
            if not isinstance(negative_embedding, torch.Tensor):
                negative_embedding = torch.tensor(negative_embedding, dtype=torch.float32)
            negative_embedding = negative_embedding.to(device)
        
        # 确保维度匹配
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if positive_embedding.dim() == 1:
            positive_embedding = positive_embedding.unsqueeze(0)
        if negative_embedding.dim() == 1:
            negative_embedding = negative_embedding.unsqueeze(0)
        
        # 计算相似度
        pos_sim = F.cosine_similarity(query_embedding, positive_embedding, dim=-1)
        neg_sim = F.cosine_similarity(query_embedding, negative_embedding, dim=-1)
        
        # 使用InfoNCE损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1) / self.temperature
        labels = torch.zeros(query_embedding.size(0), dtype=torch.long, device=device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss

    def _generate_hard_negative(self, query_embedding, positive_embedding, device):
        """
        生成硬负样本
        """
        # 策略1：添加噪声
        noise = torch.randn_like(positive_embedding, device=device) * 0.1
        hard_negative = positive_embedding + noise
        
        # 策略2：随机旋转
        if hard_negative.size(-1) >= 2:
            angle = torch.rand(1, device=device) * 2 * torch.pi
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)
            
            # 只对前两个维度进行旋转
            rotated = torch.matmul(hard_negative[..., :2], rotation_matrix)
            hard_negative[..., :2] = rotated
        
        return hard_negative
