import torch
import torch.nn.functional as F

class RegionLoss(torch.nn.Module):
    """
    区域粒度重建损失：用于区域级别的生成与重建
    """
    def __init__(self, lambda_reg=1.0):
        super(RegionLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, region_embeddings, reconstructed_embeddings):
        """
        计算区域重建损失
        :param region_embeddings: 真实区域的嵌入
        :param reconstructed_embeddings: 重建的区域嵌入
        :return: 重建损失
        """
        # 确保输入是张量
        if not isinstance(region_embeddings, torch.Tensor):
            region_embeddings = torch.tensor(region_embeddings, dtype=torch.float32)
        if not isinstance(reconstructed_embeddings, torch.Tensor):
            reconstructed_embeddings = torch.tensor(reconstructed_embeddings, dtype=torch.float32)
        
        # 确保设备一致
        device = region_embeddings.device
        reconstructed_embeddings = reconstructed_embeddings.to(device)
        
        # 确保维度匹配
        if region_embeddings.shape != reconstructed_embeddings.shape:
            # 如果维度不匹配，尝试调整
            if region_embeddings.dim() == 1 and reconstructed_embeddings.dim() == 2:
                # 如果region_embeddings是1D，reconstructed_embeddings是2D
                region_embeddings = region_embeddings.unsqueeze(0)
            elif region_embeddings.dim() == 2 and reconstructed_embeddings.dim() == 1:
                # 如果region_embeddings是2D，reconstructed_embeddings是1D
                reconstructed_embeddings = reconstructed_embeddings.unsqueeze(0)
            
            # 再次检查维度
            if region_embeddings.shape != reconstructed_embeddings.shape:
                min_dim = min(region_embeddings.shape[0], reconstructed_embeddings.shape[0])
                region_embeddings = region_embeddings[:min_dim]
                reconstructed_embeddings = reconstructed_embeddings[:min_dim]
        
        # L2 损失，确保区域嵌入与重建嵌入的差距最小
        loss = F.mse_loss(region_embeddings, reconstructed_embeddings)
        return self.lambda_reg * loss
