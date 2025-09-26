import torch
import torch.nn as nn

class ReRanker(nn.Module):
    """
    二阶段精排模块：在第一阶段检索后的候选页面中重新排序
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(self, query_embedding, candidates_embedding):
        """
        对候选页面进行精排
        """
        similarities = torch.matmul(candidates_embedding, query_embedding.t())
        return similarities
