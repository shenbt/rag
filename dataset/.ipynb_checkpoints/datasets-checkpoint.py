import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from transformers import LayoutLMv3Tokenizer

# ===== 新增：工具函数 =====
def _to_1000_box(box, w, h):
    # 支持 [x1,y1,x2,y2] & 自动修正顺序
    x1, y1, x2, y2 = box
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    
    # 简化的缩放函数
    if w > 0 and h > 0:
        x1 = max(0, min(1000, int(round(x1 / w * 1000))))
        y1 = max(0, min(1000, int(round(y1 / h * 1000))))
        x2 = max(0, min(1000, int(round(x2 / w * 1000))))
        y2 = max(0, min(1000, int(round(y2 / h * 1000))))
    else:
        # 如果图像尺寸无效，返回默认值
        return [0, 0, 100, 100]
    
    # 至少有 1 的宽高
    if x2 <= x1: x2 = min(1000, x1 + 1)
    if y2 <= y1: y2 = min(1000, y1 + 1)
    return [x1, y1, x2, y2]

class OpenDocVQADataset(Dataset):
    """
    OpenDocVQADataset 数据集加载器
    """
    def __init__(self, data_dir, tokenizer, image_size=224):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        
        # 图像预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载数据集
        with open(os.path.join(data_dir, "train.json"), 'r') as f:
            self.data = json.load(f)
        
        # 获取问题与答案
        self.questions = [item['question'] for item in self.data]
        self.answers = [item['answer'] for item in self.data]
        
        # 其他字段（图像路径，OCR文本，bbox）
        self.images = [item['image'] for item in self.data]
        self.bboxes = [item['bbox'] for item in self.data]
        self.ocr_texts = [item['ocr'] for item in self.data]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取当前数据项
        image_path = os.path.join(self.data_dir, self.images[idx])
        
        # ===== 在 __getitem__ 中，读取图片后立刻拿原始尺寸 =====
        try:
            orig_image = Image.open(image_path).convert("RGB")
            w, h = orig_image.size
            image = self.transform(orig_image)
        except Exception as e:
            # 如果无法获取原始图像尺寸，使用默认值
            w, h = 800, 600  # 假设的原始尺寸
            try:
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)
            except:
                # 如果图像文件有问题，创建一个空白图像
                image = torch.zeros(3, self.image_size, self.image_size)
        
        # OCR文本与bbox
        ocr_text = self.ocr_texts[idx]
        
        # 使用tokenizer处理文本
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # 为LayoutLMv3准备输入
        # 首先处理问题文本
        question_tokens = self.tokenizer.tokenize(question)
        
        # 处理OCR文本
        if isinstance(ocr_text, list):
            ocr_string = " ".join(ocr_text)
        else:
            ocr_string = str(ocr_text)
        
        ocr_tokens = self.tokenizer.tokenize(ocr_string)
        
        # 合并问题和OCR的token
        all_tokens = question_tokens + ocr_tokens
        
        # 你的 JSON 里 bboxes 可能是单框或多框；统一做成 list[list[int]]
        raw_bbox = self.bboxes[idx]
        if isinstance(raw_bbox[0], (int, float)):  # 单个框
            norm_bbox = _to_1000_box(raw_bbox, w, h)
            per_token_boxes = [norm_bbox] * len(all_tokens)
        else:  # 多个框，长度与 OCR 行或单词相关；不足时回填最后一个
            norm_boxes = [_to_1000_box(b, w, h) for b in raw_bbox]
            last = norm_boxes[-1] if norm_boxes else [0,0,1000,1000]
            per_token_boxes = (norm_boxes + [last] * len(all_tokens))[:len(all_tokens)]

        inputs = self.tokenizer(
            all_tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt',
            boxes=per_token_boxes  # ← 直接传规范化后的 boxes
        )

        # 如果你保留手工 bbox_tensor，确保和 inputs 长度一致：
        input_length = inputs['input_ids'].shape[1]
        bbox_tensor = torch.zeros(input_length, 4, dtype=torch.long)
        for i in range(min(len(per_token_boxes), input_length)):
            bbox_tensor[i] = torch.tensor(per_token_boxes[i], dtype=torch.long)

        return {
            'image': image,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'bbox': bbox_tensor,  # 使用手动构建的bbox张量
            'question': question,
            'answer': answer
        }
