#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理数据集：支持四步推理训练
包括think、tool、rethink、answer四个阶段的数据
"""

import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from transformers import LayoutLMv3Tokenizer
from typing import Dict, List, Any, Optional

class OpenDocVQAReasoningDataset(Dataset):
    """
    推理数据集：为每条样本构造四步推理链
    """
    def __init__(self, data_dir: str, tokenizer, image_size: int = 224, 
                 mode: str = "train"):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.mode = mode
        
        # 图像预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载数据集
        self.data = self._load_data()
        
        # 生成推理链
        self.reasoning_data = self._generate_reasoning_chains()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载原始数据"""
        try:
            with open(os.path.join(self.data_dir, f"{self.mode}.json"), 'r') as f:
                data = json.load(f)
            print(f"✅ 成功加载 {len(data)} 条{self.mode}数据")
            return data
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return []
    
    def _generate_reasoning_chains(self) -> List[Dict[str, Any]]:
        """生成推理链数据"""
        reasoning_data = []
        
        for item in self.data:
            # 为每个样本生成推理链
            reasoning_chain = self._create_reasoning_chain(item)
            reasoning_data.append(reasoning_chain)
        
        print(f"✅ 生成了 {len(reasoning_data)} 条推理链")
        return reasoning_data
    
    def _create_reasoning_chain(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """为单个样本创建推理链"""
        question = item.get('question', '')
        answer = item.get('answer', '')
        ocr_text = item.get('ocr', [])
        bbox = item.get('bbox', [0, 0, 100, 100])
        
        # 合并OCR文本
        if isinstance(ocr_text, list):
            evidence_text = " ".join(ocr_text)
        else:
            evidence_text = str(ocr_text)
        
        # 生成四步推理链
        think_trace = self._generate_think_trace(question, evidence_text)
        tool_result = self._generate_tool_result(question, evidence_text)
        rethink_trace = self._generate_rethink_trace(question, think_trace, tool_result)
        final_answer = self._generate_final_answer(question, answer)
        
        return {
            'question': question,
            'evidence_text': evidence_text,
            'bbox': bbox,
            'image': item.get('image', ''),
            'think_trace': think_trace,
            'tool_result': tool_result,
            'rethink_trace': rethink_trace,
            'final_answer': final_answer,
            'original_answer': answer
        }
    
    def _generate_think_trace(self, question: str, evidence: str) -> str:
        """生成思考过程"""
        # 基于问题和证据生成思考过程
        think_prompt = f"""基于以下证据，思考如何回答问题：

问题：{question}
证据：{evidence}

思考过程："""
        
        # 这里可以使用简单的模板生成，或者调用模型生成
        think_trace = f"""<think>
1. 理解问题：{question}
2. 分析证据：{evidence[:100]}...
3. 确定答案方向：基于证据内容
4. 需要确认的信息：可能需要进一步验证
</think>"""
        
        return think_trace
    
    def _generate_tool_result(self, question: str, evidence: str) -> str:
        """生成工具结果"""
        # 模拟专家工具的输出
        tool_result = f"""<tool>
工具：OCR识别
结果：{evidence[:50]}...
置信度：0.85
</tool>"""
        
        return tool_result
    
    def _generate_rethink_trace(self, question: str, think_trace: str, 
                              tool_result: str) -> str:
        """生成重新思考过程"""
        rethink_trace = f"""<rethink>
基于工具结果重新分析：
1. 工具输出与原始思考的对比
2. 发现的一致性：工具结果支持原始分析
3. 需要调整的地方：无
4. 置信度评估：0.9
</rethink>"""
        
        return rethink_trace
    
    def _generate_final_answer(self, question: str, original_answer: str) -> str:
        """生成最终答案"""
        return f"""<answer>
{original_answer}
</answer>"""
    
    def __len__(self):
        return len(self.reasoning_data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        item = self.reasoning_data[idx]
        
        # 处理图像
        image_path = os.path.join(self.data_dir, item['image'])
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except:
            # 如果图像加载失败，使用零张量
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # 处理文本输入
        input_text = f"{item['question']} {item['evidence_text']}"
        
        # 使用tokenizer处理文本
        inputs = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 处理bbox
        bbox = torch.tensor(item['bbox'], dtype=torch.long)
        
        return {
            'image': image,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'bbox': bbox,
            'question': item['question'],
            'evidence_text': item['evidence_text'],
            'think_trace': item['think_trace'],
            'tool_result': item['tool_result'],
            'rethink_trace': item['rethink_trace'],
            'final_answer': item['final_answer'],
            'original_answer': item['original_answer']
        }
    
    def get_reasoning_prompt(self, idx: int) -> str:
        """获取完整的推理提示词"""
        item = self.reasoning_data[idx]
        
        prompt = f"""请基于以下信息进行四步推理：

问题：{item['question']}

证据：{item['evidence_text']}

请按以下格式输出：

{item['think_trace']}

{item['tool_result']}

{item['rethink_trace']}

{item['final_answer']}"""
        
        return prompt
    
    def get_training_target(self, idx: int) -> str:
        """获取训练目标文本"""
        item = self.reasoning_data[idx]
        
        target = f"{item['think_trace']}\n{item['tool_result']}\n{item['rethink_trace']}\n{item['final_answer']}"
        return target


class ReasoningDataProcessor:
    """
    推理数据处理器：用于生成和验证推理数据
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def generate_reasoning_data(self, output_file: str = "reasoning_data.json"):
        """生成推理数据文件"""
        try:
            # 加载原始数据
            with open(os.path.join(self.data_dir, "train.json"), 'r') as f:
                original_data = json.load(f)
            
            reasoning_data = []
            
            for item in original_data:
                # 为每个样本生成推理链
                reasoning_item = self._process_single_item(item)
                reasoning_data.append(reasoning_item)
            
            # 保存推理数据
            output_path = os.path.join(self.data_dir, output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(reasoning_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 推理数据已保存到 {output_path}")
            return reasoning_data
            
        except Exception as e:
            print(f"❌ 生成推理数据失败: {e}")
            return []
    
    def _process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个样本"""
        question = item.get('question', '')
        answer = item.get('answer', '')
        ocr_text = item.get('ocr', [])
        
        # 合并OCR文本
        if isinstance(ocr_text, list):
            evidence_text = " ".join(ocr_text)
        else:
            evidence_text = str(ocr_text)
        
        # 生成推理链
        reasoning_item = {
            'question': question,
            'evidence_text': evidence_text,
            'original_answer': answer,
            'think_trace': self._generate_think_trace(question, evidence_text),
            'tool_result': self._generate_tool_result(question, evidence_text),
            'rethink_trace': self._generate_rethink_trace(question, evidence_text),
            'final_answer': self._generate_final_answer(answer),
            'bbox': item.get('bbox', [0, 0, 100, 100]),
            'image': item.get('image', '')
        }
        
        return reasoning_item
    
    def _generate_think_trace(self, question: str, evidence: str) -> str:
        """生成思考过程"""
        return f"""<think>
基于问题"{question}"和证据"{evidence[:100]}..."进行分析：
1. 问题理解：需要从证据中提取相关信息
2. 证据分析：证据包含关键信息
3. 答案方向：基于证据内容确定答案
4. 验证需求：可能需要进一步确认
</think>"""
    
    def _generate_tool_result(self, question: str, evidence: str) -> str:
        """生成工具结果"""
        return f"""<tool>
工具：OCR识别
输入：{question}
输出：{evidence[:50]}...
置信度：0.85
</tool>"""
    
    def _generate_rethink_trace(self, question: str, evidence: str) -> str:
        """生成重新思考过程"""
        return f"""<rethink>
重新分析问题"{question}"：
1. 工具结果验证：证据内容准确
2. 一致性检查：工具输出与原始分析一致
3. 调整建议：无需调整
4. 置信度：0.9
</rethink>"""
    
    def _generate_final_answer(self, answer: str) -> str:
        """生成最终答案"""
        return f"""<answer>
{answer}
</answer>"""
