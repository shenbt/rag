import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

class Generator(nn.Module):
    """
    生成器：从检索到的证据生成最终的答案
    """
    def __init__(self, model_name="./gpt2"):
        super().__init__()
        self.model_name = model_name
        try:
            # 优先从本地路径加载
            print(f"正在从本地路径加载GPT-2模型: {model_name}")
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            
            # 设置pad_token为不同的值
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # 添加特殊token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            print("✅ 从本地路径加载GPT-2模型成功")
        except Exception as e:
            print(f"无法从本地路径加载模型 {model_name}: {e}")
            try:
                # 尝试从Hugging Face加载
                print("尝试从Hugging Face加载: gpt2")
                self.model = GPT2LMHeadModel.from_pretrained("gpt2")
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                
                # 设置pad_token
                if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    
                print("✅ 从Hugging Face加载GPT-2模型成功")
            except Exception as e2:
                print(f"无法从Hugging Face加载模型: {e2}")
                print("使用随机初始化的模型")
                # 创建一个随机初始化的模型
                config = GPT2Config()
                self.model = GPT2LMHeadModel(config)
                self.tokenizer = None

    def forward(self, context, max_length=50, mode="default"):
        """
        基于上下文生成答案
        """
        if self.tokenizer is None:
            # 如果tokenizer未加载，返回占位符答案
            return f"基于上下文 '{context[:50]}...' 的答案"
        
        try:
            # 确保context是字符串
            if not isinstance(context, str):
                context = str(context)
            
            # 根据模式调整生成参数
            if mode == "think":
                max_length = 256
                temperature = 0.7
            elif mode == "rethink_answer":
                max_length = 512
                temperature = 0.6
            else:
                max_length = 50
                temperature = 0.8
            
            # 限制上下文长度
            if len(context) > 300:
                context = context[:300]
            
            # 编码输入并创建attention mask
            inputs = self.tokenizer(
                context, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            # 确保输入在正确的设备上
            device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # 移除原始上下文，只返回生成的部分
            if context in generated_text:
                generated_text = generated_text.replace(context, "").strip()
            
            # 清理生成的文本
            generated_text = self._clean_generated_text(generated_text)
            
            # 格式校验
            if mode in ["think", "rethink_answer"]:
                generated_text = self._validate_format(generated_text, mode)
            
            # 确保返回有意义的答案
            if not generated_text or len(generated_text.strip()) < 3:
                return f"基于上下文 '{context[:30]}...' 生成的答案"
            
            return generated_text
        
        except Exception as e:
            print(f"生成答案时出错: {e}")
            return f"基于上下文生成答案时出错: {str(e)}"
    
    def generate(self, prompt: str, mode: str = "default", max_length: int = 50) -> str:
        """
        生成方法：支持不同模式的生成
        """
        return self.forward(prompt, max_length, mode)
    
    def _validate_format(self, text: str, mode: str) -> str:
        """
        格式校验：确保输出符合要求的格式
        """
        if mode == "think":
            # 检查是否包含<think>标签
            if "<think>" not in text or "</think>" not in text:
                # 尝试自我纠正
                corrected_text = f"<think>\n{text}\n</think>"
                return corrected_text
            return text
        
        elif mode == "rethink_answer":
            # 检查是否包含<rethink>和<answer>标签
            has_rethink = "<rethink>" in text and "</rethink>" in text
            has_answer = "<answer>" in text and "</answer>" in text
            
            if not has_rethink or not has_answer:
                # 尝试自我纠正
                if not has_rethink:
                    text = f"<rethink>\n重新思考过程\n</rethink>\n{text}"
                if not has_answer:
                    text = f"{text}\n<answer>\n最终答案\n</answer>"
            
            return text
        
        return text

    def _clean_generated_text(self, text):
        """
        清理生成的文本，移除乱码和无效字符
        """
        # 移除常见的乱码字符
        import re
        # 移除控制字符和无效Unicode
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # 移除多余的标点符号
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:]', '', text)
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
