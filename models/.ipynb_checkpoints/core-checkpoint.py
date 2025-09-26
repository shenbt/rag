#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniRAG核心管道：整合所有创新点的推理编排系统
包括四步推理：think -> tools -> rethink -> answer
"""

import torch
import time
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# 导入组件
from models.retriever import TwoStageRetriever, LayoutPlanner
from models.generator import Generator
from models.consistency_judge import ConsistencyJudge
from models.evidence_cards import EvidenceCard, EvidenceCardCollection
from retrieval.indexer import HaystackRetriever

logger = logging.getLogger(__name__)


class ReasoningController:
    """
    推理编排器：管理四步推理流程
    """
    def __init__(self, generator: Generator, consistency_judge: ConsistencyJudge):
        self.generator = generator
        self.consistency_judge = consistency_judge
        
    def run_think(self, question: str, evidence_cards: EvidenceCardCollection) -> str:
        """
        第一步：VLM自读思考
        给Generator一个"只读证据的思考模板"，要求输出 <think>…</think>
        """
        # 构建思考提示词
        evidence_text = self._format_evidence_for_thinking(evidence_cards)
        
        think_prompt = f"""请仔细阅读以下证据，然后进行思考：

问题：{question}

证据：
{evidence_text}

请输出你的思考过程，格式如下：
<think>
[在这里详细描述你的思考过程，包括：
1. 对问题的理解
2. 对证据的分析
3. 可能的答案方向
4. 需要进一步确认的信息]
</think>"""

        # 使用Generator生成思考
        think_output = self.generator.generate(
            prompt=think_prompt,
            mode="think",
            max_length=512
        )
        
        # 提取<think>标签内容
        think_content = self._extract_tag_content(think_output, "think")
        return think_content
    
    def run_tools(self, question: str, think_trace: str, image_path: str = None) -> EvidenceCardCollection:
        """
        第二步：调用专家工具并封装为证据卡
        通过专家工具枢纽获取结构化结果，落成EvidenceCard
        """
        # 导入专家工具枢纽
        from tools.expert_hub import ExpertModelHub
        
        expert_hub = ExpertModelHub()
        tool_cards = EvidenceCardCollection()
        
        # 分析思考过程，确定需要调用的工具
        required_tools = self._analyze_required_tools(question, think_trace)
        
        for tool_name in required_tools:
            try:
                # 调用专家工具
                if tool_name == "ocr":
                    if image_path:
                        result = expert_hub.run_ocr(image_path)
                    else:
                        # 如果没有图片路径，使用模拟结果
                        result = {
                            'text': f"模拟OCR结果：{question}",
                            'bbox': [0, 0, 100, 100],
                            'confidence': 0.8,
                            'field_map': {'ocr_text': f"模拟OCR结果：{question}"},
                            'source': 'simulation'
                        }
                elif tool_name == "table":
                    if image_path:
                        result = expert_hub.run_table(image_path)
                    else:
                        result = {
                            'text': f"模拟表格结果：{question}",
                            'bbox': [0, 0, 100, 100],
                            'confidence': 0.8,
                            'field_map': {'table_content': f"模拟表格结果：{question}"},
                            'source': 'simulation'
                        }
                elif tool_name == "formula":
                    if image_path:
                        result = expert_hub.run_formula(image_path)
                    else:
                        result = {
                            'text': f"模拟公式结果：{question}",
                            'bbox': [0, 0, 100, 100],
                            'confidence': 0.8,
                            'field_map': {'formula_content': f"模拟公式结果：{question}"},
                            'source': 'simulation'
                        }
                else:
                    continue
                
                # 封装为证据卡
                evidence_card = EvidenceCard(
                    ocr_text=result.get('text', ''),
                    bbox=result.get('bbox', [0, 0, 100, 100]),
                    page_id=f"tool_{tool_name}_{int(time.time())}",
                    confidence=result.get('confidence', 0.8),
                    source_type=tool_name
                )
                
                # 添加工具元数据
                evidence_card.add_metadata('source', f"tool:{tool_name}")
                evidence_card.add_metadata('field_map', result.get('field_map', {}))
                evidence_card.add_metadata('tool_result', result)
                
                tool_cards.add_card(evidence_card)
                
            except Exception as e:
                logger.warning(f"⚠️ 工具 {tool_name} 调用失败: {e}")
                continue
        
        return tool_cards
    
    def run_rethink(self, question: str, think_trace: str, 
                   tool_cards: EvidenceCardCollection) -> Dict[str, str]:
        """
        第三步：重新思考，对齐/纠错/融合
        构造比对提示词，引导模型给出<rethink>段落和候选<answer>
        """
        # 构建重新思考提示词
        tool_evidence = self._format_tool_evidence(tool_cards)
        
        rethink_prompt = f"""基于之前的思考和专家工具的结果，请重新思考并给出最终答案：

问题：{question}

原始思考：
{think_trace}

专家工具结果：
{tool_evidence}

请逐字段对比分析，输出：
<rethink>
[在这里详细说明：
1. 与专家结果的一致性分析
2. 发现的错误或差异
3. 纠错理由
4. 置信度评估]
</rethink>

<answer>
[最终答案]
</answer>"""

        # 使用Generator生成重新思考和答案
        rethink_output = self.generator.generate(
            prompt=rethink_prompt,
            mode="rethink_answer",
            max_length=1024
        )
        
        # 提取内容
        rethink_content = self._extract_tag_content(rethink_output, "rethink")
        answer_content = self._extract_tag_content(rethink_output, "answer")
        
        return {
            'rethink': rethink_content,
            'answer': answer_content
        }
    
    def _format_evidence_for_thinking(self, evidence_cards: EvidenceCardCollection) -> str:
        """格式化证据用于思考"""
        formatted_evidence = []
        for i, card in enumerate(evidence_cards.get_all_cards()):
            formatted_evidence.append(f"证据{i+1} ({card.source_type}): {card.ocr_text}")
        return "\n".join(formatted_evidence)
    
    def _analyze_required_tools(self, question: str, think_trace: str) -> List[str]:
        """分析需要调用的工具"""
        required_tools = []
        
        # 简单的关键词匹配
        if any(keyword in question.lower() for keyword in ['表格', 'table', '数据']):
            required_tools.append('table')
        
        if any(keyword in question.lower() for keyword in ['公式', 'formula', '数学']):
            required_tools.append('formula')
        
        # 默认总是需要OCR
        required_tools.append('ocr')
        
        return required_tools
    
    def _format_tool_evidence(self, tool_cards: EvidenceCardCollection) -> str:
        """格式化工具证据"""
        formatted_tools = []
        for card in tool_cards.get_all_cards():
            tool_name = card.metadata.get('source', 'unknown').replace('tool:', '')
            formatted_tools.append(f"{tool_name}: {card.ocr_text}")
        return "\n".join(formatted_tools)
    
    def _extract_tag_content(self, text: str, tag: str) -> str:
        """提取标签内容"""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            return text[start_idx + len(start_tag):end_idx].strip()
        else:
            return text.strip()


class UniRAGPipeline:
    """
    统一RAG管道：整合所有创新点的推理系统
    """
    def __init__(self, 
                 retriever: TwoStageRetriever,
                 layout_planner: LayoutPlanner,
                 generator: Generator,
                 consistency_judge: ConsistencyJudge,
                 evidence_cards: EvidenceCardCollection):
        
        self.retriever = retriever
        self.layout_planner = layout_planner
        self.generator = generator
        self.consistency_judge = consistency_judge
        self.evidence_cards = evidence_cards
        
        # 初始化推理编排器
        self.reasoning_controller = ReasoningController(generator, consistency_judge)
        
        # 重试参数
        self.max_retries = 3
        self.consistency_threshold = 0.4  # 降低阈值，减少重试
        
    def query(self, question: str) -> Dict[str, Any]:
        """
        主查询接口：四步推理编排
        检索→layout→think→tools→rethink+answer→一致性判别→（必要时）重试/降级
        """
        start_time = time.time()
        
        try:
            # 第一步：检索证据
            logger.info("🔍 第一步：检索证据")
            evidence_cards = self.retriever.retrieve(question)
            
            # 第二步：版式规划
            logger.info("📋 第二步：版式规划")
            layout_type = self.layout_planner.predict_layout(question, evidence_cards)
            
            # 第三步：VLM自读思考
            logger.info("🧠 第三步：VLM自读思考")
            think_trace = self.reasoning_controller.run_think(question, evidence_cards)
            
            # 第四步：调用专家工具
            logger.info("🔧 第四步：调用专家工具")
            tool_cards = self.reasoning_controller.run_tools(question, think_trace, image_path=None)
            
            # 合并所有证据
            all_evidence = EvidenceCardCollection()
            all_evidence.extend(evidence_cards)
            all_evidence.extend(tool_cards)
            
            # 第五步：重新思考并生成答案
            logger.info("🔄 第五步：重新思考并生成答案")
            rethink_result = self.reasoning_controller.run_rethink(question, think_trace, tool_cards)
            
            final_answer = rethink_result['answer']
            
            # 第六步：一致性判别
            logger.info("✅ 第六步：一致性判别")
            consistency_result = self.consistency_judge.check(final_answer, all_evidence)
            
            # 检查一致性，必要时重试
            retry_count = 0
            while (consistency_result['overall_score'] < self.consistency_threshold and 
                   retry_count < self.max_retries):
                
                logger.warning(f"⚠️ 一致性不足，进行第{retry_count + 1}次重试")
                
                try:
                    # 重新思考，使用更保守的策略
                    conservative_prompt = f"""由于一致性检查失败，请给出更保守的答案：

问题：{question}
原始答案：{final_answer}
一致性评分：{consistency_result['overall_score']}
问题诊断：{consistency_result.get('issues', [])}

请重新思考并给出更准确的答案：
<rethink>
[分析问题所在，给出纠错理由]
</rethink>

<answer>
[更准确的答案]
</answer>"""

                    retry_output = self.generator.generate(
                        prompt=conservative_prompt,
                        mode="rethink_answer",
                        max_length=1024
                    )
                    
                    final_answer = self.reasoning_controller._extract_tag_content(retry_output, "answer")
                    
                    # 如果答案为空，使用原始答案
                    if not final_answer or len(final_answer.strip()) == 0:
                        logger.warning("⚠️ 重试生成的答案为空，使用原始答案")
                        break
                    
                    consistency_result = self.consistency_judge.check(final_answer, all_evidence)
                    retry_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ 重试过程中发生错误: {e}")
                    # 如果重试失败，跳出循环
                    break
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 构建结果
            result = {
                'success': True,
                'answer': final_answer,
                'think_trace': think_trace,
                'rethink': rethink_result['rethink'],
                'evidence_cards': all_evidence,
                'tool_cards': tool_cards,
                'layout_type': layout_type,
                'consistency_score': consistency_result['overall_score'],
                'consistency_details': consistency_result,
                'response_time': response_time,
                'retry_count': retry_count
            }
            
            logger.info(f"✅ 查询完成，响应时间: {response_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ 查询失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def initialize(self, documents: List[Dict[str, Any]]):
        """初始化管道"""
        try:
            # 初始化检索器
            self.retriever.add_documents(documents)
            logger.info("✅ 管道初始化完成")
        except Exception as e:
            logger.error(f"❌ 管道初始化失败: {e}")
            raise
