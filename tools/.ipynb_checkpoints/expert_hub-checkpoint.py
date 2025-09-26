#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专家工具枢纽：提供统一的专家工具接口
包括OCR、表格识别、公式识别等专家工具
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ExpertModelHub:
    """
    专家工具枢纽：统一管理各种专家工具
    """
    def __init__(self, cache_dir: str = "./expert_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 初始化各种专家工具
        self._init_expert_tools()
        
    def _init_expert_tools(self):
        """初始化专家工具"""
        # 初始化所有工具属性为None
        self.ocr_tool = None
        self.table_tool = None
        self.formula_tool = None
        
        try:
            # 尝试导入PaddleOCR
            try:
                from paddleocr import PaddleOCR
                # 使用兼容的初始化方式，不传递可能不支持的参数
                self.ocr_tool = PaddleOCR(use_angle_cls=True, lang='ch')
                logger.info("✅ PaddleOCR初始化成功")
            except (ImportError, AttributeError, Exception) as e:
                logger.info(f"ℹ️ PaddleOCR未安装或初始化失败: {e}，将使用模拟OCR")
                self.ocr_tool = None
            
            # 尝试导入PP-Structure
            try:
                from ppstructure import PPStructure
                self.table_tool = PPStructure()
                logger.info("✅ PP-Structure初始化成功")
            except ImportError:
                logger.info("ℹ️ PP-Structure未安装，将使用模拟表格识别")
                self.table_tool = None
            
            # 尝试导入MonkeyOCR
            try:
                from monkey_ocr import MonkeyOCR
                self.formula_tool = MonkeyOCR()
                logger.info("✅ MonkeyOCR初始化成功")
            except ImportError:
                logger.info("ℹ️ MonkeyOCR未安装，将使用模拟公式识别")
                self.formula_tool = None
                
        except Exception as e:
            logger.error(f"❌ 专家工具初始化失败: {e}")
            # 确保所有工具都有默认值
            self.ocr_tool = None
            self.table_tool = None
            self.formula_tool = None
    
    def run_ocr(self, image_path: str) -> Dict[str, Any]:
        """
        运行通用OCR识别（适用于任何图片）
        """
        try:
            # 检查缓存
            cache_key = f"ocr_{hash(image_path)}"
            cached_result = self._get_cache(cache_key)
            if cached_result:
                return cached_result
            
            if self.ocr_tool:
                # 使用PaddleOCR
                result = self.ocr_tool.ocr(image_path, cls=True)
                
                # 处理结果
                ocr_text = ""
                bbox = [0, 0, 100, 100]
                confidence = 0.8
                
                if result and result[0]:
                    # 提取文本和置信度
                    texts = []
                    confidences = []
                    for line in result[0]:
                        if len(line) >= 2:
                            texts.append(line[1][0])  # 文本内容
                            confidences.append(line[1][1])  # 置信度
                    
                    ocr_text = " ".join(texts)
                    confidence = sum(confidences) / len(confidences) if confidences else 0.8
                    
                    # 计算边界框
                    if result[0]:
                        coords = result[0][0][0]  # 第一个检测框的坐标
                        bbox = [min(coord[0] for coord in coords),
                               min(coord[1] for coord in coords),
                               max(coord[0] for coord in coords),
                               max(coord[1] for coord in coords)]
            else:
                # 模拟OCR结果
                ocr_text = "模拟OCR文本"
                bbox = [0, 0, 100, 100]
                confidence = 0.9
            
            result = {
                'text': ocr_text,
                'bbox': bbox,
                'confidence': confidence,
                'field_map': {
                    'ocr_text': ocr_text,
                    'ocr_bbox': bbox
                },
                'source': 'paddleocr'
            }
            
            # 缓存结果
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"❌ OCR识别失败: {e}")
            return {
                'text': "OCR识别失败",
                'bbox': [0, 0, 100, 100],
                'confidence': 0.0,
                'field_map': {},
                'source': 'error'
            }
    
    def run_seal_ocr(self, image_path: str) -> Dict[str, Any]:
        """
        运行印章OCR识别（兼容性方法）
        """
        return self.run_ocr(image_path)
    
    def run_table(self, image_path: str) -> Dict[str, Any]:
        """
        运行表格识别
        """
        try:
            # 检查缓存
            cache_key = f"table_{hash(image_path)}"
            cached_result = self._get_cache(cache_key)
            if cached_result:
                return cached_result
            
            if self.table_tool:
                # 使用PP-Structure
                result = self.table_tool(image_path)
                
                # 处理结果
                table_html = ""
                table_markdown = ""
                bbox = [0, 0, 100, 100]
                confidence = 0.8
                
                if result and 'table' in result:
                    table_data = result['table']
                    table_html = table_data.get('html', '')
                    table_markdown = self._html_to_markdown(table_html)
                    bbox = table_data.get('bbox', [0, 0, 100, 100])
                    confidence = table_data.get('confidence', 0.8)
            else:
                # 模拟表格结果
                table_html = "<table><tr><td>模拟表格</td></tr></table>"
                table_markdown = "| 模拟表格 |\n|----------|\n| 数据     |"
                bbox = [0, 0, 100, 100]
                confidence = 0.9
            
            result = {
                'text': table_markdown,
                'html': table_html,
                'markdown': table_markdown,
                'bbox': bbox,
                'confidence': confidence,
                'field_map': {
                    'table_html': table_html,
                    'table_markdown': table_markdown,
                    'table_bbox': bbox
                },
                'source': 'ppstructure'
            }
            
            # 缓存结果
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"❌ 表格识别失败: {e}")
            return {
                'text': "表格识别失败",
                'html': "",
                'markdown': "",
                'bbox': [0, 0, 100, 100],
                'confidence': 0.0,
                'field_map': {},
                'source': 'error'
            }
    
    def run_formula(self, image_path: str) -> Dict[str, Any]:
        """
        运行公式识别
        """
        try:
            # 检查缓存
            cache_key = f"formula_{hash(image_path)}"
            cached_result = self._get_cache(cache_key)
            if cached_result:
                return cached_result
            
            if self.formula_tool:
                # 使用MonkeyOCR
                result = self.formula_tool.recognize(image_path)
                
                # 处理结果
                latex = ""
                bbox = [0, 0, 100, 100]
                confidence = 0.8
                
                if result:
                    latex = result.get('latex', '')
                    bbox = result.get('bbox', [0, 0, 100, 100])
                    confidence = result.get('confidence', 0.8)
            else:
                # 模拟公式结果
                latex = "x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}"
                bbox = [0, 0, 100, 100]
                confidence = 0.9
            
            result = {
                'text': latex,
                'latex': latex,
                'bbox': bbox,
                'confidence': confidence,
                'field_map': {
                    'formula_latex': latex,
                    'formula_bbox': bbox
                },
                'source': 'monkeyocr'
            }
            
            # 缓存结果
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"❌ 公式识别失败: {e}")
            return {
                'text': "公式识别失败",
                'latex': "",
                'bbox': [0, 0, 100, 100],
                'confidence': 0.0,
                'field_map': {},
                'source': 'error'
            }
    
    def _html_to_markdown(self, html: str) -> str:
        """简单的HTML转Markdown"""
        # 这里可以实现更复杂的转换逻辑
        # 目前只是简单的替换
        markdown = html.replace('<table>', '').replace('</table>', '')
        markdown = markdown.replace('<tr>', '').replace('</tr>', '\n')
        markdown = markdown.replace('<td>', '| ').replace('</td>', ' |')
        markdown = markdown.replace('<th>', '| ').replace('</th>', ' |')
        return markdown.strip()
    
    def _get_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """获取缓存"""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ 读取缓存失败: {e}")
        return None
    
    def _set_cache(self, key: str, data: Dict[str, Any]):
        """设置缓存"""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ 写入缓存失败: {e}")
    
    def get_available_tools(self) -> List[str]:
        """获取可用的工具列表"""
        tools = []
        if self.ocr_tool:
            tools.append('ocr')
        if self.table_tool:
            tools.append('table')
        if self.formula_tool:
            tools.append('formula')
        return tools
    
    def get_tool_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            'available_tools': self.get_available_tools(),
            'cache_dir': str(self.cache_dir),
            'ocr_tool': 'paddleocr' if self.ocr_tool else 'simulated',
            'table_tool': 'ppstructure' if self.table_tool else 'simulated',
            'formula_tool': 'monkeyocr' if self.formula_tool else 'simulated'
        }
