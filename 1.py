#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终的bbox测试脚本
"""

import torch
import numpy as np

def test_bbox_function():
    """测试bbox处理函数"""
    print("🧪 测试bbox处理函数...")
    
    try:
        from dataset.datasets import _to_1000_box
        
        # 测试各种bbox情况
        test_cases = [
            {
                'name': '正常bbox',
                'box': [100, 200, 300, 400],
                'w': 800,
                'h': 600
            },
            {
                'name': '超出范围bbox',
                'box': [900, 700, 1100, 800],
                'w': 800,
                'h': 600
            },
            {
                'name': '负坐标bbox',
                'box': [-10, -10, 110, 110],
                'w': 800,
                'h': 600
            },
            {
                'name': '顺序错误bbox',
                'box': [300, 400, 100, 200],
                'w': 800,
                'h': 600
            }
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n测试 {i+1}: {case['name']}")
            print(f"  原始bbox: {case['box']}")
            
            try:
                result = _to_1000_box(case['box'], case['w'], case['h'])
                print(f"  ✅ 处理结果: {result}")
                
                # 验证结果
                if all(0 <= coord <= 1000 for coord in result):
                    print(f"  ✅ 坐标范围正确")
                else:
                    print(f"  ❌ 坐标范围错误")
                
                if result[0] < result[2] and result[1] < result[3]:
                    print(f"  ✅ 坐标顺序正确")
                else:
                    print(f"  ❌ 坐标顺序错误")
                    
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
        
    except Exception as e:
        print(f"  ❌ bbox函数测试失败: {e}")

def test_encoder_bbox():
    """测试编码器bbox处理"""
    print("\n🧪 测试编码器bbox处理...")
    
    try:
        from models.encoder import RegionAwareEncoder
        
        encoder = RegionAwareEncoder()
        
        # 测试各种bbox格式
        test_cases = [
            {
                'name': '正常2D bbox',
                'bbox': torch.tensor([[0, 0, 125, 167]] * 10, dtype=torch.long)
            },
            {
                'name': '1D bbox',
                'bbox': torch.tensor([0, 0, 125, 167], dtype=torch.long)
            },
            {
                'name': '3D bbox',
                'bbox': torch.tensor([[[0, 0, 125, 167]] * 10] * 2, dtype=torch.long)
            },
            {
                'name': '超出范围bbox',
                'bbox': torch.tensor([[-10, -10, 1100, 1100]] * 10, dtype=torch.long)
            }
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n测试 {i+1}: {case['name']}")
            print(f"  原始bbox形状: {case['bbox'].shape}")
            print(f"  原始bbox范围: {torch.min(case['bbox']).item()}-{torch.max(case['bbox']).item()}")
            
            try:
                input_ids = torch.randint(0, 1000, (1, 10))
                attention_mask = torch.ones(1, 10)
                
                outputs = encoder._encode(input_ids, attention_mask, case['bbox'], None)
                print(f"  ✅ 编码成功，输出形状: {outputs.shape}")
                
                # 检查输出是否为零张量（表示使用了fallback）
                if torch.all(outputs == 0):
                    print(f"  ⚠️ 使用了fallback（零张量）")
                else:
                    print(f"  ✅ 正常编码输出")
                    
            except Exception as e:
                print(f"  ❌ 编码失败: {e}")
        
    except Exception as e:
        print(f"  ❌ 编码器bbox测试失败: {e}")

def test_training_loop():
    """测试训练循环"""
    print("\n🧪 测试训练循环...")
    
    try:
        from models.encoder import RegionAwareEncoder, RegionPreTrainer
        
        encoder = RegionAwareEncoder()
        trainer = RegionPreTrainer(encoder)
        
        # 创建测试数据
        batch_data = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10),
            'bbox': torch.tensor([[0, 0, 125, 167]] * 10, dtype=torch.long),
            'region_masks': torch.ones(1, 10).float(),
            'region_labels': torch.randint(0, 5, (1, 10))
        }
        
        print(f"  测试数据bbox形状: {batch_data['bbox'].shape}")
        print(f"  测试数据bbox范围: {torch.min(batch_data['bbox']).item()}-{torch.max(batch_data['bbox']).item()}")
        
        # 测试重构训练
        try:
            result = trainer.train_reconstruction(batch_data)
            print(f"  ✅ 重构训练成功: {result}")
        except Exception as e:
            print(f"  ❌ 重构训练失败: {e}")
        
        # 测试分类训练
        try:
            result = trainer.train_classification(batch_data)
            print(f"  ✅ 分类训练成功: {result}")
        except Exception as e:
            print(f"  ❌ 分类训练失败: {e}")
        
        # 测试对齐训练
        try:
            result = trainer.train_alignment(batch_data)
            print(f"  ✅ 对齐训练成功: {result}")
        except Exception as e:
            print(f"  ❌ 对齐训练失败: {e}")
        
    except Exception as e:
        print(f"  ❌ 训练循环测试失败: {e}")

if __name__ == "__main__":
    test_bbox_function()
    test_encoder_bbox()
    test_training_loop()
    print("\n✅ 所有测试完成")
