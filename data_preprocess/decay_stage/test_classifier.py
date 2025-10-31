#!/usr/bin/env python3
"""
测试文本分类脚本的简单示例
"""

import json
import os
import tempfile

def create_test_data():
    """创建测试数据"""
    test_data = [
        {"text": "def hello_world():\n    print('Hello, World!')\n\nhello_world()"},
        {"text": "数学公式：E = mc²，这是爱因斯坦的质能方程。"},
        {"text": "太阳系有八大行星，包括水星、金星、地球、火星、木星、土星、天王星和海王星。"},
        {"text": "这是一个质量很差的文本，内容不完整，逻辑混乱。"},
        {"text": ""},  # 空文本测试
    ]
    return test_data

def test_classifier():
    """测试分类器功能"""
    from text_classifier import TextClassifier
    
    # 创建测试数据
    test_data = create_test_data()
    
    # 创建分类器实例
    classifier = TextClassifier()
    
    print("测试分类器功能...")
    print("=" * 50)
    
    for i, item in enumerate(test_data):
        text = item["text"]
        print(f"\n测试样本 {i+1}:")
        print(f"文本: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # 分类
        result = classifier.classify_text(text)
        print(f"分类结果: {result}")
        
        # 验证结果格式
        assert "topic" in result, "结果缺少topic字段"
        assert "quality" in result, "结果缺少quality字段"
        assert result["topic"] in ["code", "math", "knowledge", "unknown"], f"无效的topic值: {result['topic']}"
        assert result["quality"] in [1, 2, 3, 4, 5], f"无效的quality值: {result['quality']}"
        
        print("✓ 结果格式正确")
    
    print("\n" + "=" * 50)
    print("所有测试通过！")

def test_jsonl_io():
    """测试JSONL文件读写功能"""
    from text_classifier import load_jsonl_data, save_classified_data
    
    # 创建测试数据
    test_data = create_test_data()
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        temp_input = f.name
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    try:
        # 测试加载功能
        print("测试JSONL文件加载...")
        loaded_data = load_jsonl_data(temp_input)
        assert len(loaded_data) == len(test_data), "加载数据数量不匹配"
        print("✓ JSONL文件加载成功")
        
        # 测试保存功能
        print("测试JSONL文件保存...")
        temp_output = temp_input.replace('.jsonl', '_output.jsonl')
        
        # 添加一些模拟的分类结果
        classified_data = []
        for item in loaded_data:
            classified_item = item.copy()
            classified_item.update({"topic": "knowledge", "quality": 3})
            classified_data.append(classified_item)
        
        save_classified_data(classified_data, temp_output)
        
        # 验证保存的文件
        with open(temp_output, 'r', encoding='utf-8') as f:
            saved_lines = f.readlines()
            assert len(saved_lines) == len(classified_data), "保存数据数量不匹配"
            
            for line in saved_lines:
                item = json.loads(line.strip())
                assert "text" in item, "保存的数据缺少text字段"
                assert "topic" in item, "保存的数据缺少topic字段"
                assert "quality" in item, "保存的数据缺少quality字段"
        
        print("✓ JSONL文件保存成功")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_input):
            os.unlink(temp_input)
        if os.path.exists(temp_output):
            os.unlink(temp_output)

if __name__ == "__main__":
    print("开始测试文本分类脚本...")
    
    try:
        test_classifier()
        test_jsonl_io()
        print("\n🎉 所有测试成功完成！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
