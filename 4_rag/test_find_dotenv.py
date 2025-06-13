#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 find_dotenv 函数的行为
验证 raise_error_if_not_found=True 参数是否真的会报错
"""

import os
from dotenv import find_dotenv

def test_find_dotenv_behavior():
    print("=" * 80)
    print("测试 find_dotenv 函数的行为")
    print("=" * 80)
    
    print(f"当前工作目录: {os.getcwd()}")
    print(f"当前脚本目录: {os.path.dirname(os.path.abspath(__file__))}")
    
    # 测试1: 查找存在的文件
    print("\n" + "-" * 50)
    print("测试1: 查找存在的 .env 文件")
    print("-" * 50)
    
    try:
        dotenv_path = find_dotenv("../.env", raise_error_if_not_found=True)
        print(f"✅ 成功找到: {dotenv_path}")
        print(f"绝对路径: {os.path.abspath(dotenv_path)}")
        print(f"文件存在: {os.path.exists(dotenv_path)}")
    except Exception as e:
        print(f"❌ 出现异常: {type(e).__name__}: {e}")
    
    # 测试2: 查找不存在的文件 (raise_error_if_not_found=True)
    print("\n" + "-" * 50)
    print("测试2: 查找不存在的文件 (raise_error_if_not_found=True)")
    print("-" * 50)
    
    try:
        dotenv_path = find_dotenv("nonexistent.env", raise_error_if_not_found=True)
        print(f"🤔 意外成功: {dotenv_path}")
        print(f"文件存在: {os.path.exists(dotenv_path) if dotenv_path else 'None'}")
    except Exception as e:
        print(f"✅ 正确抛出异常: {type(e).__name__}: {e}")
    
    # 测试3: 查找不存在的文件 (raise_error_if_not_found=False, 默认值)
    print("\n" + "-" * 50)
    print("测试3: 查找不存在的文件 (raise_error_if_not_found=False)")
    print("-" * 50)
    
    try:
        dotenv_path = find_dotenv("nonexistent.env", raise_error_if_not_found=False)
        print(f"返回值: {dotenv_path}")
        print(f"返回值类型: {type(dotenv_path)}")
        print(f"是否为空字符串: {dotenv_path == ''}")
        print(f"是否为None: {dotenv_path is None}")
    except Exception as e:
        print(f"❌ 意外异常: {type(e).__name__}: {e}")
    
    # 测试4: 查找不存在的文件 (不指定 raise_error_if_not_found 参数)
    print("\n" + "-" * 50)
    print("测试4: 查找不存在的文件 (默认参数)")
    print("-" * 50)
    
    try:
        dotenv_path = find_dotenv("nonexistent.env")
        print(f"返回值: {dotenv_path}")
        print(f"返回值类型: {type(dotenv_path)}")
        print(f"是否为空字符串: {dotenv_path == ''}")
        print(f"是否为None: {dotenv_path is None}")
    except Exception as e:
        print(f"❌ 意外异常: {type(e).__name__}: {e}")
    
    # 测试5: 测试不同的搜索路径
    print("\n" + "-" * 50)
    print("测试5: 测试不同的搜索路径")
    print("-" * 50)
    
    search_patterns = [
        ".env",
        "../.env", 
        "../../.env",
        "../../../.env",
        "nonexistent.env"
    ]
    
    for pattern in search_patterns:
        try:
            result = find_dotenv(pattern, raise_error_if_not_found=False)
            exists = os.path.exists(result) if result else False
            print(f"模式 '{pattern}': {result} (存在: {exists})")
        except Exception as e:
            print(f"模式 '{pattern}': 异常 - {type(e).__name__}: {e}")
    
    # 测试6: 验证 find_dotenv 的搜索机制
    print("\n" + "-" * 50)
    print("测试6: find_dotenv 的搜索机制")
    print("-" * 50)
    
    # find_dotenv 会从当前目录开始向上搜索
    print("find_dotenv 的搜索机制:")
    print("1. 从当前工作目录开始")
    print("2. 逐级向上搜索父目录")
    print("3. 直到找到指定的文件或到达根目录")
    
    # 手动模拟搜索过程
    current_dir = os.getcwd()
    search_file = ".env"
    
    print(f"\n手动搜索 '{search_file}' 文件:")
    search_dir = current_dir
    level = 0
    
    while level < 5:  # 限制搜索层级
        test_path = os.path.join(search_dir, search_file)
        exists = os.path.exists(test_path)
        print(f"  级别 {level}: {test_path} (存在: {exists})")
        
        if exists:
            print(f"  ✅ 在级别 {level} 找到文件")
            break
            
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:  # 到达根目录
            print("  🔚 到达根目录，停止搜索")
            break
            
        search_dir = parent_dir
        level += 1

if __name__ == "__main__":
    test_find_dotenv_behavior()
