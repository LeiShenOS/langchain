#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细测试 find_dotenv 函数对路径参数的处理
"""

import os
from dotenv import find_dotenv

def test_find_dotenv_path_handling():
    print("=" * 80)
    print("详细测试 find_dotenv 函数对路径参数的处理")
    print("=" * 80)
    
    print(f"当前工作目录: {os.getcwd()}")
    
    # 测试不同的路径参数
    test_cases = [
        (".env", "搜索 .env 文件"),
        ("../.env", "搜索 ../.env 路径"),
        ("../../.env", "搜索 ../../.env 路径"),
        ("../../../.env", "搜索 ../../../.env 路径"),
        ("specific_name.env", "搜索特定名称的文件"),
    ]
    
    for filename, description in test_cases:
        print(f"\n" + "-" * 60)
        print(f"测试: {description}")
        print(f"参数: '{filename}'")
        print("-" * 60)
        
        try:
            # 测试 raise_error_if_not_found=False
            result_no_error = find_dotenv(filename, raise_error_if_not_found=False)
            print(f"raise_error_if_not_found=False:")
            print(f"  返回值: '{result_no_error}'")
            print(f"  是否为空: {result_no_error == ''}")
            
            if result_no_error:
                abs_path = os.path.abspath(result_no_error)
                exists = os.path.exists(result_no_error)
                print(f"  绝对路径: {abs_path}")
                print(f"  文件存在: {exists}")
            
            # 测试 raise_error_if_not_found=True
            try:
                result_with_error = find_dotenv(filename, raise_error_if_not_found=True)
                print(f"raise_error_if_not_found=True:")
                print(f"  返回值: '{result_with_error}'")
                if result_with_error:
                    abs_path = os.path.abspath(result_with_error)
                    exists = os.path.exists(result_with_error)
                    print(f"  绝对路径: {abs_path}")
                    print(f"  文件存在: {exists}")
            except Exception as e:
                print(f"raise_error_if_not_found=True:")
                print(f"  ❌ 抛出异常: {type(e).__name__}: {e}")
                
        except Exception as e:
            print(f"❌ 测试过程中出现异常: {type(e).__name__}: {e}")
    
    # 特别测试：理解 find_dotenv 的搜索逻辑
    print(f"\n" + "=" * 80)
    print("特别测试：find_dotenv 的搜索逻辑")
    print("=" * 80)
    
    print("\n关键发现：")
    print("1. find_dotenv 的第一个参数是要搜索的文件名，不是完整路径")
    print("2. 它会从当前目录开始，向上逐级搜索该文件名")
    print("3. 如果参数包含路径分隔符（如 '../.env'），它会按字面意思处理")
    
    # 验证这个理解
    print(f"\n验证搜索行为:")
    
    # 手动检查各级目录中是否存在 .env 文件
    current_dir = os.getcwd()
    levels_to_check = [
        (current_dir, "当前目录 (4_rag)"),
        (os.path.dirname(current_dir), "父目录 (langchain-crash-course)"),
        (os.path.dirname(os.path.dirname(current_dir)), "祖父目录 (project_one)"),
        (os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), "曾祖父目录 (learning_longchain)"),
    ]
    
    for directory, description in levels_to_check:
        env_file = os.path.join(directory, ".env")
        exists = os.path.exists(env_file)
        print(f"  {description}: {env_file} (存在: {exists})")
        
        if exists:
            # 显示文件的前几行
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line and not first_line.startswith('#'):
                        print(f"    第一行: {first_line}")
            except:
                pass
    
    # 最终测试：验证你的代码中的具体用法
    print(f"\n" + "=" * 80)
    print("验证你的代码中的具体用法")
    print("=" * 80)
    
    print("你的代码: find_dotenv('../.env', raise_error_if_not_found=True)")
    
    try:
        your_result = find_dotenv("../.env", raise_error_if_not_found=True)
        print(f"✅ 成功执行，返回: {your_result}")
        print(f"绝对路径: {os.path.abspath(your_result)}")
        print(f"文件存在: {os.path.exists(your_result)}")
        
        # 检查这个路径实际指向哪里
        target_path = os.path.join(os.getcwd(), "..", ".env")
        normalized_path = os.path.normpath(target_path)
        print(f"'../.env' 从当前目录解析为: {normalized_path}")
        print(f"该路径文件存在: {os.path.exists(normalized_path)}")
        
    except Exception as e:
        print(f"❌ 执行失败: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_find_dotenv_path_handling()
