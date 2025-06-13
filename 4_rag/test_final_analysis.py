#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终分析：为什么两种 load_dotenv 用法会产生不同结果
"""

import os
from dotenv import find_dotenv, load_dotenv

def analyze_dotenv_difference():
    print("=" * 80)
    print("分析 load_dotenv 不同用法产生不同结果的原因")
    print("=" * 80)
    
    # 获取当前信息
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    current_working_dir = os.getcwd()
    
    print(f"当前脚本目录: {current_script_dir}")
    print(f"当前工作目录: {current_working_dir}")
    
    # 分析可能的原因
    print("\n" + "=" * 80)
    print("可能导致不同结果的原因分析:")
    print("=" * 80)
    
    print("\n1. 工作目录不同导致相对路径解析不同")
    print("-" * 50)
    
    # 模拟不同工作目录
    scenarios = [
        ("4_rag目录", current_script_dir),
        ("langchain-crash-course目录", os.path.dirname(current_script_dir)),
        ("project_one目录", os.path.dirname(os.path.dirname(current_script_dir))),
        ("learning_longchain目录", os.path.dirname(os.path.dirname(os.path.dirname(current_script_dir)))),
    ]
    
    for name, directory in scenarios:
        if os.path.exists(directory):
            # 计算相对路径 "../.env" 在该目录下指向哪里
            relative_target = os.path.join(directory, "..", ".env")
            normalized_target = os.path.normpath(relative_target)
            exists = os.path.exists(normalized_target)
            
            print(f"\n在 {name} ({directory}):")
            print(f"  '../.env' 指向: {normalized_target}")
            print(f"  文件存在: {exists}")
            
            if exists:
                try:
                    with open(normalized_target, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line and not first_line.startswith('#'):
                            print(f"  第一行内容: {first_line}")
                except:
                    print("  无法读取文件")
    
    print("\n2. 环境变量名称不匹配")
    print("-" * 50)
    
    # 检查不同 .env 文件中的变量名
    env_files = [
        ("langchain-crash-course/.env", os.path.join(current_script_dir, "..", ".env")),
        ("chat_model/.env", os.path.join(current_script_dir, "..", "..", "chat_model", ".env")),
    ]
    
    for name, path in env_files:
        if os.path.exists(path):
            print(f"\n{name} 文件内容:")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            print(f"  {line_num}: {line}")
            except Exception as e:
                print(f"  读取失败: {e}")
    
    print("\n3. 测试实际的差异情况")
    print("-" * 50)
    
    # 清除环境变量
    for var in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"]:
        if var in os.environ:
            del os.environ[var]
    
    # 找到 dotenv_path
    try:
        dotenv_path = find_dotenv("../.env", raise_error_if_not_found=True)
        print(f"find_dotenv 找到: {dotenv_path}")
        print(f"绝对路径: {os.path.abspath(dotenv_path)}")
    except Exception as e:
        print(f"find_dotenv 失败: {e}")
        return
    
    # 测试场景：模拟可能的差异
    print("\n测试场景A: 先加载相对路径，再加载绝对路径")
    
    # 清除环境变量
    for var in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"]:
        if var in os.environ:
            del os.environ[var]
    
    # 第一步：相对路径
    print("步骤1: load_dotenv(dotenv_path='../.env', override=True)")
    result1 = load_dotenv(dotenv_path="../.env", override=True)
    print(f"  返回值: {result1}")
    api_key_1 = os.getenv("OPENAI_API_KEY")
    base_url_1 = os.getenv("OPENAI_BASE_URL")
    api_base_1 = os.getenv("OPENAI_API_BASE")  # 注意这个变量名
    print(f"  OPENAI_API_KEY: {api_key_1}")
    print(f"  OPENAI_BASE_URL: {base_url_1}")
    print(f"  OPENAI_API_BASE: {api_base_1}")
    
    # 第二步：绝对路径
    print(f"\n步骤2: load_dotenv(dotenv_path='{os.path.abspath(dotenv_path)}', override=True)")
    result2 = load_dotenv(dotenv_path=os.path.abspath(dotenv_path), override=True)
    print(f"  返回值: {result2}")
    api_key_2 = os.getenv("OPENAI_API_KEY")
    base_url_2 = os.getenv("OPENAI_BASE_URL")
    api_base_2 = os.getenv("OPENAI_API_BASE")
    print(f"  OPENAI_API_KEY: {api_key_2}")
    print(f"  OPENAI_BASE_URL: {base_url_2}")
    print(f"  OPENAI_API_BASE: {api_base_2}")
    
    print(f"\n结果比较:")
    print(f"  API_KEY 相同: {api_key_1 == api_key_2}")
    print(f"  BASE_URL 相同: {base_url_1 == base_url_2}")
    print(f"  API_BASE 相同: {api_base_1 == api_base_2}")
    
    # 关键发现
    print("\n" + "=" * 80)
    print("关键发现和结论:")
    print("=" * 80)
    
    print("1. 变量名不匹配问题:")
    print("   - .env 文件中是 OPENAI_BASE_URL")
    print("   - 但代码中可能使用了 OPENAI_API_BASE")
    print("   - 这会导致获取到 None 值")
    
    print("\n2. 工作目录影响:")
    print("   - 相对路径 '../.env' 依赖于当前工作目录")
    print("   - 如果脚本在不同目录下运行，相对路径会指向不同位置")
    print("   - 绝对路径不受工作目录影响")
    
    print("\n3. 文件存在性:")
    print("   - 如果相对路径指向的文件不存在，load_dotenv 会静默失败")
    print("   - 绝对路径如果正确，通常能找到文件")

if __name__ == "__main__":
    analyze_dotenv_difference()
