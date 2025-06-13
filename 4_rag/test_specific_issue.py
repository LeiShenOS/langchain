#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试具体的 dotenv 问题
模拟你遇到的具体情况
"""

import os
from dotenv import find_dotenv, load_dotenv

def clear_all_env():
    """清除所有相关环境变量"""
    vars_to_clear = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"]
    for var in vars_to_clear:
        if var in os.environ:
            del os.environ[var]

def test_scenario():
    print("=" * 80)
    print("测试具体场景：模拟你的代码中的问题")
    print("=" * 80)
    
    # 场景1：可能存在多个 .env 文件的情况
    print("\n场景1：检查 find_dotenv 的行为")
    print("-" * 50)
    
    # 清除环境变量
    clear_all_env()
    
    # 使用 find_dotenv 查找
    dotenv_path = find_dotenv("../.env", raise_error_if_not_found=True)
    print(f"find_dotenv 找到的路径: {dotenv_path}")
    print(f"绝对路径: {os.path.abspath(dotenv_path)}")
    
    # 场景2：模拟你的原始代码
    print("\n场景2：模拟原始代码的两行调用")
    print("-" * 50)
    
    clear_all_env()
    
    # 第一行：load_dotenv(dotenv_path="../.env", override=True)
    print("执行: load_dotenv(dotenv_path='../.env', override=True)")
    load_dotenv(dotenv_path="../.env", override=True)
    api_key_after_first = os.getenv("OPENAI_API_KEY")
    base_url_after_first = os.getenv("OPENAI_BASE_URL")
    print(f"第一次加载后 - API Key: {api_key_after_first}")
    print(f"第一次加载后 - Base URL: {base_url_after_first}")
    
    # 第二行：load_dotenv(dotenv_path=os.path.abspath(dotenv_path), override=True)
    print(f"\n执行: load_dotenv(dotenv_path=os.path.abspath('{dotenv_path}'), override=True)")
    print(f"实际路径: {os.path.abspath(dotenv_path)}")
    load_dotenv(dotenv_path=os.path.abspath(dotenv_path), override=True)
    api_key_after_second = os.getenv("OPENAI_API_KEY")
    base_url_after_second = os.getenv("OPENAI_BASE_URL")
    print(f"第二次加载后 - API Key: {api_key_after_second}")
    print(f"第二次加载后 - Base URL: {base_url_after_second}")
    
    print(f"\n结果比较:")
    print(f"第一次和第二次结果是否相同: {api_key_after_first == api_key_after_second}")
    
    # 场景3：检查路径解析的差异
    print("\n场景3：检查路径解析")
    print("-" * 50)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前脚本目录: {current_dir}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 相对路径解析
    relative_path = "../.env"
    resolved_relative = os.path.join(current_dir, relative_path)
    normalized_relative = os.path.normpath(resolved_relative)
    
    print(f"相对路径: {relative_path}")
    print(f"从脚本目录解析: {resolved_relative}")
    print(f"标准化后: {normalized_relative}")
    
    # find_dotenv 的路径
    find_dotenv_path = find_dotenv("../.env", raise_error_if_not_found=True)
    find_dotenv_abs = os.path.abspath(find_dotenv_path)
    
    print(f"find_dotenv 路径: {find_dotenv_path}")
    print(f"find_dotenv 绝对路径: {find_dotenv_abs}")
    
    print(f"\n路径比较:")
    print(f"标准化相对路径 == find_dotenv绝对路径: {normalized_relative == find_dotenv_abs}")
    
    # 场景4：测试不同工作目录的影响
    print("\n场景4：测试工作目录的影响")
    print("-" * 50)
    
    # 保存当前工作目录
    original_cwd = os.getcwd()
    
    try:
        # 切换到不同的工作目录
        parent_dir = os.path.dirname(current_dir)
        os.chdir(parent_dir)
        print(f"切换工作目录到: {os.getcwd()}")
        
        clear_all_env()
        
        # 在新工作目录下测试相对路径
        print("在新工作目录下执行: load_dotenv(dotenv_path='../.env', override=True)")
        try:
            load_dotenv(dotenv_path="../.env", override=True)
            api_key_new_cwd = os.getenv("OPENAI_API_KEY")
            print(f"新工作目录下的结果: {api_key_new_cwd}")
        except Exception as e:
            print(f"在新工作目录下加载失败: {e}")
            
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)
        print(f"恢复工作目录到: {os.getcwd()}")

if __name__ == "__main__":
    test_scenario()
