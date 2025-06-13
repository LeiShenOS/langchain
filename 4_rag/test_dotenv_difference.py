#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 load_dotenv 不同用法的差异
演示为什么相对路径和绝对路径会导致不同的结果
"""

import os
from dotenv import find_dotenv, load_dotenv

def clear_env_vars():
    """清除相关的环境变量"""
    vars_to_clear = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"]
    for var in vars_to_clear:
        if var in os.environ:
            del os.environ[var]

def test_dotenv_loading():
    print("=" * 80)
    print("测试 load_dotenv 不同用法的差异")
    print("=" * 80)

    print(f"当前工作目录: {os.getcwd()}")
    print(f"当前脚本位置: {os.path.dirname(os.path.abspath(__file__))}")

    # 使用 find_dotenv 查找 .env 文件
    try:
        dotenv_path = find_dotenv("../.env", raise_error_if_not_found=True)
        print(f"find_dotenv 找到的路径: {dotenv_path}")
        print(f"绝对路径: {os.path.abspath(dotenv_path)}")
    except Exception as e:
        print(f"find_dotenv 出错: {e}")
        return

    # 测试场景1: 模拟原始代码的问题
    print("\n" + "=" * 80)
    print("场景1: 模拟可能导致不同结果的情况")
    print("=" * 80)

    # 先设置一些系统环境变量（模拟系统中已有的环境变量）
    os.environ["OPENAI_API_KEY"] = "system_default_key"
    os.environ["OPENAI_BASE_URL"] = "system_default_url"

    print("设置了系统环境变量:")
    print(f"  OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY')}")
    print(f"  OPENAI_BASE_URL: {os.environ.get('OPENAI_BASE_URL')}")

    print("\n" + "-" * 50)
    print("测试1a: 使用相对路径 '../.env' (override=False)")
    print("-" * 50)

    # 方法1: 使用相对路径，不覆盖
    load_dotenv(dotenv_path="../.env", override=False)
    api_key_1a = os.getenv("OPENAI_API_KEY")
    base_url_1a = os.getenv("OPENAI_BASE_URL")
    print(f"API Key: {api_key_1a}")
    print(f"Base URL: {base_url_1a}")

    print("\n" + "-" * 50)
    print("测试1b: 使用相对路径 '../.env' (override=True)")
    print("-" * 50)

    # 方法1: 使用相对路径，覆盖
    load_dotenv(dotenv_path="../.env", override=True)
    api_key_1b = os.getenv("OPENAI_API_KEY")
    base_url_1b = os.getenv("OPENAI_BASE_URL")
    print(f"API Key: {api_key_1b}")
    print(f"Base URL: {base_url_1b}")

    print("\n" + "-" * 50)
    print("测试2: 使用绝对路径 (override=True)")
    print("-" * 50)

    # 方法2: 使用绝对路径
    load_dotenv(dotenv_path=os.path.abspath(dotenv_path), override=True)
    api_key_2 = os.getenv("OPENAI_API_KEY")
    base_url_2 = os.getenv("OPENAI_BASE_URL")
    print(f"API Key: {api_key_2}")
    print(f"Base URL: {base_url_2}")

    print("\n" + "=" * 80)
    print("结果比较:")
    print("=" * 80)
    print(f"相对路径(override=False) - API Key: {api_key_1a}")
    print(f"相对路径(override=True)  - API Key: {api_key_1b}")
    print(f"绝对路径(override=True)  - API Key: {api_key_2}")
    print(f"1a == 1b: {api_key_1a == api_key_1b}")
    print(f"1b == 2:  {api_key_1b == api_key_2}")

def check_env_files():
    """检查可能的 .env 文件位置"""
    print("\n" + "=" * 80)
    print("检查可能的 .env 文件位置:")
    print("=" * 80)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, ".env"),
        os.path.join(current_dir, "..", ".env"),
        os.path.join(current_dir, "..", "..", ".env"),
        os.path.join(current_dir, "..", "..", "chat_model", ".env"),
    ]

    for i, path in enumerate(possible_paths, 1):
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        print(f"\n{i}. 路径: {path}")
        print(f"   绝对路径: {abs_path}")
        print(f"   存在: {exists}")
        if exists:
            # 读取文件内容
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print("   文件内容:")
                    for j, line in enumerate(lines, 1):
                        if line.strip() and not line.strip().startswith('#'):
                            print(f"     {j}: {line.strip()}")
            except Exception as e:
                print(f"   读取文件出错: {e}")

if __name__ == "__main__":
    test_dotenv_loading()
    check_env_files()
