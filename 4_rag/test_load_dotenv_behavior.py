#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 load_dotenv 函数的行为
验证它在找不到文件时是否会报错
"""

import os
from dotenv import load_dotenv

def clear_env_vars():
    """清除测试相关的环境变量"""
    vars_to_clear = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "TEST_VAR"]
    for var in vars_to_clear:
        if var in os.environ:
            del os.environ[var]

def test_load_dotenv_behavior():
    print("=" * 80)
    print("测试 load_dotenv 函数的行为")
    print("=" * 80)
    
    print(f"当前工作目录: {os.getcwd()}")
    
    # 测试1: 加载存在的文件
    print("\n" + "-" * 60)
    print("测试1: 加载存在的 .env 文件")
    print("-" * 60)
    
    clear_env_vars()
    
    try:
        result = load_dotenv(dotenv_path="../.env", override=True, verbose=True)
        print(f"load_dotenv 返回值: {result}")
        print(f"返回值类型: {type(result)}")
        
        # 检查环境变量是否被加载
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        print(f"OPENAI_API_KEY: {api_key}")
        print(f"OPENAI_BASE_URL: {base_url}")
        
    except Exception as e:
        print(f"❌ 异常: {type(e).__name__}: {e}")
    
    # 测试2: 加载不存在的文件 (verbose=True)
    print("\n" + "-" * 60)
    print("测试2: 加载不存在的文件 (verbose=True)")
    print("-" * 60)
    
    clear_env_vars()
    
    try:
        result = load_dotenv(dotenv_path="nonexistent.env", override=True, verbose=True)
        print(f"load_dotenv 返回值: {result}")
        print(f"返回值类型: {type(result)}")
        
        # 检查环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        print(f"OPENAI_API_KEY: {api_key}")
        print(f"OPENAI_BASE_URL: {base_url}")
        
    except Exception as e:
        print(f"❌ 异常: {type(e).__name__}: {e}")
    
    # 测试3: 加载不存在的文件 (verbose=False)
    print("\n" + "-" * 60)
    print("测试3: 加载不存在的文件 (verbose=False)")
    print("-" * 60)
    
    clear_env_vars()
    
    try:
        result = load_dotenv(dotenv_path="nonexistent.env", override=True, verbose=False)
        print(f"load_dotenv 返回值: {result}")
        print(f"返回值类型: {type(result)}")
        
        # 检查环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        print(f"OPENAI_API_KEY: {api_key}")
        print(f"OPENAI_BASE_URL: {base_url}")
        
    except Exception as e:
        print(f"❌ 异常: {type(e).__name__}: {e}")
    
    # 测试4: 测试不同的路径情况
    print("\n" + "-" * 60)
    print("测试4: 测试不同的路径情况")
    print("-" * 60)
    
    test_paths = [
        ("../.env", "存在的相对路径"),
        ("../../.env", "不存在的相对路径"),
        ("../../../.env", "更深层的不存在路径"),
        ("/nonexistent/path/.env", "绝对路径不存在"),
        ("", "空路径"),
        (None, "None路径"),
    ]
    
    for path, description in test_paths:
        print(f"\n测试路径: {path} ({description})")
        clear_env_vars()
        
        try:
            if path is not None:
                result = load_dotenv(dotenv_path=path, override=True, verbose=True)
            else:
                result = load_dotenv(dotenv_path=path, override=True, verbose=True)
            
            print(f"  返回值: {result}")
            api_key = os.getenv("OPENAI_API_KEY")
            print(f"  OPENAI_API_KEY: {api_key}")
            
        except Exception as e:
            print(f"  ❌ 异常: {type(e).__name__}: {e}")
    
    # 测试5: 模拟你的具体情况
    print("\n" + "=" * 80)
    print("测试5: 模拟你的具体情况")
    print("=" * 80)
    
    # 模拟从不同工作目录运行的情况
    original_cwd = os.getcwd()
    
    test_scenarios = [
        (original_cwd, "从 4_rag 目录运行"),
        (os.path.dirname(original_cwd), "从 langchain-crash-course 目录运行"),
        (os.path.dirname(os.path.dirname(original_cwd)), "从 project_one 目录运行"),
    ]
    
    for test_dir, description in test_scenarios:
        if os.path.exists(test_dir):
            print(f"\n{description}:")
            print(f"工作目录: {test_dir}")
            
            try:
                os.chdir(test_dir)
                clear_env_vars()
                
                # 检查 ../.env 文件是否存在
                target_path = os.path.join(test_dir, "..", ".env")
                normalized_path = os.path.normpath(target_path)
                file_exists = os.path.exists(normalized_path)
                
                print(f"  '../.env' 指向: {normalized_path}")
                print(f"  文件存在: {file_exists}")
                
                # 执行 load_dotenv
                result = load_dotenv(dotenv_path="../.env", override=True, verbose=True)
                print(f"  load_dotenv 返回值: {result}")
                
                api_key = os.getenv("OPENAI_API_KEY")
                print(f"  加载后的 OPENAI_API_KEY: {api_key}")
                
            except Exception as e:
                print(f"  ❌ 异常: {type(e).__name__}: {e}")
            finally:
                os.chdir(original_cwd)  # 恢复原始工作目录
    
    print(f"\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    print("1. load_dotenv 在找不到文件时 **不会抛出异常**")
    print("2. 它会返回 False 表示加载失败，True 表示加载成功")
    print("3. verbose=True 会打印调试信息，但不会改变错误处理行为")
    print("4. 相对路径的解析依赖于当前工作目录")
    print("5. 这就是为什么你的两种用法可能产生不同结果的原因")

if __name__ == "__main__":
    test_load_dotenv_behavior()
