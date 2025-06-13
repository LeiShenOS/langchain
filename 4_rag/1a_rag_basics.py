# RAG（检索增强生成）基础示例 - 第一部分：向量存储初始化
# 本文件演示了如何创建和初始化一个向量数据库，这是RAG系统的核心组件之一
# RAG系统通过将文档转换为向量嵌入，然后存储在向量数据库中，实现高效的语义搜索

import os


from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
#What hanpend?就是我修改了一些内容
# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# 导入LangChain相关模块
from langchain.text_splitter import CharacterTextSplitter  # 文本分割器，用于将长文档分割成小块
from langchain_community.document_loaders import TextLoader  # 文本加载器，用于读取文本文件
from langchain_community.vectorstores import Chroma  # Chroma向量数据库，用于存储和检索向量嵌入
from langchain_openai import OpenAIEmbeddings  # OpenAI嵌入模型，用于将文本转换为向量
from dotenv import find_dotenv, load_dotenv
# 加载 .env 文件（指定路径，覆盖现有环境变量）
# 找到具体加载的 .env 文件路径（会搜索当前脚本目录及其父目录）
#因为这个有bug，就是load_dotenv函数之间有bug，导致没有找到也没有报错
#dotenv_path = find_dotenv("../.env", raise_error_if_not_found=True)
#print("最终使用的 .env 路径：", dotenv_path)
#print("最终使用的 .env 路径:", os.path.abspath(dotenv_path))
#就是这个之后可以注意一下，就是这种方式可能不太一样
load_dotenv(dotenv_path=r"C:\Users\LeiShen\Desktop\learning_longchain\project_one\langchain-crash-course\.env", override=True)
#load_dotenv(dotenv_path=os.path.abspath(dotenv_path), override=True)
#load_dotenv(dotenv_path=r"C:\Users\LeiShen\Desktop\learning_longchain\project_one\langchain-crash-course\.env", override=True)




# 从环境变量获取配置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

# 定义文件路径和持久化目录
# 这些路径用于指定要处理的文本文件位置和向量数据库的存储位置
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录
file_path = os.path.join(current_dir, "books", "odyssey.txt")  # 构建要处理的文本文件路径（奥德赛）
persistent_directory = os.path.join(current_dir, "db", "chroma_db")  # 构建向量数据库的持久化存储目录

# 检查Chroma向量数据库是否已经存在
# 如果不存在，则需要创建新的向量数据库；如果已存在，则跳过初始化过程
if not os.path.exists(persistent_directory):
    print("持久化目录不存在。正在初始化向量存储...")

    # 确保要处理的文本文件存在
    # 在开始处理之前验证文件是否存在，避免后续处理出错
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"文件 {file_path} 不存在。请检查路径。"
        )

    # 从文件中读取文本内容
    # TextLoader是LangChain提供的文档加载器，专门用于加载纯文本文件
    loader = TextLoader(file_path)
    documents = loader.load()  # 加载文档，返回Document对象列表
    #print(documents)

    # 将文档分割成小块（chunks）
    # 这是RAG系统中的关键步骤，因为：
    # 1. 大型语言模型有输入长度限制
    # 2. 较小的文本块能提供更精确的检索结果
    # 3. 向量嵌入对较短文本的表示更准确
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # chunk_size=1000: 每个文本块最大1000个字符
    # chunk_overlap=0: 文本块之间没有重叠（也可以设置重叠以保持上下文连续性）
    docs = text_splitter.split_documents(documents)

    # 显示分割后的文档信息
    # 这有助于了解文档被分割的情况，便于调试和优化
    print("\n--- 文档块信息 ---")
    print(f"文档块数量: {len(docs)}")
    print(f"示例文档块:\n{docs[0].page_content}\n")

    # 创建嵌入模型
    # 嵌入模型将文本转换为高维向量，这些向量能够捕获文本的语义信息
    print("\n--- 创建嵌入模型 ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # 使用OpenAI的小型嵌入模型，平衡性能和成本
        api_key=api_key,
        base_url=base_url
    )  # 如果需要，可以更新为其他有效的嵌入模型
    print("\n--- 嵌入模型创建完成 ---")

    # 创建向量存储并自动持久化
    # Chroma是一个开源的向量数据库，专门用于存储和检索向量嵌入
    print("\n--- 创建向量存储 ---")
    db = Chroma.from_documents(
        docs,  # 要存储的文档块
        embeddings,  # 嵌入模型，用于将文档转换为向量
        persist_directory=persistent_directory  # 持久化目录，数据库将保存在此目录中
    )
    # 这个过程包括：
    # 1. 为每个文档块生成向量嵌入
    # 2. 将向量和原始文本存储在Chroma数据库中
    # 3. 创建索引以支持高效检索
    print("\n--- 向量存储创建完成 ---")

else:
    print("向量存储已存在。无需初始化。")
