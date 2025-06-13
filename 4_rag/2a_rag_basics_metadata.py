# RAG基础示例 - 带元数据的多文档处理（第一部分）
# 本文件演示了如何处理多个文档并为每个文档添加元数据信息
# 元数据在RAG系统中非常重要，它可以帮助：
# 1. 追踪信息来源
# 2. 实现基于来源的过滤
# 3. 提供更丰富的上下文信息
# 4. 支持引用和溯源功能

import os

# 导入LangChain相关模块
from langchain.text_splitter import CharacterTextSplitter  # 文本分割器
from langchain_community.document_loaders import TextLoader  # 文本文件加载器
from langchain_community.vectorstores import Chroma  # Chroma向量数据库
from langchain_openai import OpenAIEmbeddings  # OpenAI嵌入模型
from dotenv import find_dotenv, load_dotenv
load_dotenv(dotenv_path=r"C:\Users\LeiShen\Desktop\learning_longchain\project_one\langchain-crash-course\.env", override=True)

# 定义包含文本文件的目录和持久化目录
# 这次我们处理整个books目录中的所有文本文件，而不是单个文件
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
books_dir = os.path.join(current_dir, "books")  # 书籍文件目录
db_dir = os.path.join(current_dir, "db")  # 数据库目录
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")  # 带元数据的向量数据库目录

# 显示目录信息，便于调试和确认路径
print(f"书籍目录: {books_dir}")
print(f"持久化目录: {persistent_directory}")

# 检查Chroma向量数据库是否已经存在
if not os.path.exists(persistent_directory):
    print("持久化目录不存在。正在初始化向量存储...")

    # 确保书籍目录存在
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"目录 {books_dir} 不存在。请检查路径。"
        )

    # 列出目录中的所有文本文件
    # 使用列表推导式筛选出所有.txt扩展名的文件
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    print(f"找到 {len(book_files)} 个文本文件")

    # 从每个文件中读取文本内容并存储元数据
    # 这是多文档处理的核心部分
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        print(f"正在处理文件: {book_file}")

        # 使用TextLoader加载单个文件
        loader = TextLoader(file_path)
        book_docs = loader.load()  # 加载文档（通常每个文件返回一个Document对象）

        # 为每个文档添加元数据
        for doc in book_docs:
            # 添加元数据以标识文档来源
            # 元数据是一个字典，可以包含任何有用的信息
            doc.metadata = {"source": book_file}
            # 可以添加更多元数据，例如：
            # doc.metadata = {
            #     "source": book_file,
            #     "file_type": "txt",
            #     "processed_date": datetime.now().isoformat(),
            #     "category": "literature"  # 根据文件名或内容分类
            # }
            documents.append(doc)

    print(f"总共加载了 {len(documents)} 个文档")

    # 将文档分割成小块
    # 对所有文档使用相同的分割策略
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    # 分割后，每个文档块会保留原始文档的元数据

    # 显示分割后的文档信息
    print("\n--- 文档块信息 ---")
    print(f"文档块总数: {len(docs)}")

    # 显示每个来源文件的文档块数量
    source_counts = {}
    for doc in docs:
        source = doc.metadata.get("source", "未知")
        source_counts[source] = source_counts.get(source, 0) + 1

    print("每个来源的文档块数量:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} 块")

    # 创建嵌入模型
    print("\n--- 创建嵌入模型 ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # 如果需要，可以更新为其他有效的嵌入模型
    print("\n--- 嵌入模型创建完成 ---")

    # 创建向量存储并持久化
    # 这个过程会为所有文档块生成嵌入向量，并保存到数据库中
    print("\n--- 创建并持久化向量存储 ---")
    db = Chroma.from_documents(
        docs,  # 包含元数据的文档块列表
        embeddings,  # 嵌入模型
        persist_directory=persistent_directory  # 持久化目录
    )
    print("\n--- 向量存储创建和持久化完成 ---")
    print(f"向量数据库已保存到: {persistent_directory}")

else:
    print("向量存储已存在。无需初始化。")
