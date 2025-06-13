# RAG基础示例 - 带元数据的多文档检索（第二部分）
# 本文件演示了如何从包含多个文档的向量数据库中检索信息
# 并展示元数据在检索结果中的重要作用
# 元数据帮助用户了解信息来源，提高检索结果的可信度和可追溯性

import os

# 导入必要的LangChain模块
from langchain_community.vectorstores import Chroma  # Chroma向量数据库
from langchain_openai import OpenAIEmbeddings  # OpenAI嵌入模型

# 定义持久化目录
# 这个目录应该包含在2a_rag_basics_metadata.py中创建的带元数据的向量数据库
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本目录
db_dir = os.path.join(current_dir, "db")  # 数据库目录
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")  # 带元数据的向量数据库路径

# 定义嵌入模型
# 必须使用与创建向量数据库时相同的嵌入模型，确保向量空间一致性
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 加载现有的向量存储
# 这里加载的是包含多个文档和元数据的向量数据库
db = Chroma(
    persist_directory=persistent_directory,  # 指定数据库存储位置
    embedding_function=embeddings  # 指定嵌入函数
)

# 定义用户的问题
# 这是一个关于莎士比亚《罗密欧与朱丽叶》的问题，用于测试多文档检索
query = "How did Juliet die?"  # 朱丽叶是怎么死的？

# 基于查询检索相关文档
# 配置检索器以获得最佳的检索效果
retriever = db.as_retriever(
    search_type="similarity_score_threshold",  # 使用相似度分数阈值搜索
    search_kwargs={
        "k": 3,  # 最多返回3个最相关的文档
        "score_threshold": 0.1  # 相似度分数阈值设为0.1（较低的阈值，允许更多结果）
        # 注意：这里使用较低的阈值是为了确保能找到相关结果
        # 在实际应用中，可能需要根据具体情况调整这个值
    },
)

# 执行检索操作
# 系统会在所有文档中搜索与查询最相关的内容
relevant_docs = retriever.invoke(query)

# 显示检索到的相关文档及其元数据
# 元数据信息对于多文档检索特别重要，因为它告诉用户信息来自哪个文档
print("\n--- 相关文档 ---")
print(f"找到 {len(relevant_docs)} 个相关文档")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\n文档 {i}:")
    print(f"内容:\n{doc.page_content}\n")

    # 显示文档来源信息
    # 这是元数据的核心价值：提供信息溯源
    print(f"来源: {doc.metadata['source']}")

    # 如果有其他元数据，也可以显示
    if len(doc.metadata) > 1:
        print("其他元数据:")
        for key, value in doc.metadata.items():
            if key != 'source':  # 避免重复显示source
                print(f"  {key}: {value}")

    print("-" * 50)  # 分隔线，便于阅读

# 总结检索结果
print(f"\n检索总结:")
print(f"查询: {query}")
print(f"找到相关文档数量: {len(relevant_docs)}")

# 统计来源分布
sources = [doc.metadata['source'] for doc in relevant_docs]
source_counts = {}
for source in sources:
    source_counts[source] = source_counts.get(source, 0) + 1

print("结果来源分布:")
for source, count in source_counts.items():
    print(f"  {source}: {count} 个文档块")
