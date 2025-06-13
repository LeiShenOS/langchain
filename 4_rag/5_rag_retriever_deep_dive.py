# RAG检索器深度解析
# 本文件深入探讨了RAG系统中不同检索策略的使用和比较
# 检索器是RAG系统的核心组件，负责从向量数据库中找到最相关的文档
#
# 检索策略的重要性：
# 1. 相关性：确保检索到的文档与查询高度相关
# 2. 多样性：避免检索到过于相似的重复信息
# 3. 质量控制：通过阈值过滤低质量结果
# 4. 性能优化：平衡检索质量和计算效率

import os

from dotenv import load_dotenv  # 加载环境变量
from langchain_community.vectorstores import Chroma  # Chroma向量数据库
from langchain_openai import OpenAIEmbeddings  # OpenAI嵌入模型

# 加载环境变量（如API密钥）
load_dotenv()

# 定义持久化目录
# 使用之前创建的包含元数据的向量数据库
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
db_dir = os.path.join(current_dir, "db")  # 数据库目录
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")  # 带元数据的数据库路径

# 定义嵌入模型
# 必须与创建向量数据库时使用的模型一致
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 加载现有的向量存储
# 这个数据库包含多个文档和相应的元数据
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)
print(f"✅ 成功加载向量数据库: {persistent_directory}")


# 查询向量存储的通用函数，支持不同的搜索类型和参数
def query_vector_store(store_name, query, embedding_function, search_type, search_kwargs):
    """
    使用不同搜索策略查询向量存储

    参数:
    store_name: 存储名称（用于显示）
    query: 查询问题
    embedding_function: 嵌入函数
    search_type: 搜索类型（similarity, mmr, similarity_score_threshold）
    search_kwargs: 搜索参数字典
    """
    if os.path.exists(persistent_directory):
        print(f"\n{'='*70}")
        print(f"查询向量存储: {store_name}")
        print(f"搜索类型: {search_type}")
        print(f"搜索参数: {search_kwargs}")
        print(f"{'='*70}")

        # 加载向量数据库
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )

        # 创建检索器，配置特定的搜索策略
        retriever = db.as_retriever(
            search_type=search_type,  # 搜索类型
            search_kwargs=search_kwargs,  # 搜索参数
        )

        # 执行检索
        relevant_docs = retriever.invoke(query)

        # 显示检索结果
        if relevant_docs:
            print(f"找到 {len(relevant_docs)} 个相关文档")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"\n📄 文档 {i}:")
                print(f"长度: {len(doc.page_content)} 字符")
                print(f"内容: {doc.page_content[:200]}...")
                if doc.metadata:
                    print(f"来源: {doc.metadata.get('source', '未知')}")
                print("-" * 50)
        else:
            print("❌ 未找到相关文档")
    else:
        print(f"❌ 向量存储 {store_name} 不存在。")


# ========== 三种主要检索策略的比较 ==========

# 定义测试查询
# 使用关于朱丽叶的问题来测试不同检索策略的效果
query = "How did Juliet die?"  # 朱丽叶是怎么死的？

print(f"\n{'#'*80}")
print(f"测试查询: {query}")
print(f"比较不同检索策略的效果")
print(f"{'#'*80}")

# 展示不同的检索方法

# 1. 相似度搜索（Similarity Search）
# 工作原理：
# - 计算查询向量与所有文档向量的余弦相似度
# - 返回相似度最高的前k个文档
# - 这是最基础和常用的检索方法
# 优点：简单直接，计算效率高
# 缺点：可能返回内容相似的重复文档
# 适用场景：大多数基础RAG应用，对多样性要求不高的场景
print("\n🔍 1. 相似度搜索（Similarity Search）")
print("特点：返回与查询最相似的文档，简单高效")
print("参数：k=3（返回前3个最相似的文档）")
query_vector_store(
    "chroma_db_with_metadata",
    query,
    embeddings,
    "similarity",
    {"k": 3}  # 返回前3个最相似的文档
)

# 2. 最大边际相关性（Max Marginal Relevance, MMR）
# 工作原理：
# - 首先获取fetch_k个最相似的文档
# - 然后在相关性和多样性之间找平衡
# - lambda_mult控制平衡：1.0完全基于相关性，0.0完全基于多样性
# - 逐个选择文档，确保既相关又不重复
# 优点：避免信息冗余，提供多样化的结果
# 缺点：计算复杂度较高，可能牺牲一些相关性
# 适用场景：需要多样化信息的应用，如综合性问答、研究辅助
print("\n🎯 2. 最大边际相关性（MMR - Max Marginal Relevance）")
print("特点：平衡相关性和多样性，避免重复信息")
print("参数：k=3, fetch_k=20, lambda_mult=0.5")
print("  - fetch_k=20: 先获取20个候选文档")
print("  - lambda_mult=0.5: 相关性和多样性各占50%")
print("  - k=3: 最终返回3个文档")
query_vector_store(
    "chroma_db_with_metadata",
    query,
    embeddings,
    "mmr",
    {
        "k": 3,  # 最终返回的文档数量
        "fetch_k": 20,  # 初始获取的候选文档数量
        "lambda_mult": 0.5  # 相关性vs多样性的权重（0.0-1.0）
        # lambda_mult = 1.0: 完全基于相关性（等同于similarity search）
        # lambda_mult = 0.0: 完全基于多样性（可能牺牲相关性）
        # lambda_mult = 0.5: 平衡相关性和多样性（推荐值）
    },
)

# 3. 相似度分数阈值（Similarity Score Threshold）
# 工作原理：
# - 计算查询与所有文档的相似度分数
# - 只返回分数超过指定阈值的文档
# - 可以设置最大返回数量k作为上限
# 优点：确保结果质量，过滤不相关文档
# 缺点：可能返回很少或没有结果（如果阈值设置过高）
# 适用场景：对结果质量要求严格的应用，宁缺毋滥的场景
print("\n⚡ 3. 相似度分数阈值（Similarity Score Threshold）")
print("特点：只返回相似度超过阈值的文档，确保结果质量")
print("参数：k=3, score_threshold=0.1")
print("  - score_threshold=0.1: 只返回相似度>0.1的文档")
print("  - k=3: 最多返回3个文档（如果有足够多超过阈值的文档）")
query_vector_store(
    "chroma_db_with_metadata",
    query,
    embeddings,
    "similarity_score_threshold",
    {
        "k": 3,  # 最大返回文档数量
        "score_threshold": 0.1  # 相似度分数阈值（0.0-1.0）
        # 阈值设置建议：
        # 0.0-0.3: 宽松，可能包含不太相关的文档
        # 0.3-0.7: 中等，平衡相关性和召回率
        # 0.7-1.0: 严格，只返回高度相关的文档
    },
)

print(f"\n{'#'*80}")
print("检索策略比较总结:")
print("1. Similarity Search: 基础策略，适合大多数场景")
print("2. MMR: 需要多样化结果时的最佳选择")
print("3. Score Threshold: 对结果质量要求严格时使用")
print("4. 实际应用中可以根据具体需求组合使用这些策略")
print(f"{'#'*80}")

print("\n✅ 不同搜索类型的查询演示完成")

# 检索策略选择指南：
# 1. 基础问答系统 → Similarity Search
# 2. 研究和分析工具 → MMR（避免信息重复）
# 3. 高精度应用 → Similarity Score Threshold
# 4. 复杂应用 → 可以组合多种策略，如先用MMR获取多样化结果，再用阈值过滤
