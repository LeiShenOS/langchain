# RAG嵌入模型深度解析
# 本文件深入探讨了RAG系统中不同嵌入模型的使用和比较
# 嵌入模型是RAG系统的核心组件，负责将文本转换为向量表示
#
# 嵌入模型的重要性：
# 1. 语义理解：将文本转换为能够捕获语义信息的向量
# 2. 相似度计算：通过向量距离计算文本相似度
# 3. 检索质量：嵌入质量直接影响检索的准确性
# 4. 成本考虑：不同模型在成本、性能、部署方式上有显著差异

import os

# 导入不同的嵌入模型
from langchain.embeddings import HuggingFaceEmbeddings  # Hugging Face开源嵌入模型
from langchain.text_splitter import CharacterTextSplitter  # 文本分割器
from langchain_community.document_loaders import TextLoader  # 文本加载器
from langchain_community.vectorstores import Chroma  # 向量数据库
from langchain_openai import OpenAIEmbeddings  # OpenAI嵌入模型

# 定义文件路径和目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
file_path = os.path.join(current_dir, "books", "odyssey.txt")  # 奥德赛文本文件
db_dir = os.path.join(current_dir, "db")  # 数据库存储目录

# 检查文本文件是否存在
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"文件 {file_path} 不存在。请检查路径。"
    )

# 从文件中读取文本内容
loader = TextLoader(file_path)
documents = loader.load()
print(f"加载的文档数量: {len(documents)}")

# 将文档分割成块
# 使用统一的分割策略，以便公平比较不同嵌入模型的效果
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 显示分割后的文档信息
print("\n--- 文档块信息 ---")
print(f"文档块数量: {len(docs)}")
print(f"示例文档块:\n{docs[0].page_content[:200]}...\n")


# 创建和持久化向量存储的辅助函数
def create_vector_store(docs, embeddings, store_name):
    """
    创建向量存储的辅助函数

    参数:
    docs: 文档块列表
    embeddings: 嵌入模型实例
    store_name: 存储名称，用于区分不同嵌入模型的数据库
    """
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- 创建向量存储 {store_name} ---")
        print(f"使用嵌入模型: {type(embeddings).__name__}")

        # 创建向量数据库
        # 这个过程会为每个文档块生成嵌入向量
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"--- 完成创建向量存储 {store_name} ---")
        print(f"向量维度: {len(embeddings.embed_query('test'))} 维")
    else:
        print(f"向量存储 {store_name} 已存在。无需初始化。")


# ========== 不同嵌入模型的比较 ==========

# 1. OpenAI嵌入模型
# 优点：
# - 高质量的语义理解能力
# - 经过大规模数据训练，泛化能力强
# - API调用简单，无需本地部署
# - 持续更新和优化
# 缺点：
# - 需要付费使用（按token计费）
# - 依赖网络连接
# - 数据需要发送到OpenAI服务器
# 适用场景：对质量要求高，预算充足的商业应用
print("\n--- 使用OpenAI嵌入模型 ---")
print("特点：商业级质量，API调用，按使用付费")
print("定价信息：https://openai.com/api/pricing/")

openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"  # OpenAI的经典嵌入模型
    # 也可以使用更新的模型：
    # model="text-embedding-3-small"  # 更新、更便宜的小型模型
    # model="text-embedding-3-large"  # 最高质量的大型模型
)
create_vector_store(docs, openai_embeddings, "chroma_db_openai")

# 2. Hugging Face开源嵌入模型
# 优点：
# - 完全免费使用
# - 本地运行，数据隐私性好
# - 模型选择丰富，可针对特定领域优化
# - 支持离线使用
# 缺点：
# - 需要本地计算资源
# - 首次下载模型需要时间
# - 质量可能不如商业模型
# 适用场景：预算有限、对数据隐私要求高、或需要离线运行的应用
print("\n--- 使用Hugging Face开源嵌入模型 ---")
print("特点：免费开源，本地运行，保护数据隐私")
print("更多模型：https://huggingface.co/models?other=embeddings")

huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
    # 这是一个高质量的通用嵌入模型
    # 其他推荐模型：
    # "sentence-transformers/all-MiniLM-L6-v2"  # 更小更快的模型
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 多语言支持
    # "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"  # 基于DistilBERT
)
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")

print("\n✅ OpenAI和Hugging Face嵌入模型演示完成")

# 嵌入模型选择建议：
# 1. 如果预算充足且对质量要求高 → OpenAI嵌入模型
# 2. 如果需要控制成本或保护数据隐私 → Hugging Face模型
# 3. 如果需要特定领域优化 → 寻找专门的Hugging Face模型
# 4. 如果需要多语言支持 → 选择多语言嵌入模型


# ========== 嵌入模型效果比较 ==========

# 查询向量存储的辅助函数
def query_vector_store(store_name, query, embedding_function):
    """
    查询特定向量存储并显示结果

    参数:
    store_name: 向量存储名称
    query: 查询问题
    embedding_function: 嵌入函数（必须与创建时使用的相同）
    """
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n{'='*60}")
        print(f"查询向量存储: {store_name}")
        print(f"嵌入模型: {type(embedding_function).__name__}")
        print(f"{'='*60}")

        # 加载向量数据库
        # 注意：必须使用与创建时相同的嵌入函数
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )

        # 配置检索器
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,  # 返回前3个最相关的结果
                "score_threshold": 0.1  # 相似度阈值
            },
        )

        # 执行检索
        relevant_docs = retriever.invoke(query)

        # 显示检索结果
        if relevant_docs:
            print(f"找到 {len(relevant_docs)} 个相关文档")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"\n文档 {i}:")
                print(f"内容长度: {len(doc.page_content)} 字符")
                print(f"内容: {doc.page_content[:300]}...")
                if doc.metadata:
                    print(f"来源: {doc.metadata.get('source', '未知')}")
        else:
            print("未找到相关文档")

    else:
        print(f"向量存储 {store_name} 不存在。")


# 定义测试查询
# 使用关于奥德赛的问题来测试不同嵌入模型的检索效果
query = "Who is Odysseus' wife?"  # 奥德修斯的妻子是谁？

print(f"\n{'#'*80}")
print(f"测试查询: {query}")
print(f"比较不同嵌入模型的检索效果")
print(f"{'#'*80}")

# 使用不同嵌入模型查询相同问题，比较结果质量
print("\n🔍 嵌入模型效果比较:")

# 1. OpenAI嵌入模型结果
query_vector_store("chroma_db_openai", query, openai_embeddings)

# 2. Hugging Face嵌入模型结果
query_vector_store("chroma_db_huggingface", query, huggingface_embeddings)

print(f"\n{'#'*80}")
print("嵌入模型比较总结:")
print("1. OpenAI模型：通常提供更高质量的语义理解")
print("2. Hugging Face模型：免费且可本地运行，质量也很不错")
print("3. 选择建议：根据预算、隐私需求和质量要求来选择")
print("4. 性能测试：建议在实际数据上测试不同模型的效果")
print(f"{'#'*80}")

print("\n✅ 查询演示完成")

# 实际应用中的考虑因素：
# 1. 成本：OpenAI按使用量收费，Hugging Face免费但需要计算资源
# 2. 延迟：本地模型首次加载较慢，但后续查询更快
# 3. 质量：通常OpenAI模型质量更高，但差距在缩小
# 4. 隐私：本地模型不会将数据发送到外部服务器
# 5. 可定制性：开源模型可以针对特定领域进行微调
