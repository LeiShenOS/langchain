# RAG文本分割深度解析
# 本文件深入探讨了RAG系统中文本分割的各种策略和方法
# 文本分割是RAG系统的关键步骤，不同的分割策略会显著影响检索效果
#
# 文本分割的重要性：
# 1. 控制输入长度：确保文本块适合模型的输入限制
# 2. 保持语义完整性：避免在句子或段落中间分割
# 3. 优化检索精度：合适的块大小能提高检索相关性
# 4. 平衡上下文：在保持足够上下文和精确定位之间找到平衡

import os

# 导入各种文本分割器
from langchain.text_splitter import (
    CharacterTextSplitter,  # 基于字符数的分割器
    RecursiveCharacterTextSplitter,  # 递归字符分割器（推荐）
    SentenceTransformersTokenTextSplitter,  # 基于句子的分割器
    TextSplitter,  # 文本分割器基类
    TokenTextSplitter,  # 基于token的分割器
)
from langchain_community.document_loaders import TextLoader  # 文本加载器
from langchain_community.vectorstores import Chroma  # 向量数据库
from langchain_openai import OpenAIEmbeddings  # 嵌入模型

# 定义包含文本文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")  # 罗密欧与朱丽叶文本文件
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
print(f"文档总长度: {len(documents[0].page_content)} 字符")

# 定义嵌入模型
# 所有分割策略将使用相同的嵌入模型以便比较
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)  # 如果需要，可以更新为其他有效的嵌入模型


# 创建和持久化向量存储的辅助函数
def create_vector_store(docs, store_name):
    """
    创建向量存储的辅助函数

    参数:
    docs: 文档块列表
    store_name: 存储名称，用于区分不同分割策略的数据库
    """
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- 创建向量存储 {store_name} ---")
        print(f"文档块数量: {len(docs)}")

        # 显示文档块大小统计
        chunk_sizes = [len(doc.page_content) for doc in docs]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        max_size = max(chunk_sizes)
        min_size = min(chunk_sizes)
        print(f"平均块大小: {avg_size:.0f} 字符")
        print(f"最大块大小: {max_size} 字符")
        print(f"最小块大小: {min_size} 字符")

        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
        print(f"--- 完成创建向量存储 {store_name} ---")
    else:
        print(f"向量存储 {store_name} 已存在。无需初始化。")


# ========== 五种不同的文本分割策略比较 ==========

# 1. 基于字符数的分割（Character-based Splitting）
# 优点：简单直接，块大小一致
# 缺点：可能在句子中间分割，破坏语义完整性
# 适用场景：对块大小有严格要求，但对语义完整性要求不高的场景
print("\n--- 使用基于字符数的分割 ---")
char_splitter = CharacterTextSplitter(
    chunk_size=1000,  # 每个块最大1000个字符
    chunk_overlap=100  # 块之间重叠100个字符，保持上下文连续性
)
char_docs = char_splitter.split_documents(documents)
print(f"字符分割产生 {len(char_docs)} 个文档块")
create_vector_store(char_docs, "chroma_db_char")

# 2. 基于句子的分割（Sentence-based Splitting）
# 优点：保持句子完整性，语义连贯性好
# 缺点：块大小可能不均匀
# 适用场景：需要保持语义完整性的应用，如问答系统
print("\n--- 使用基于句子的分割 ---")
sent_splitter = SentenceTransformersTokenTextSplitter(
    chunk_size=1000  # 目标块大小（会尽量在句子边界分割）
)
sent_docs = sent_splitter.split_documents(documents)
print(f"句子分割产生 {len(sent_docs)} 个文档块")
create_vector_store(sent_docs, "chroma_db_sent")

# 3. 基于Token的分割（Token-based Splitting）
# 优点：精确控制token数量，适合transformer模型
# 缺点：可能在词汇中间分割，需要了解tokenizer
# 适用场景：需要精确控制输入token数的模型（如GPT系列）
print("\n--- 使用基于Token的分割 ---")
token_splitter = TokenTextSplitter(
    chunk_overlap=0,  # token分割通常不需要重叠
    chunk_size=512  # 每个块最大512个token（适合大多数transformer模型）
)
token_docs = token_splitter.split_documents(documents)
print(f"Token分割产生 {len(token_docs)} 个文档块")
create_vector_store(token_docs, "chroma_db_token")

# 4. 递归字符分割（Recursive Character-based Splitting）【推荐】
# 优点：在保持字符限制的同时，尽量在自然边界分割
# 分割优先级：段落 -> 句子 -> 单词 -> 字符
# 适用场景：大多数RAG应用的首选方案
print("\n--- 使用递归字符分割 ---")
rec_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 目标块大小
    chunk_overlap=100,  # 重叠大小
    # 默认分隔符优先级: ["\n\n", "\n", " ", ""]
    # 会优先在段落边界分割，然后是行边界，最后是单词边界
)
rec_char_docs = rec_char_splitter.split_documents(documents)
print(f"递归字符分割产生 {len(rec_char_docs)} 个文档块")
create_vector_store(rec_char_docs, "chroma_db_rec_char")

# 5. 自定义分割（Custom Splitting）
# 优点：完全控制分割逻辑，可以处理特殊格式
# 缺点：需要深入了解文档结构，开发复杂度高
# 适用场景：处理特殊格式文档（如剧本、诗歌、代码等）
print("\n--- 使用自定义分割 ---")


class CustomTextSplitter(TextSplitter):
    """
    自定义文本分割器示例
    这个例子按段落分割，适合处理结构化文本
    """
    def split_text(self, text):
        # 自定义分割逻辑：按双换行符（段落）分割
        paragraphs = text.split("\n\n")

        # 过滤掉空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # 可以添加更复杂的逻辑，比如：
        # - 合并过短的段落
        # - 分割过长的段落
        # - 基于特定标记分割（如场景标记）

        return paragraphs


custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
print(f"自定义分割产生 {len(custom_docs)} 个文档块")
create_vector_store(custom_docs, "chroma_db_custom")


# ========== 分割策略效果比较 ==========

# 查询向量存储的辅助函数
def query_vector_store(store_name, query):
    """
    查询特定向量存储并显示结果

    参数:
    store_name: 向量存储名称
    query: 查询问题
    """
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n{'='*60}")
        print(f"查询向量存储: {store_name}")
        print(f"{'='*60}")

        # 加载向量数据库
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )

        # 配置检索器
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 1,  # 只返回最相关的1个结果，便于比较
                "score_threshold": 0.1  # 较低的阈值确保能找到结果
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
                print(f"内容预览: {doc.page_content[:200]}...")
                if len(doc.page_content) > 200:
                    print(f"...（还有 {len(doc.page_content) - 200} 个字符）")

                if doc.metadata:
                    print(f"来源: {doc.metadata.get('source', '未知')}")
        else:
            print("未找到相关文档")

    else:
        print(f"向量存储 {store_name} 不存在。")


# 定义测试查询
# 这是一个关于《罗密欧与朱丽叶》的具体问题，用于比较不同分割策略的效果
query = "How did Juliet die?"  # 朱丽叶是怎么死的？
print(f"\n{'#'*80}")
print(f"测试查询: {query}")
print(f"{'#'*80}")

# 依次查询每个使用不同分割策略的向量存储
# 通过比较结果，可以观察不同分割策略对检索效果的影响

print("\n🔍 比较不同文本分割策略的检索效果:")

# 1. 字符分割结果
query_vector_store("chroma_db_char", query)

# 2. 句子分割结果
query_vector_store("chroma_db_sent", query)

# 3. Token分割结果
query_vector_store("chroma_db_token", query)

# 4. 递归字符分割结果（通常效果最好）
query_vector_store("chroma_db_rec_char", query)

# 5. 自定义分割结果
query_vector_store("chroma_db_custom", query)

print(f"\n{'#'*80}")
print("分割策略比较总结:")
print("1. 字符分割: 块大小一致，但可能破坏语义")
print("2. 句子分割: 保持语义完整，但块大小不均")
print("3. Token分割: 精确控制token数，适合特定模型")
print("4. 递归字符分割: 平衡了大小控制和语义完整性（推荐）")
print("5. 自定义分割: 针对特定文档结构优化")
print(f"{'#'*80}")

# 实际应用建议：
# - 对于大多数RAG应用，推荐使用RecursiveCharacterTextSplitter
# - 如果需要精确控制输入长度，使用TokenTextSplitter
# - 对于特殊格式文档，考虑自定义分割器
# - 始终根据具体应用场景测试和调优分割参数
