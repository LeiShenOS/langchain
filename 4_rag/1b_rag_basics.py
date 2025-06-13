# RAG（检索增强生成）基础示例 - 第二部分：向量检索
# 本文件演示了如何从已创建的向量数据库中检索相关文档
# 这是RAG系统的第二个核心步骤：基于用户查询检索最相关的文档片段

import os
from dotenv import find_dotenv, load_dotenv
# 导入必要的LangChain模块
from langchain_community.vectorstores import Chroma  # Chroma向量数据库
from langchain_openai import OpenAIEmbeddings  # OpenAI嵌入模型
load_dotenv(dotenv_path=r"C:\Users\LeiShen\Desktop\learning_longchain\project_one\langchain-crash-course\.env", override=True,verbose=False)

# 从环境变量获取配置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
# 定义持久化目录
# 这个目录应该包含之前在1a_rag_basics.py中创建的向量数据库
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本目录
persistent_directory = os.path.join(current_dir, "db", "chroma_db")  # 向量数据库存储路径

# 定义嵌入模型
# 必须使用与创建向量数据库时相同的嵌入模型，以确保向量空间的一致性
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 加载现有的向量存储
# 这里不是创建新的数据库，而是加载之前已经创建并持久化的数据库
db = Chroma(
    persist_directory=persistent_directory,  # 指定数据库存储位置
    embedding_function=embeddings  # 指定嵌入函数，用于将查询转换为向量
)

# 定义用户的问题
# 这是一个关于奥德赛内容的具体问题，用于测试检索系统的效果
query = "Who is Odysseus' wife?"  # 奥德修斯的妻子是谁？

# 基于查询检索相关文档
# 创建检索器对象，配置检索参数以获得最佳检索效果
retriever = db.as_retriever(
    search_type="similarity_score_threshold",  # 使用相似度分数阈值搜索
    # 这种搜索类型会：
    # 1. 计算查询向量与数据库中所有文档向量的相似度分数
    # 2. 只返回相似度分数超过指定阈值的文档
    # 3. 确保检索结果的质量和相关性
    search_kwargs={
        "k": 3,  # 最多返回3个最相关的文档
        "score_threshold": 0.4  # 相似度分数阈值为0.9（范围0-1，越高越严格）
    },
)
# 执行检索操作
# invoke方法会：
# 1. 将查询文本转换为向量嵌入
# 2. 在向量数据库中搜索最相似的文档
# 3. 返回满足条件的文档列表
relevant_docs = retriever.invoke(query)

# 显示检索到的相关文档及其元数据
print("\n--- 相关文档 ---")
# 遍历检索到的每个文档
for i, doc in enumerate(relevant_docs, 1):
    print(f"文档 {i}:\n{doc.page_content}\n")  # 显示文档内容
    # 如果文档包含元数据（如来源信息），则显示出来
    if doc.metadata:
        print(f"来源: {doc.metadata.get('source', '未知')}\n")
        # 元数据可能包含：
        # - source: 文档来源文件路径
        # - page: 页码信息
        # - chunk_id: 文档块标识符
        # - 其他自定义元数据
