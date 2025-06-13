# 完整的RAG单次问答系统
# 本文件演示了一个完整的RAG（检索增强生成）工作流程
# 这是将之前学习的所有组件整合在一起的实际应用示例
#
# RAG系统的完整流程：
# 1. 加载向量数据库（之前创建的包含多个文档的数据库）
# 2. 接收用户查询
# 3. 从向量数据库中检索相关文档
# 4. 将检索到的文档与用户查询结合
# 5. 使用大语言模型生成基于检索内容的回答
# 6. 返回最终答案

import os

from dotenv import load_dotenv  # 加载环境变量
from langchain_community.vectorstores import Chroma  # 向量数据库
from langchain_core.messages import HumanMessage, SystemMessage  # 消息类型
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI模型

# 加载环境变量（包含API密钥等配置）
load_dotenv()

# 定义持久化目录
# 使用之前创建的包含多个文档和元数据的向量数据库
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

print(f"🗂️  加载向量数据库: {persistent_directory}")

# 定义嵌入模型
# 必须与创建向量数据库时使用的模型一致
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 加载现有的向量存储
# 这个数据库包含了多个文档的向量嵌入
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

print("✅ 向量数据库加载成功")

# 定义用户的问题
# 这是一个关于LangChain学习的问题，用于测试RAG系统
query = "Waht is LangChain?"
print(f"\n❓ 用户问题: {query}")

# 第一步：检索相关文档
# 基于用户查询从向量数据库中检索最相关的文档
print("\n🔍 步骤1: 检索相关文档...")
retriever = db.as_retriever(
    search_type="similarity",  # 使用相似度搜索
    search_kwargs={"k": 1},  # 只检索最相关的1个文档
)
relevant_docs = retriever.invoke(query)

# 显示检索到的相关文档
print(f"📄 找到 {len(relevant_docs)} 个相关文档")
print("\n--- 检索到的相关文档 ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"文档 {i}:")
    print(f"长度: {len(doc.page_content)} 字符")
    print(f"内容: {doc.page_content[:200]}...")
    if hasattr(doc, 'metadata') and doc.metadata:
        print(f"来源: {doc.metadata.get('source', '未知')}")
    print()

# 第二步：构建增强提示
# 将用户查询和检索到的文档内容结合，形成给LLM的完整提示
print("🔧 步骤2: 构建增强提示...")
combined_input = (
    "以下是一些可能有助于回答问题的文档: "
    + query
    + "\n\n相关文档:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\n请仅基于提供的文档回答问题。如果文档中没有找到答案，请回复'我不确定'。"
)

print(f"📝 增强提示长度: {len(combined_input)} 字符")

# 第三步：创建语言模型
# 使用GPT-4来生成最终答案
print("\n🤖 步骤3: 创建语言模型...")
model = ChatOpenAI(
    model="gpt-4o",  # 使用GPT-4o模型
    temperature=0  # 设置为0以获得更确定性的回答
)

# 第四步：构建消息
# 定义系统消息和用户消息
messages = [
    SystemMessage(content="你是一个有用的助手，专门基于提供的文档回答问题。"),
    HumanMessage(content=combined_input),
]

print("💬 消息构建完成")

# 第五步：生成回答
# 调用语言模型生成基于检索内容的回答
print("\n⚡ 步骤4: 生成回答...")
result = model.invoke(messages)

# 显示最终结果
print("\n" + "="*60)
print("🎯 RAG系统生成的回答")
print("="*60)
print(result.content)
print("="*60)

# RAG系统的优势：
# 1. 基于事实：回答基于实际文档内容，减少幻觉
# 2. 可追溯：可以追踪答案来源
# 3. 实时更新：通过更新文档库来更新知识
# 4. 领域专业：可以针对特定领域构建专业知识库

print(f"\n✅ RAG问答流程完成")
print(f"📊 处理统计:")
print(f"  - 检索文档数: {len(relevant_docs)}")
print(f"  - 输入长度: {len(combined_input)} 字符")
print(f"  - 输出长度: {len(result.content)} 字符")
