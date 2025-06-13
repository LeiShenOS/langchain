# 对话式RAG系统（Conversational RAG）
# 本文件演示了一个支持多轮对话的RAG系统
# 与单次问答不同，对话式RAG需要维护对话历史，并能理解上下文引用
#
# 对话式RAG的核心挑战：
# 1. 上下文理解：理解代词和引用（如"它"、"那个"、"刚才提到的"）
# 2. 历史维护：保持对话历史以提供连贯的体验
# 3. 查询重构：将依赖上下文的问题转换为独立的查询
# 4. 状态管理：在多轮对话中维护系统状态
#
# 系统架构：
# 用户输入 → 历史感知检索器 → 文档检索 → 答案生成 → 历史更新

import os

from dotenv import load_dotenv  # 环境变量加载
# LangChain链式组件
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma  # 向量数据库
from langchain_core.messages import HumanMessage, SystemMessage  # 消息类型
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 提示模板
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI模型
from dotenv import load_dotenv  # 加载环境变量
# 加载环境变量
load_dotenv(r'C:\Users\LeiShen\Desktop\learning_longchain\project_one\langchain-crash-course\.env',override=True)

print("🚀 初始化对话式RAG系统...")
# 定义持久化目录
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

print(f"📂 向量数据库路径: {persistent_directory}")

# 定义嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 加载现有的向量存储
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# 创建检索器
# 这是RAG系统的第一个组件：从向量数据库中检索相关文档
retriever = db.as_retriever(
    search_type="similarity",  # 使用相似度搜索
    search_kwargs={"k": 3},  # 检索前3个最相关的文档
)

print("✅ 检索器创建成功")

# 创建语言模型
# 这将用于两个目的：1) 重构查询 2) 生成最终答案
llm = ChatOpenAI(model="gpt-4o",)

# ========== 第一步：创建历史感知检索器 ==========

# 上下文化问题的系统提示
# 这个提示帮助AI理解如何基于对话历史重构问题，使其成为独立的查询
# 例如：用户问"它是什么时候写的？"，系统需要根据历史知道"它"指的是什么
contextualize_q_system_prompt = (
    "基于聊天历史和最新的用户问题，"
    "该问题可能引用了聊天历史中的上下文，"
    "请将其重新表述为一个独立的问题，"
    "使其在没有聊天历史的情况下也能被理解。"
    "不要回答问题，只需要在必要时重新表述，"
    "否则按原样返回。"
)

print("🔧 创建查询上下文化提示...")

# 创建用于上下文化问题的提示模板
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),  # 对话历史占位符
        ("human", "{input}"),  # 用户当前输入
    ]
)

# 创建历史感知检索器
# 这是对话式RAG的核心组件，它能够：
# 1. 理解当前问题在对话历史中的上下文
# 2. 将依赖上下文的问题重构为独立查询
# 3. 使用重构后的查询进行文档检索
history_aware_retriever = create_history_aware_retriever(
    llm,  # 用于重构查询的语言模型
    retriever,  # 基础检索器
    contextualize_q_prompt  # 上下文化提示
)

print("✅ 历史感知检索器创建成功")

# ========== 第二步：创建问答链 ==========

# 问答系统提示
# 这个提示定义了AI如何基于检索到的文档回答问题
qa_system_prompt = (
    "你是一个问答任务的助手。使用以下检索到的上下文片段来回答问题。"
    "如果你不知道答案，就说你不知道。"
    "最多使用三句话，保持答案简洁。"
    "\n\n"
    "{context}"  # 检索到的文档内容将插入这里
)

print("🔧 创建问答提示...")

# 创建用于回答问题的提示模板
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),  # 对话历史
        ("human", "{input}"),  # 用户问题
    ]
)

# 创建文档组合链
# 这个链负责将检索到的多个文档组合起来，并基于它们生成答案
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

print("✅ 问答链创建成功")

# ========== 第三步：创建完整的RAG链 ==========

# 创建检索链，将历史感知检索器和问答链组合起来
# 这是完整的对话式RAG流水线：
# 输入 → 查询重构 → 文档检索 → 答案生成 → 输出
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

print("🎯 完整的对话式RAG链创建成功")


# ========== 对话式交互函数 ==========

def continual_chat():
    """
    持续对话函数 - 对话式RAG系统的主要交互界面

    功能：
    1. 维护对话历史
    2. 处理用户输入
    3. 调用RAG链生成回答
    4. 更新对话状态
    """
    print("\n" + "="*60)
    print("🤖 对话式RAG系统已启动！")
    print("💬 开始与AI聊天吧！输入 'exit' 结束对话。")
    print("📚 系统已加载多个文档，可以回答相关问题。")
    print("🔄 支持多轮对话，能理解上下文引用。")
    print("="*60)

    # 初始化对话历史
    # 这是对话式RAG的关键：维护完整的对话上下文
    chat_history = []  # 存储对话历史的消息序列

    conversation_count = 0  # 对话轮次计数

    while True:
        # 获取用户输入
        print(f"\n[轮次 {conversation_count + 1}]")
        query = input("👤 您: ")

        # 检查退出条件
        if query.lower() in ["exit", "退出", "quit", "bye"]:
            print("👋 再见！感谢使用对话式RAG系统！")
            break

        # 检查空输入
        if not query.strip():
            print("⚠️  请输入有效的问题。")
            continue

        try:
            print("🔍 正在检索相关文档并生成回答...")

            # 通过RAG链处理用户查询
            # 这个过程包括：
            # 1. 基于历史重构查询（如果需要）
            # 2. 检索相关文档
            # 3. 生成基于文档的回答
            result = rag_chain.invoke({
                "input": query,  # 用户当前问题
                "chat_history": chat_history  # 完整对话历史
            })

            # 显示AI的回答
            print(f"🤖 AI: {result['answer']}")

            # 可选：显示检索到的文档信息（调试用）
            if 'context' in result:
                print(f"📄 参考了 {len(result['context'])} 个文档片段")

            # 更新对话历史
            # 这是维护对话连续性的关键步骤
            chat_history.append(HumanMessage(content=query))  # 添加用户消息
            chat_history.append(SystemMessage(content=result["answer"]))  # 添加AI回答

            conversation_count += 1

            # 可选：限制历史长度以避免上下文过长
            # 保留最近的10轮对话（20条消息）
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

        except Exception as e:
            print(f"❌ 处理查询时出错: {str(e)}")
            print("🔄 请重试或输入其他问题。")

    # 显示对话统计
    print(f"\n📊 对话统计:")
    print(f"  - 总轮次: {conversation_count}")
    print(f"  - 历史消息数: {len(chat_history)}")


# 主函数入口
if __name__ == "__main__":
    print("🎬 启动对话式RAG系统...")

    # 系统使用说明
    print("\n💡 使用提示:")
    print("1. 可以问关于已加载文档的任何问题")
    print("2. 支持多轮对话，可以使用'它'、'那个'等代词")
    print("3. 例如：先问'谁是奥德修斯？'，再问'他的妻子是谁？'")
    print("4. 系统会基于文档内容回答，确保答案的准确性")

    # 启动对话
    continual_chat()

    print("\n🏁 程序结束")

# 对话式RAG系统的优势：
# 1. 上下文连续性：能够理解代词和引用
# 2. 多轮交互：支持复杂的多步骤查询
# 3. 历史感知：基于对话历史提供更准确的回答
# 4. 用户友好：提供自然的对话体验

# 技术要点：
# 1. 历史感知检索：自动重构依赖上下文的查询
# 2. 状态管理：维护对话历史和系统状态
# 3. 错误处理：优雅处理各种异常情况
# 4. 性能优化：限制历史长度以控制计算成本
