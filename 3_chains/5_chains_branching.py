from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
load_dotenv(override=True)

# 创建 ChatOpenAI 模型
model = ChatOpenAI(model="gpt-4o",base_url='https://api.chatanywhere.tech/v1')

# 为不同反馈类型定义提示模板
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有用的助手。"),
        ("human",
         "为这个正面反馈生成一份感谢信：{feedback}。"),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有用的助手。"),
        ("human",
         "为这个负面反馈生成一个回应：{feedback}。"),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有用的助手。"),
        (
            "human",
            "为这个中性反馈生成一个请求更多详细信息的回复：{feedback}。",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有用的助手。"),
        (
            "human",
            "生成一条消息将此反馈升级给人工客服：{feedback}。",
        ),
    ]
)

# 定义反馈分类模板
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有用的助手。"),
        ("human",
         "将此反馈的情感分类为正面、负面、中性或升级：{feedback}。"),
    ]
)

# 定义处理反馈的可运行分支
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()  # 正面反馈链
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()  # 负面反馈链
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()  # 中性反馈链
    ),
    escalate_feedback_template | model | StrOutputParser()
)

# 创建分类链
classification_chain = classification_template | model | StrOutputParser()

# 将分类和响应生成合并为一个链
chain = classification_chain | branches

# 使用示例评论运行链
# 好评 - "这个产品非常棒。我真的很喜欢使用它，发现它非常有用。"
# 差评 - "这个产品很糟糕。只用了一次就坏了，质量很差。"
# 中性评论 - "这个产品还可以。它按预期工作，但没有什么特别的。"
# 默认 - "我还不确定这个产品。你能告诉我更多关于它的功能和优点吗？"

review = "这个产品还可以。它按预期工作，但没有什么特别的。"
result = chain.invoke({"feedback": review})

# 输出结果
print(result)
