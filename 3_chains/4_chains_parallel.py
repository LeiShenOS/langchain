from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
load_dotenv(override=True)

# 创建 ChatOpenAI 模型
model = ChatOpenAI(model="gpt-4o",base_url='https://api.chatanywhere.tech/v1')

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位专业的产品评测专家。"),
        ("human", "请列出产品 {product_name} 的主要特性。"),
    ]
)


# 定义优点分析步骤
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位专业的产品评测专家。"),
            (
                "human",
                "基于这些特性：{features}，请列出这些特性的优点。",
            ),
        ]
    )
    return pros_template.format_prompt(features=features)


# 定义缺点分析步骤
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位专业的产品评测专家。"),
            (
                "human",
                "基于这些特性：{features}，请列出这些特性的缺点。",
            ),
        ]
    )
    return cons_template.format_prompt(features=features)


# 将优点和缺点合并为最终评测
def combine_pros_cons(pros, cons):
    return f"优点：\n{pros}\n\n缺点：\n{cons}"


# 使用 LCEL 简化分支
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# 使用 LangChain 表达式语言 (LCEL) 创建组合链
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: print("最终输出", x) or combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# 运行链
result = chain.invoke({"product_name": "MacBook Pro"})

# 输出结果
print(result)
