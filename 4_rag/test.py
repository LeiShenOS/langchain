import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# 加载 .env 文件（指定路径，覆盖现有环境变量）
load_dotenv(dotenv_path=r"C:\Users\LeiShen\Desktop\learning_longchain\project_one\langchain-crash-course\.env", override=True)

# 从环境变量获取配置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

print(api_key)
print(base_url)
# 使用环境变量配置 OpenAIEmbeddings
emb = OpenAIEmbeddings(
    api_key=api_key,
    base_url=base_url
)
print(emb.embed_query("Hello world"))  # 看能否正常返回向量
