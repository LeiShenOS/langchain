from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv(override=True)

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o",base_url='https://api.chatanywhere.tech/v1')

# PART 1: Create a ChatPromptTemplate using a template string
print("-----Prompt from Template-----")
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "cats"})
result = model.invoke(prompt)
print(result.content)

# PART 2: Prompt with Multiple Placeholders
print("\n----- Prompt with Multiple Placeholders -----\n")
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} short story about a {animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})

result = model.invoke(prompt)
print(result.content)

# PART 3: Prompt with System and Human Messages (Using Tuples)
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
messages = [
    ("system", "你是一位讲笑话的喜剧演员，你的笑话内容是关于 {topic}."),
    ("human", "告诉我 {joke_count} 笑话."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "数学家", "joke_count": 2})
result = model.invoke(prompt)
print(result.content)
