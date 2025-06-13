# RAG嵌入成本计算器
# 本工具用于估算使用OpenAI嵌入API处理文档的成本
# 在构建RAG系统时，了解成本是非常重要的，特别是处理大量文档时
#
# 成本计算的重要性：
# 1. 预算规划：提前了解处理文档的成本
# 2. 优化决策：帮助选择合适的文档处理策略
# 3. 成本控制：避免意外的高额API费用
# 4. 方案比较：比较不同嵌入模型的成本效益

import os

import tiktoken  # OpenAI的tokenizer库，用于准确计算token数量

# 定义要分析的文档文件路径
# 这里使用奥德赛文本作为示例，你可以替换为任何其他文档
file_path = os.path.join(os.path.dirname(__file__), "..", "books", "odyssey.txt")

# 检查文件是否存在
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"文件 {file_path} 不存在。请检查路径。"
    )

print(f"📄 分析文件: {os.path.basename(file_path)}")
print(f"📁 文件路径: {file_path}")

# 读取文件内容
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print(f"📊 文件大小: {len(text)} 字符")

# 初始化tokenizer
# cl100k_base是GPT-4和text-embedding-ada-002使用的编码
tokenizer = tiktoken.get_encoding(
    "cl100k_base"  # 使用与目标模型相匹配的编码
    # 不同模型使用不同的编码：
    # - GPT-4, GPT-3.5-turbo, text-embedding-ada-002: cl100k_base
    # - GPT-3: p50k_base
    # - Codex: p50k_base
)

# 对文本进行tokenization并计算token数量
tokens = tokenizer.encode(text)
total_tokens = len(tokens)

print(f"🔢 总token数量: {total_tokens:,}")

# 基于OpenAI的定价计算成本
# 注意：价格可能会变化，请查看最新的OpenAI定价页面
# https://openai.com/api/pricing/

# text-embedding-ada-002的定价（截至2024年）
cost_per_million_tokens = 0.02  # 每百万token $0.02

# 计算总成本
cost = (total_tokens / 1_000_000) * cost_per_million_tokens

# 显示结果
print(f"\n{'='*50}")
print(f"💰 成本估算报告")
print(f"{'='*50}")
print(f"模型: text-embedding-ada-002")
print(f"定价: ${cost_per_million_tokens} / 百万tokens")
print(f"文档tokens: {total_tokens:,}")
print(f"估算成本: ${cost:.6f}")

# 提供不同规模的成本参考
if cost < 0.01:
    print(f"💡 成本评估: 非常低（< $0.01）")
elif cost < 0.1:
    print(f"💡 成本评估: 较低（< $0.10）")
elif cost < 1.0:
    print(f"💡 成本评估: 中等（< $1.00）")
else:
    print(f"💡 成本评估: 较高（≥ $1.00）")

# 提供优化建议
print(f"\n📋 优化建议:")
if total_tokens > 1_000_000:
    print("- 考虑将大文档分批处理")
    print("- 评估是否需要处理整个文档")
    print("- 考虑使用更便宜的开源嵌入模型")
else:
    print("- 文档大小合理，可以直接处理")
    print("- 成本在可接受范围内")

print(f"\n🔗 最新定价信息: https://openai.com/api/pricing/")
print(f"⚠️  注意: 价格可能会变化，请查看官方最新定价")
