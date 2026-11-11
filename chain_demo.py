# 文件路径：chain_demo.py
# Chain 基础用法：提示词模板 | 模型 | 输出解析器
from dotenv import load_dotenv
load_dotenv()

import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 准备模型
model = init_chat_model(
    "deepseek-chat",
    model_provider="openai",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    timeout=60,
)

# 2. 提示词模板：{变量} 由 invoke 时传入
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{style}风格的文案助手"),
    ("user", "用一句话介绍{topic}"),
])

# 3. 串联：prompt 的输出给 model，model 的输出给 parser
chain = prompt | model | StrOutputParser()

# 4. 使用：传入模板变量，直接得到字符串
result = chain.invoke({"style": "幽默", "topic": "美国"})
print("invoke 结果:")
print(result)

# 5. 流式输出：逐字打印，适合做打字机效果
print("\nstream 结果:")
for chunk in chain.stream({"style": "正式", "topic": "Python"}):
    print(chunk, end="", flush=True)

# ============ 第二部分：给 AI 加上联网搜索能力 ============
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

# 6. 准备搜索工具：模型可以随时用它上网查资料
search = TavilySearch(max_results=3)

# 7. 创建 Agent：普通模型只会用脑子里的旧知识，
#    装上工具后，遇到不知道的事它会自己上网搜
agent = create_agent(model, tools=[search])

# 8. 问一个"不联网就答不了"的问题
question = "新联电子（002546）今天的股价大概是多少？"
print(f"\n\n提问: {question}\n")

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"recursion_limit": 25},
)

# 9. 展示完整过程：模型先调搜索工具，拿到结果再总结回答
for msg in result["messages"]:
    role = getattr(msg, "type", "?")
    if role == "ai" and msg.tool_calls:
        print(f"[模型] 我需要搜索: {[tc['name'] for tc in msg.tool_calls]}")
    elif role == "tool":
        print("[工具] 搜索完成，返回结果")
    elif role == "ai":
        print(f"[模型] 最终回答:\n{msg.content}")
