# 文件路径：web_search_demo.py
# 演示：给 DeepSeek 模型挂上 Tavily 联网搜索工具（Tool Calling）
# 模型自己决定何时调用搜索，再把搜索结果总结成回答
from dotenv import load_dotenv
load_dotenv()

import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

# 1. 准备模型（DeepSeek，OpenAI 兼容模式）
model = init_chat_model(
    "deepseek-chat",
    model_provider="openai",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    timeout=60,
)

# 2. 准备搜索工具：max_results 限制返回条数，搜索结果会自动提炼成适合 LLM 的摘要
search = TavilySearch(max_results=3)

# 3. 创建 Agent：模型 + 工具。模型会自主决定"要不要搜、搜什么"
agent = create_agent(model, tools=[search])

# 4. 提问 —— 这个问题不联网根本答不了
question = "新联电子（002546）今天的股价大概是多少？"
print(f"提问: {question}\n")

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"recursion_limit": 25},
)

# 5. 打印完整过程：能看到模型先发起工具调用、拿到结果、再总结回答
print("=" * 50)
for msg in result["messages"]:
    role = getattr(msg, "type", "?")
    if role == "ai" and msg.tool_calls:
        print(f"[模型] 决定调用工具: {[tc['name'] for tc in msg.tool_calls]}")
    elif role == "tool":
        print("[工具] 返回搜索结果（摘要）")
    elif role == "ai":
        print(f"[模型] 最终回答:\n{msg.content}")

print("\n" + "=" * 50)
print("最终答案:")
print(result["messages"][-1].content)
