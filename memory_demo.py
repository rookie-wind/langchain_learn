# 文件路径：memory_demo.py
# 演示：thread_id 如何让 Agent 拥有多轮对话记忆
# 原理：checkpointer 负责把每轮对话存档，thread_id 是档案柜的"抽屉编号"
from dotenv import load_dotenv
load_dotenv()

import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch

model = init_chat_model(
    "deepseek-chat",
    model_provider="openai",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    timeout=30,
)

# 关键区别 1：创建 agent 时挂上一个 checkpointer（存档器）
# InMemorySaver = 存在内存里，程序关了就没了
search = TavilySearch(max_results=3)
agent = create_agent(model, tools=[search], checkpointer=InMemorySaver())


def chat(text, thread_id):
    """和 agent 说一句话，thread_id 决定它"翻哪个抽屉"找历史"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},  # 关键区别 2
    )
    return result["messages"][-1].content


# ---- 会话 A（thread_id = "xiaoming"）：同一个抽屉，连续对话 ----
print("=== 会话 A（thread_id=xiaoming）===")
print("我:", "我叫小明，我最喜欢的数字是 7")
print("AI:", chat("我叫小明，我最喜欢的数字是 7", "xiaoming"))
print()
print("我:", "我叫什么名字？喜欢什么数字？")
print("AI:", chat("我叫什么名字？喜欢什么数字？", "xiaoming"))

# ---- 会话 B（thread_id = "stranger"）：换了个抽屉，AI 失忆了 ----
print("\n=== 会话 B（thread_id=stranger）===")
print("我:", "我叫什么名字？")
print("AI:", chat("我叫什么名字？", "stranger"))
