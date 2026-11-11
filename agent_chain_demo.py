# 文件路径：agent_chain_demo.py
# 演示：把 Chain 包装成工具，交给 Agent 自主调用
# Agent 像主管，Chain 像流水线工人：主管决定"先搜资料，再让流水线写文案"
from dotenv import load_dotenv
load_dotenv()

import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

# 准备模型
model = init_chat_model(
    "deepseek-chat",
    model_provider="openai",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    timeout=60,
)

# 1. 先搭一条普通 chain：文案生成流水线
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{style}风格的文案助手，只输出一句话文案"),
    ("user", "介绍{topic}"),
])
chain = prompt | model | StrOutputParser()

# 2. 用 @tool 装饰器把 chain 包装成工具
#    函数名 = 工具名，docstring = 给模型看的"使用说明书"，模型靠它判断何时调用
@tool
def copywriting(topic: str, style: str = "幽默") -> str:
    """根据主题和风格生成一句介绍文案。topic 是要介绍的对象，style 是文案风格。"""
    return chain.invoke({"style": style, "topic": topic})

# 3. Agent 现在有两个工具：联网搜索 + 文案流水线
agent = create_agent(model, tools=[TavilySearch(max_results=3), copywriting])

# 4. 下达一个组合任务：需要"先查资料，再写文案"
question = "先上网搜一下新联电子（002546）是做什么业务的，然后基于搜索结果，用幽默风格为它写一句介绍文案"
print(f"任务: {question}\n")

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"recursion_limit": 25},
)

# 5. 展示 Agent 的决策过程
for msg in result["messages"]:
    role = getattr(msg, "type", "?")
    if role == "ai" and msg.tool_calls:
        print(f"[Agent] 决定调用: {[tc['name'] for tc in msg.tool_calls]}")
    elif role == "tool":
        print(f"[工具 {msg.name}] 执行完毕")
    elif role == "ai":
        print(f"[Agent] 最终产出:\n{msg.content}")
