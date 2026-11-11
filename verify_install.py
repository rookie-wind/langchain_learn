# 文件路径：verify_install.py
# 验证 LangChain 安装和 API Key 配置
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
question = "南京新联电子是一家什么样的公司？"
print(f"提问: {question}\n")

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"recursion_limit": 25},
)

print("最终答案:")
print(result["messages"][-1].content)