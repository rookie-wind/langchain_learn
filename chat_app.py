# 文件路径：chat_app.py
# 终端聊天助手：Agent + Chain + Tools + 多轮对话记忆
# 体验类似 TRAE 对话：流式打字输出、实时显示工具调用、记住上下文
#
# 架构：
#   你(终端输入) -> Agent(大脑,负责决策)
#                    ├─ tavily_search  工具1: 联网搜索
#                    └─ copywriting    工具2: 内部跑一条 chain 文案流水线
#   记忆: checkpointer 存档, thread_id 定位会话抽屉
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

# 防止 Windows 控制台打印特殊字符时崩溃
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

model = init_chat_model(
    "deepseek-chat",
    model_provider="openai",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    timeout=60,
)

# ---------- Chain 部分：一条文案流水线，稍后包装成工具 ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{style}风格的文案助手，只输出文案本身，不要多余解释"),
    ("user", "{topic}"),
])
writing_chain = prompt | model | StrOutputParser()


# ---------- Tools 部分：把 chain 包装成工具 + 联网搜索 ----------
@tool
def copywriting(topic: str, style: str = "正式") -> str:
    """根据主题和风格生成介绍文案。topic 是要介绍的对象，style 是风格（如幽默、正式、文艺）。"""
    return writing_chain.invoke({"style": style, "topic": topic})


search = TavilySearch(max_results=3)

# ---------- Agent 部分：挂上工具 + 记忆存档器 ----------
agent = create_agent(
    model,
    tools=[search, copywriting],
    checkpointer=InMemorySaver(),  # 对话存档器，配合 thread_id 实现多轮记忆
)

THREAD_ID = "trae-chat-001"  # 当前会话的"抽屉编号"


def reply(user_input: str):
    """流式回复：像 TRAE 一样边生成边打印，工具调用实时显示"""
    printed_text = False
    for chunk, _meta in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": THREAD_ID}},
        stream_mode="messages",  # 按消息粒度流式输出
    ):
        if isinstance(chunk, ToolMessage):
            # 模型调用的工具执行完毕
            print(f"\n  [工具 {chunk.name}] 执行完毕", flush=True)
            printed_text = False
        elif isinstance(chunk.content, str) and chunk.content:
            # AI 生成的文字，逐段打印（打字机效果）
            print(chunk.content, end="", flush=True)
            printed_text = True
    print()


# ---------- 终端对话主循环 ----------
print("=" * 56)
print("终端 AI 助手 (输入 exit / 退出 结束对话)")
print("能力: 联网搜索 | 文案生成 | 多轮记忆")
print("=" * 56)

while True:
    try:
        user_input = input("\n你 > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见!")
        break
    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit", "退出"}:
        print("再见!")
        break
    print("AI > ", end="", flush=True)
    reply(user_input)
