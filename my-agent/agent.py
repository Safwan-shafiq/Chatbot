"""
LangChain Agent - Terminal Version
Groq + LangGraph (LangChain 1.x compatible)
Run: python agent.py
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv("../.env")

# ─── LLM ─────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

# ─── Tools ───────────────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Math calculator. Input: expression like 2+2, 10*5"""
    try:
        allowed = set("0123456789+-*/().% ")
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "Only basic math allowed"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No results found."
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search error: {str(e)}"

tools = [calculator, web_search]

# ─── Agent ───────────────────────────────────────────────────────────────────
agent = create_react_agent(llm, tools)

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🤖 LangChain Agent Ready! (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Agent: Khuda Hafiz!")
            break

        if not user_input:
            continue

        result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
        print(f"\nAgent: {result['messages'][-1].content}\n")
