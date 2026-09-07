"""
LangChain Agent - Streamlit Web UI
LangChain 1.x + LangGraph compatible
Run: streamlit run streamlit_app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Keys load karo — local mein .env se, Streamlit Cloud mein st.secrets se
load_dotenv("../.env")
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="My AI Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 My LangChain AI Agent")
st.caption("Powered by Groq + LangGraph")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    groq_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password"
    )

    model_choice = st.selectbox(
        "Model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

    st.divider()
    st.markdown("**Available Tools:**")
    st.markdown("- 🔍 Web Search (DuckDuckGo)")
    st.markdown("- 🧮 Calculator")
    st.markdown("- 📝 Text Tools")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ─── Tools ────────────────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Simple math calculator.
    Use this for any math calculations.
    Input: math expression like '2+2', '10*5', '100/4', '2**10'
    """
    try:
        # Safe eval - sirf numbers aur operators allow
        allowed = set("0123456789+-*/().% ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"Result: {result}"
        else:
            return "Only basic math allowed: + - * / ( ) ."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    Use this when you need to find recent news, facts, or any online information.
    Input: search query string
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No results found."
            output = ""
            for r in results:
                output += f"**{r['title']}**\n{r['body']}\n\n"
            return output
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def word_counter(text: str) -> str:
    """
    Count words and characters in a text.
    Input: any text string
    """
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    return f"Words: {words} | Characters: {chars} | Sentences: {sentences}"

tools = [calculator, web_search, word_counter]

# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Agent Builder ────────────────────────────────────────────────────────────
def get_agent(api_key: str, model: str, temp: float):
    llm = ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temp,
    )
    agent = create_react_agent(llm, tools)
    return agent

# ─── Chat History Display ─────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── Chat Input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Kuch bhi poocho... (Urdu ya English)"):

    if not groq_key:
        st.error("⚠️ Pehle sidebar mein Groq API Key daalo!")
        st.stop()

    # User message show karo
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Soch raha hun..."):
            try:
                agent = get_agent(groq_key, model_choice, temperature)

                # Chat history LangGraph format mein
                history = []
                for msg in st.session_state.messages[:-1]:  # last message skip (current)
                    if msg["role"] == "user":
                        history.append(HumanMessage(content=msg["content"]))
                    else:
                        history.append(AIMessage(content=msg["content"]))
                history.append(HumanMessage(content=user_input))

                result = agent.invoke({"messages": history})
                answer = result["messages"][-1].content

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
