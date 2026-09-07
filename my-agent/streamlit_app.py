import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv("../.env")
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

st.set_page_config(
    page_title="My AI Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Icons aur Streamlit floating button hide karo
st.markdown("""
<style>
[data-testid="chatAvatarIcon-user"]      { display: none !important; }
[data-testid="chatAvatarIcon-assistant"] { display: none !important; }
[data-testid="stStatusWidget"]           { display: none !important; }
.stDeployButton                          { display: none !important; }
#streamlit-badge-overflow-button         { display: none !important; }
.viewerBadge_container__r5tak           { display: none !important; }
.viewerBadge_link__qRIco                { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    groq_key = st.text_input(
        "Groq API Key (optional — apni key daalo)",
        value="",
        placeholder="gsk_...",
        type="password"
    )
    # Agar user ny khud key nahi daali to background mein secrets wali use ho
    if not groq_key:
        groq_key = os.getenv("GROQ_API_KEY", "")

    model_choice = st.selectbox(
        "Model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

    st.divider()
    st.markdown("**Available Tools:**")
    st.markdown("- Web Search (DuckDuckGo)")
    st.markdown("- Calculator")
    st.markdown("- Word Counter")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── TOOLS ─────────────────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Math calculator. Input: math expression like 2+2, 10*5"""
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

@tool
def word_counter(text: str) -> str:
    """Count words and characters in a text."""
    return f"Words: {len(text.split())} | Characters: {len(text)}"

tools = [calculator, web_search, word_counter]

# ── AGENT ─────────────────────────────────────────────────────────────────────
def get_agent(api_key, model, temp):
    llm = ChatGroq(model=model, api_key=api_key, temperature=temp)
    return create_react_agent(llm, tools)

# ── SESSION ───────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── TITLE ─────────────────────────────────────────────────────────────────────
st.title("My AI Agent")
st.caption("Powered by Groq + LangGraph")

# ── MESSAGES ──────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── INPUT ─────────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Kuch bhi poocho..."):
    if not groq_key:
        st.error("Pehle sidebar mein Groq API Key daalo.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                agent = get_agent(groq_key, model_choice, temperature)
                history = []
                for m in st.session_state.messages[:-1]:
                    if m["role"] == "user":
                        history.append(HumanMessage(content=m["content"]))
                    else:
                        history.append(AIMessage(content=m["content"]))
                history.append(HumanMessage(content=user_input))

                result = agent.invoke({"messages": history})
                answer = result["messages"][-1].content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")
