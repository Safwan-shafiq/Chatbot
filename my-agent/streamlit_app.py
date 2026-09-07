import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Keys load
load_dotenv("../.env")
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stSidebarCollapsedControl"] { display: none; }

/* Global */
* { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #0f0f0f;
    color: #ececec;
}

/* Main container */
.main .block-container {
    max-width: 760px;
    margin: 0 auto;
    padding: 0 1rem 6rem 1rem;
}

/* Header */
.chat-header {
    text-align: center;
    padding: 3rem 0 2rem 0;
}

.chat-header h1 {
    font-size: 2rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}

.chat-header p {
    color: #666;
    font-size: 0.85rem;
    margin-top: 0.4rem;
}

/* Messages */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 0.8rem 0 !important;
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) 
[data-testid="stChatMessageContent"] {
    background: #1f1f1f;
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.1rem;
    color: #ececec;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) 
[data-testid="stChatMessageContent"] {
    background: transparent;
    padding: 0.2rem 0.5rem;
    color: #ececec;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* Hide avatars */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    max-width: 760px;
    background: #0f0f0f;
    padding: 1rem;
    border-top: 1px solid #1f1f1f;
}

[data-testid="stChatInput"] textarea {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    color: #ececec !important;
    font-size: 0.95rem !important;
    padding: 0.8rem 1rem !important;
    resize: none !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #444 !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] button {
    background: #ffffff !important;
    border-radius: 8px !important;
    color: #000 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #141414;
    border-right: 1px solid #1f1f1f;
}

[data-testid="stSidebar"] * {
    color: #ccc !important;
}

[data-testid="stSidebar"] input {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #ececec !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
}

/* Sidebar toggle button */
[data-testid="stSidebarNav"] { display: none; }

/* Spinner */
.stSpinner > div {
    border-top-color: #555 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f0f0f; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }

/* Welcome screen */
.welcome-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 2rem;
}

.welcome-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    cursor: pointer;
    transition: border-color 0.2s;
}

.welcome-card:hover {
    border-color: #444;
}

.welcome-card p {
    color: #999;
    font-size: 0.82rem;
    margin: 0.3rem 0 0 0;
}

.welcome-card h4 {
    color: #ececec;
    font-size: 0.9rem;
    font-weight: 500;
    margin: 0;
}

.model-badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    color: #666;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    st.divider()

    groq_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_..."
    )

    model_choice = st.selectbox(
        "Model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
        label_visibility="collapsed"
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, label_visibility="collapsed")

    st.divider()
    st.markdown("**Tools**")
    st.markdown("Web Search · Calculator · Text")

    st.divider()
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─── Tools ────────────────────────────────────────────────────────────────────
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

@tool
def word_counter(text: str) -> str:
    """Count words and characters in a text."""
    words = len(text.split())
    chars = len(text)
    return f"Words: {words} | Characters: {chars}"

tools = [calculator, web_search, word_counter]

# ─── Agent ────────────────────────────────────────────────────────────────────
def get_agent(api_key, model, temp):
    llm = ChatGroq(model=model, api_key=api_key, temperature=temp)
    return create_react_agent(llm, tools)

# ─── Session ──────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Header ───────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="chat-header">
        <h1>What can I help with?</h1>
        <div class="model-badge">Groq + LangGraph</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Messages ─────────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── Input ────────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Message AI Assistant..."):

    if not groq_key:
        st.error("Open sidebar and add your Groq API Key")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                agent = get_agent(groq_key, model_choice, temperature)

                history = []
                for msg in st.session_state.messages[:-1]:
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
                st.error(f"Error: {str(e)}")
