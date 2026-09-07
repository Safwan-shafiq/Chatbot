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
    page_title="AI Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* App background */
.stApp { background: #1a1a1a; }

/* Main content area */
.main .block-container {
    max-width: 680px;
    margin: 0 auto;
    padding: 2rem 1.5rem 6rem 1.5rem;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #2c2c2c !important;
}
[data-testid="stSidebar"] section {
    padding: 1.5rem 1.2rem !important;
}
[data-testid="stSidebar"] label { color: #888 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stSidebar"] p { color: #aaa !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] input {
    background: #1e1e1e !important;
    border: 1px solid #2c2c2c !important;
    border-radius: 8px !important;
    color: #eee !important;
    font-size: 0.88rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1e1e1e !important;
    border: 1px solid #2c2c2c !important;
    border-radius: 8px !important;
    color: #eee !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1e1e1e !important;
    border: 1px solid #2c2c2c !important;
    color: #ccc !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #2a2a2a !important;
    border-color: #444 !important;
    color: #fff !important;
}
hr { border-color: #2c2c2c !important; margin: 1rem 0 !important; }

/* ── HIDE AVATARS ── */
[data-testid="chatAvatarIcon-user"]    { display: none !important; }
[data-testid="chatAvatarIcon-assistant"] { display: none !important; }

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.4rem 0 !important;
    gap: 0 !important;
}

/* User bubble — right aligned, grey */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    justify-content: flex-end !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: #3a3a3a !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.65rem 1rem !important;
    max-width: 70% !important;
    margin-left: auto !important;
    color: #f0f0f0 !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    border: none !important;
}

/* Assistant — left, no background, just text */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
    color: #d4d4d4 !important;
    font-size: 0.92rem !important;
    line-height: 1.75 !important;
    max-width: 100% !important;
}

/* ── CHAT INPUT BOX ── */
[data-testid="stChatInput"] {
    background: #1a1a1a !important;
    border-top: 1px solid #2c2c2c !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stChatInput"] > div {
    background: #252525 !important;
    border: 1px solid #333 !important;
    border-radius: 14px !important;
    padding: 0.5rem 0.8rem !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: #ececec !important;
    font-size: 0.92rem !important;
    caret-color: #aaa;
}
[data-testid="stChatInput"] textarea::placeholder { color: #555 !important; }
[data-testid="stChatInput"] button {
    background: #333 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ccc !important;
}
[data-testid="stChatInput"] button:hover {
    background: #444 !important;
    color: #fff !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #2c2c2c; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    st.divider()

    groq_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_..."
    )

    st.markdown("**Model**")
    model_choice = st.selectbox(
        "model", 
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
        label_visibility="collapsed"
    )

    st.markdown("**Temperature**")
    temperature = st.slider("temp", 0.0, 1.0, 0.7, label_visibility="collapsed")

    st.divider()
    st.markdown("**Tools**")
    st.markdown("Web Search  ·  Calculator  ·  Word Counter")
    st.divider()

    if st.button("New Conversation", use_container_width=True):
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

# ── WELCOME ───────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0 2.5rem 0;">
        <h2 style="color:#ececec; font-weight:600; font-size:1.75rem; margin:0; letter-spacing:-0.3px;">
            What can I help with?
        </h2>
        <p style="color:#444; font-size:0.8rem; margin-top:0.5rem;">
            Groq + LangGraph
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── MESSAGES ──────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── INPUT ─────────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Message..."):
    if not groq_key:
        st.error("Add your Groq API Key in the sidebar.")
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
