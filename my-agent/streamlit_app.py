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

* { font-family: 'Inter', sans-serif; }

#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* Background */
.stApp { background-color: #212121; color: #ececec; }

/* Main area */
.main .block-container {
    max-width: 720px;
    margin: 0 auto;
    padding: 2rem 1rem 7rem 1rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2f2f2f;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #c9c9c9 !important;
}
[data-testid="stSidebar"] input {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px !important;
    color: #ececec !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px !important;
    color: #ececec !important;
}
[data-testid="stSidebar"] .stButton button {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    color: #ececec !important;
    border-radius: 8px !important;
    width: 100%;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #333 !important;
    border-color: #555 !important;
}

/* Hide chat avatars completely */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}

/* Chat messages container */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
    gap: 0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) 
[data-testid="stChatMessageContent"] {
    background: #2f2f2f !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 0.75rem 1rem !important;
    max-width: 75% !important;
    margin-left: auto !important;
    color: #ececec !important;
    font-size: 0.93rem !important;
    line-height: 1.65 !important;
}

/* Assistant message - no bubble, just text */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) 
[data-testid="stChatMessageContent"] {
    background: transparent !important;
    padding: 0.3rem 0.2rem !important;
    color: #ececec !important;
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
    max-width: 100% !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #212121 !important;
    border-top: 1px solid #2f2f2f !important;
    padding: 1rem !important;
}
[data-testid="stChatInput"] textarea {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 12px !important;
    color: #ececec !important;
    font-size: 0.93rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #555 !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] button {
    background: #ececec !important;
    border-radius: 8px !important;
    color: #000 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #212121; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }

/* Divider */
hr { border-color: #2f2f2f !important; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### AI Assistant")
    st.divider()

    groq_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_..."
    )

    st.markdown("**Model**")
    model_choice = st.selectbox(
        "Model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
        label_visibility="collapsed"
    )

    st.markdown("**Temperature**")
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.7,
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Tools available**")
    st.markdown("- Web Search\n- Calculator\n- Word Counter")
    st.divider()

    if st.button("New Conversation"):
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
    return f"Words: {len(text.split())} | Characters: {len(text)}"

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
        <div style='text-align:center; padding: 4rem 0 2rem 0;'>
            <h2 style='color:#ececec; font-weight:600; font-size:1.8rem; margin:0;'>
                What can I help with?
            </h2>
            <p style='color:#555; font-size:0.82rem; margin-top:0.5rem;'>
                Powered by Groq + LangGraph
            </p>
        </div>
    """, unsafe_allow_html=True)

# ─── Messages ─────────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── Input ────────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Message AI Assistant..."):

    if not groq_key:
        st.error("Please add your Groq API Key in the sidebar.")
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
