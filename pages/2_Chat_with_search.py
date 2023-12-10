import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain_experimental.smart_llm import SmartLLMChain 
from HuggingChatLLM import best_llm, gpt_llm, llms, img_dir, serp_search, embeddings
from credits import HUGGINGFACE_EMAIL,HUGGINGFACE_PASS,HUGGINGFACE_TOKEN,OPENAI_API_KEY,ELEVENLABS_API_KEY,SERPAPI_API_KEY
from langchain.vectorstores import FAISS


with st.sidebar:
    # Other input parameters
    class1 = st.sidebar.text_input('Enter class 1')
    class2 = st.sidebar.text_input('Enter class 2')
    class3 = st.sidebar.text_input('Enter class 3')
    persistent_dir = st.sidebar.text_input('Enter persistent directory')
    vector_in_dir = st.sidebar.text_input('Enter vector in directory')
    logfile_path = st.sidebar.text_input('Enter logfile path')
    api_keys = st.sidebar.text_input('Enter API keys csv')
    #openai_api_key = OPENAI_API_KEY
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    f"[Get an OpenAI API key][{OPENAI_API_KEY}](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🔎 LangChain - Chat with search")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
