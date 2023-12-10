from openai import OpenAI
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain_experimental.smart_llm import SmartLLMChain 
from HuggingChatLLM import best_llm, gpt_llm, llms, img_dir, tts, serp_search, embeddings
from credits import HUGGINGFACE_EMAIL,HUGGINGFACE_PASS,HUGGINGFACE_TOKEN,OPENAI_API_KEY,ELEVENLABS_API_KEY,SERPAPI_API_KEY
from langchain.llms import HuggingFaceHub
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
    openai_api_key = OPENAI_API_KEY
    #st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    #"[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    #"[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    #"[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespacesauth.savec
st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

def llm_fetch(prompt, llm):
    try:
        if not prompt:
            raise ValueError("Prompt cannot be empty or null")

        system_message=f'''Cyber Security Specialist,I want you to act as a cyber security specialist. 
                            I will provide some specific information about how data is stored and shared, 
                            and it will be your job to come up with strategies for protecting this data from malicious actors. 
                            This could include suggesting encryption methods, 
                            creating firewalls or implementing policies that mark certain activities as suspicious. 
                            My first request is: '''
        # Track messages separately
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        # Update streamlit messages manually
        st.session_state.messages.extend(messages)
        for msg in messages:
            if msg["role"] == "system":
                st.chat_message("system").write(msg["content"])
            elif msg["role"] == "user":
                st.chat_message("user").write(msg["content"])

        string_prompt = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        #print(string_prompt)

        response = llm(prompt=string_prompt)
        #print(response)

        return f"{response}"
    except Exception as e:
        # Handle exception here
        print(f"Error: {e}")
        return None

msg=llm_fetch(prompt,llms[0])
st.session_state.messages.append({"role": "assistant", "content": msg})
st.chat_message("assistant").write(msg)
