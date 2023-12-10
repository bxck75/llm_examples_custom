import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from credits import HUGGINGFACE_EMAIL,HUGGINGFACE_PASS,HUGGINGFACE_TOKEN,OPENAI_API_KEY
from hugchat import hugchat 
from hugchat.login import Login
from huggingface_hub import login
from dotenv import load_dotenv,find_dotenv
from credits import (
    HUGGINGFACE_TOKEN,
    HUGGINGFACE_TOKEN as HUGGINGFACEHUB_API_TOKEN,
    HUGGINGFACE_EMAIL,
    HUGGINGFACE_PASS,
    OPENAI_API_KEY,
    ELEVENLABS_API_KEY,
    SERPAPI_API_KEY,
    hugging_auth,
    HUGGING_BOT)
hugging_auth()


st.title("🦜🔗 Langchain - Blog Outline Generator App")

with st.sidebar:
    with st.expander('credentials')
        hug_api_key= st.sidebar.text_input("HUGGINGFACEHUB_API_TOKEN", type="password")
        hug_email= st.sidebar.text_input("HUGGINGFACE_EMAIL", type="default")
        hug_pass= st.sidebar.text_input("HUGGINGFACE_PASS", type="password")



def blog_outline(topic):
    # Instantiate LLM model
    llm = ChatBot
    # Prompt
    template = "As an experienced data scientist and technical writer, generate an outline for a blog about {topic}."
    prompt = PromptTemplate(input_variables=["topic"], template=template)
    prompt_query = prompt.format(topic=topic)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    email_text = st.text_input("Enter email:", "")
    pwd_text = st.text_input("Enter pass:", "")
    submitted = st.form_submit_button("login")
    if not openai_api_key:
        st.info("Please add your hf key to continue.")
    elif submitted:
        blog_outline(topic_text)
