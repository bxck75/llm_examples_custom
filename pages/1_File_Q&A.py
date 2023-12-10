
from HuggingChatLLM import best_llm, gpt_llm, llms, img_dir, tts, serp_search, embeddings
from credits import HUGGINGFACE_EMAIL,HUGGINGFACE_PASS,HUGGINGFACE_TOKEN,OPENAI_API_KEY,ELEVENLABS_API_KEY,SERPAPI_API_KEY
from langchain.llms import HuggingFaceHub
import streamlit as st



st.set_page_config(page_title="QnA App", page_icon=":robot:")
st.header("What's your Qs?")
input = st.text_input("Qs: ", key="input")
if st.button('Generate'):
  ans = llms[0](input)
  st.subheader("Answer: ")
  st.write(ans)