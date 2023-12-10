import os
from huggingface_hub import InferenceClient
import streamlit as st
#llm_huggingface=HuggingFaceHub(huggingfacehub_api_token="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",repo_id="tiiuae/falcon-7b-instruct",model_kwargs={"temperature":0.6,"max_length":64})
def llm(question):
    # output=InferenceApi(repo_id="tiiuae/falcon-7b-instruct",token="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",task='text-generation',gpu=True)
    # output1=output(question,raw_response=True)
    # print(output1)
    # return output1
    client=InferenceClient(token="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")
    output=client.text_generation(question,model="tiiuae/falcon-7b-instruct",
                                  temperature=1)
    return output

st.set_page_config(page_title="CHATBOT")
st.header("LANGCHAIN APP")
input=st.text_input("INPUT:",key="input")
response1=llm(input)
submit=st.button("Ask the questions")


if submit:
    st.subheader("The Response is")
    st.write(response1)
