import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from typing import Optional
import streamlit as st

from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.agents import initialize_agent, Tool, Agent, tool
from langchain.agents.agent_toolkits import VectorStoreToolkit, VectorStoreInfo
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
import warnings
from pprint import pprint
load_dotenv(find_dotenv())

from langchain.agents.agent_types import AgentType
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
from langchain.llms import HuggingFaceHub
from HuggingChatLLM import best_llm, gpt_llm, llms, img_dir, tts, serp_search, embeddings
from credits import (
    HUGGINGFACE_EMAIL,
    HUGGINGFACE_PASS,
    HUGGINGFACE_TOKEN,
    OPENAI_API_KEY,
    ELEVENLABS_API_KEY,
    SERPAPI_API_KEY,
)
login(HUGGINGFACE_TOKEN)
from streamlit import config
from langchain.chains import( LLMChain,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
#from langchain_experimental.utilities import PythonREPL
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.cache import InMemoryCache
#from langchain.globals import set_llm_cache
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.tools.eleven_labs.text2speech import ElevenLabsText2SpeechTool
#from prompter import PiratePromptGenerator, template
                            #from langchain_experimental import PythonREPLTool
import autopep8
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, tool
from langchain.schema.runnable import (
    ConfigurableField,
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
import json
import time
from time import sleep
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage


class DocRagger:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.html_code = ""
        self.css_code = ""
        self.chunk_size=512
        self.buffer_memory = None
        self.summary_memory = None
        self.combined_memory = None
        self.controller = None

    def load_html_code(self):
        with open(f'{self.script_dir}/DocRagger.html', 'r') as file:
            self.html_code = file.read()

    def load_css_code(self):
        with open(f'{self.script_dir}/DocRagger.css', 'r') as file:
            self.css_code = file.read()

    def initialize_memory(self):
        self.buffer_mDocRaggeremory = ConversationBufferMemory(
            memory_key="buf_history",
            output_key='buf_out',
            input_key='buf_in',
        )
        self.summary_memory = ConversationSummaryMemory(
            memory_key="sum_history",
            llm=best_llm,
            output_key='sum_out',
            input_key='sum_in',
        )
        self.combined_memory = CombinedMemory(memories=[self.buffer_memory, self.summary_memory])

    def initialize_controller(self):
        prompt = ""
        ideation_llm = ""
        critique_llm = ""
        resolver_llm = ""
        llm = ""
        n_ideas = ""
        return_intermediate_steps = ""
        chain = ""
        vector_store_toolkit = ""
        self.controller = SmartLLMChainController(
            prompt,
            ideation_llm,
            critique_llm,
            resolver_llm,
            llm,
            n_ideas,
            return_intermediate_steps,
            chain,
            vector_store_toolkit,
        )
        

    def run(self):
        st.markdown(self.html_code, unsafe_allow_html=True)
        st.markdown(f'<style>{self.css_code}</style>', unsafe_allow_html=True)
        st.title("🦜 PythonRagger 🦜")
        st.text("Chat With Your Docs")
        st.write("Choose a folder full of python scripts")


        folder_path = st.text_input("Enter the folder path:", key="select_folder")
        if st.button("Select Folder"):
            self.my_callback_cache(folder_path)


    def my_callback_cache(self, widget_value):
        ...
        if widget_value is not None:
            path = widget_value
            st.write(f"The new loader path is set to: '{path}'")

            loader = GenericLoader.from_filesystem(
                path=path,
                glob="**/*",
                suffixes=[".py"],
                parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
                show_progress=False,
            )

            documents = loader.load()

            #url_loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
            #data = loader.load()

            num_documents = len(documents)
            st.write(f"documents loaded: {num_documents}")

            python_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=self.chunk_size, chunk_overlap=8
            )
            chunks = python_splitter.split_documents(documents)
            num_chunks = len(chunks)

            st.write(
                f"{num_documents} Scripts are split into {num_chunks} Chunks by RecursiveCharacterTextSplitter and the PythonParser!"
            )

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            st.write(f"{num_chunks} Chunks Embedded by 'All-MiniLM-L6-v2'!")

            #F = FAISS.from_documents(chunks, embeddings)
            index_name=path.replace("/","_") # "docs"

            persist_path = os.path.join(self.script_dir, ".indedocsx")
            faiss_pers_file_path=persist_path+"/"+index_name+".faiss"
            if os.path.exists(faiss_pers_file_path):
                st.write(f"Persistent index already exists at {persist_path}")
                F = FAISS.load_local(persist_path, embeddings,index_name)
                F.add_documents(chunks)
            else:
                st.write(f"Creating new persistent index at {persist_path}")
                F = FAISS.from_documents(chunks, embeddings)
            
            F.save_local(persist_path,index_name)

            st.write(f"Index saved at: {persist_path}")

            st.write("Documents stored!")
            st.write("Chunks stored in the VectorStore!")

            #test
            query = "count the usage of langchain:"
            results = F.similarity_search(query)
            for i, result in enumerate(results):
                #document = Document(text=result[0])
                #embedding = embeddings.encode([document], return_tensors='pt')['input_ids'] 
                st.markdown(f"""<h3>Result #{i+1}</h3>{results[i]}""")
                #t.image(embedding[0].detach().numpy())
                #st.code(result[0])


            Vector_Toolkit = VectorStoreToolkit(
                name="ScriptStorage",
                description="DocStore",
                llm=best_llm,
                vectorstore_info=VectorStoreInfo(
                    name="VectorStore",
                    description="Vector Store",
                    vectorstore=F,
                ),
            )


            agent_tools = [
                Tool(
                    func=serp_search.run,
                    name="Search",
                    description="useful for when you need to answer questions about current events",
                ),
                Tool(
                    func=tts.stream_speech,
                    name="Speak",
                    description="useful for when you need to output your answer in audio voice",
                ),
                Tool(
                    name="Autopep8",
                    func=autopep8.fix_code,
                    description="useful for formatting Python code according to PEP8 style guide",
                ),
                #Vector_Toolkit
            ] 
            executor = initialize_agent(
                    tools=agent_tools,
                    llm=best_llm,
                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
            )
            #ppp = PiratePromptGenerator(agent_tools, [self.combined_memory])
            from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
            
            # Create an instance of ChatMessageHistory
            chat_history = ChatMessageHistory(key="langchain_messages")

            st.title("🦜 ChatWithDocs")
            while True:
                # Get the current Unix timestamp
                timestamp = str(int(time.time()))
                
                # Concatenate the timestamp with the key value
                input_key = f"_input_{timestamp}"
                sleep(1)
                query = st.text_input("Enter your question:", 
                                    key=f"question_{input_key}"
                                    )
                   
                if st.button("Send",key=input_key):
                    # Create a ChatPromptTemplate and format the prompt
                    prompt_template = ChatPromptTemplate(
                        template="System: Welcome to the chat!\n{context}\n{chat_history}\nUser: {user_input}",
                        context=Vector_Toolkit,
                        chat_history=chat_history,
                        user_input=query,
                    )
                    # Get the formatted prompt
                    formatted_prompt = prompt_template.format_prompt()

                    # Display the formatted prompt in Streamlit
                    st.text(formatted_prompt)

                    # Add user and AI messages to the chat history
                    chat_history.add_user_message(f"User: {query}")
                    result = executor.run(query)
                    chat_history.add_ai_message("AI: {result}")
                    st.write(f"Result: {result}")

                with st.sidebar:# Display the chat messages in Streamlit
                    for message in chat_history.messages:
                        if message.type == "user":
                            st.text(f"User: {message.content}")
                        elif message.type == "ai":
                            st.text(f"AI: {message.content}")

                # Add a condition to break the loop if the user enters a specific command or condition to exit the chat
                if query == "exit":
                    break

                



""" 
            # single question
            query = st.text_input("Enter your question:",key="question_input")
            if st.button("Send"):
                result = executor.run(query)
                st.write(f"Result:{result}")
"""

#To use the DocRagger class, you can create an instance of it and call the run() method:


class SmartLLMChainController:
    def __init__(self, prompt, ideation_llm, critique_llm, resolver_llm, llm, n_ideas, return_intermediate_steps, vector_store_toolkit, agent_tools):
        self.chain = LLMChain(prompt=prompt, aideation_llm=ideation_llm, 
                                    critique_llm=critique_llm,
                                    resolver_llm=resolver_llm, llm=llm,
                                    n_ideas=n_ideas, return_intermediate_steps=return_intermediate_steps)
        


doc_ragger = DocRagger()
doc_ragger.run()