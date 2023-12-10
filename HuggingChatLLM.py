import os ,re
script_dir = os.path.dirname(os.path.abspath(__file__))
from dotenv import load_dotenv, find_dotenv
from typing import Union
import warnings
from huggingface_hub import login
from langchain.llms import HuggingFaceHub
from elevenlabs import set_api_key
from tempfile import TemporaryDirectory
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.embeddings import HuggingFaceEmbeddings
warnings.filterwarnings('ignore')
load_dotenv(find_dotenv())
from credits import (
    HUGGINGFACE_TOKEN,
    HUGGINGFACE_TOKEN as HUGGINGFACEHUB_API_TOKEN,
    HUGGINGFACE_EMAIL,
    HUGGINGFACE_PASS,
    OPENAI_API_KEY,
    ELEVENLABS_API_KEY,
    SERPAPI_API_KEY)


from langchain.utilities.serpapi import SerpAPIWrapper
serp_search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
from langchain.tools.eleven_labs.text2speech import ElevenLabsText2SpeechTool
set_api_key(ELEVENLABS_API_KEY)
tts = ElevenLabsText2SpeechTool(eleven_api_key=ELEVENLABS_API_KEY, voice="amy")

repo_list = [
    {"user": "tiiuae", "model": "falcon-7b-instruct"},
    {"user": "mistralai", "model": "Mistral-7B-v0.1"},
    {"user": "openchat", "model": "openchat_3.5"},
    {"user": "01-ai", "model": "Yi-34B"},
    {"user": "codellama", "model": "CodeLlama-7b-Python-hf"},
]
repo_ids = []  # Initialize an empty list to store repo IDs
for i, repo in enumerate(repo_list):
    repo_id = f"{repo['user']}/{repo['model']}"
    repo_ids.append(repo_id)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["HUGGINGFACE_EMAIL"] = HUGGINGFACE_EMAIL
os.environ["HUGGINGFACE_PASS"] = HUGGINGFACE_PASS
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

login(HUGGINGFACEHUB_API_TOKEN)

embeddings= HuggingFaceEmbeddings(
                                model_name="all-MiniLM-L6-v2",
                                model_kwargs = {'device': 'cpu'},
                                encode_kwargs = {'normalize_embeddings': True}
                            )

best_llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", task="text-generation", model_kwargs = {"min_length": 200,"max_length":1000,"temperature":0.1, "max_new_tokens":512, "num_return_sequences":1})           
llms=[
    HuggingFaceHub( repo_id=repo_ids[0], task="text-generation", model_kwargs = {"min_length": 32,"max_length":1000,"temperature":0, "max_new_tokens":128, "num_return_sequences":1 }),
    HuggingFaceHub( repo_id=repo_ids[1], task="text-generation", model_kwargs = {"min_length": 200,"max_length":1000,"temperature":0.1, "max_new_tokens":512, "num_return_sequences":1 }),
    HuggingFaceHub( repo_id=repo_ids[2], task="text-generation", model_kwargs = {"min_length": 200,"max_length":1000,"temperature":0.1, "max_new_tokens":512, "num_return_sequences":1 }),
    HuggingFaceHub( repo_id=repo_ids[3], task="text-generation", model_kwargs = {"min_length": 200,"max_length":1000,"temperature":0.1, "max_new_tokens":512, "num_return_sequences":1 }),
    HuggingFaceHub( repo_id=repo_ids[4], task="text-generation", model_kwargs = {"min_length": 200,"max_length":1000,"temperature":0.1, "max_new_tokens":512, "num_return_sequences":1 })
]

gpt_llm = llms[2]

working_directory = TemporaryDirectory()
data_dir=os.path.join(script_dir,"data")
img_dir=os.path.join(script_dir,"images")


def llm_fetch(prompt):

    # Track messages separately
    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": prompt}
    ]

    # Update streamlit messages manually
    #st.session_state.messages.extend(messages)
    #for msg in messages:
    #    if msg["role"] == "system":
    #       st.chat_message("system").write(msg["content"])
    #    elif msg["role"] == "user":
    #       st.chat_message("user").write(msg["content"])

    response = best_llm(prompt=" ".join([f"{msg['role']}: {msg['content']}" for msg in messages]))
    return f"{response['text']}"

from typing import Any, Iterator, List, Optional


from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel

