import argparse

from typing import List, Callable
from g4f.Provider import (
    AItianhu, AItianhuSpace, Acytoo, AiAsk, AiService, Aibn, Aichat, Ails, Aivvm, AsyncGeneratorProvider,
    AsyncProvider, Bard, BaseProvider, Bing, ChatBase, ChatForAi, Chatgpt4Online, ChatgptAi, ChatgptDemo,
    ChatgptDuo, ChatgptFree, ChatgptLogin, ChatgptX, CodeLinkAva, Cromicle, DeepInfra, DfeHub, EasyChat,
    Equing, FastGpt, Forefront, FakeGpt, FreeGpt, GPTalk, GptChatly, GetGpt, GptForLove, GptGo, GptGod
)
#cookies=g4f.get_cookies(".google.com"),
from g4f import Provider, models, Model, RetryProvider
from langchain.llms.base import LLM
import g4f
from langchain_g4f import G4FLLM
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from hugchat import hugchat

from credits import ELEVENLABS_API_KEY, SERPAPI_API_KEY,BEARLY_API_KEY, OPENAI_API_KEY,HUGGINGFACE_EMAIL,HUGGINGFACE_PASS
from elevenlabs import generate, play
from pages.HuggingFaceChat import HuggingChat
from elevenlabs import set_api_key
set_api_key(ELEVENLABS_API_KEY)
from langchain.tools import tool
from langchain.tools import Tool
# Import things that are needed generically
from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMMathChain,LLMChain
from langchain.prompts import  PromptTemplate
from langchain.schema import SystemMessage
from pydantic import BaseModel, Field
from langchain.callbacks.base import BaseCallbackHandler
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
set_llm_cache(InMemoryCache())
from langchain.utilities import PythonREPL
from langchain.utilities.serpapi import SerpAPIWrapper
serp_search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
from langchain.tools import ElevenLabsText2SpeechTool
tts = ElevenLabsText2SpeechTool(api_key=ELEVENLABS_API_KEY, voice="amy",)
"""     best_provider=RetryProvider([
        AItianhu, AItianhuSpace, Acytoo, AiAsk, AiService, Aibn, Aichat, Ails, Aivvm, AsyncGeneratorProvider,
        AsyncProvider, Bard, BaseProvider, Bing, ChatBase, ChatForAi, Chatgpt4Online, ChatgptAi, ChatgptDemo,
        ChatgptDuo, ChatgptFree, ChatgptLogin, ChatgptX, CodeLinkAva, Cromicle, DeepInfra, DfeHub, EasyChat,
        Equing, FastGpt, Forefront, FakeGpt, FreeGpt, GPTalk, GptChatly, GetGpt, GptForLove, GptGo, GptGod
    ]), 
    """
gpt_turbo = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'openai',
    best_provider=RetryProvider([
        Bard
    ])
)
llm: LLM = G4FLLM(model = gpt_turbo)
hf_llm: LLM = HuggingChat(email=HUGGINGFACE_EMAIL,psw=HUGGINGFACE_PASS)

g4f.debug.logging = True  # Enable logging
llm_math_chain = LLMMathChain.from_llm(llm)

tools = [
    Tool.from_function(
        func=serp_search.run,
        name="Search",
        description="useful for when you need to answer questions about current events",
    ),
    Tool.from_function(
        func=tts.stream_speech,
        name="Speak",
        description="useful for when you need to output your answer in audio voice",
        return_direct=True,
    ),
]
# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=True
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process voice and question arguments.')

    parser.add_argument('question', type=str, help='The question to be answered')
    parser.add_argument('--no_voice', action='store_true', help='Disable voice output')
    args = parser.parse_args()

    #print(llm(args.question))
  
    voice = "Show the answer in text" if args.no_voice else "Show the answer in text and then with audio voice. "
    question = args.question

    final_query = f"{voice}{question}"
    print(question)
    tts.stream_speech(question)

    system_prompt="You are a pragmatic agent that will only output python code."
    final_query = f"{voice}{question}"
    agent.run(input=final_query)



        #"Answer with audio voice. Who is Leonardo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
        #"Answer with audio voice. Who is Leonardo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
    #)
