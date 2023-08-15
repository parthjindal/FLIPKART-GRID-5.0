import warnings
warnings.filterwarnings("ignore")

from recommender_tool import FashionOutfitGenerator
from langchain.agents import Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from config import *
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, OpenAIFunctionsAgent
from langchain.agents import AgentType

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
tools = [FashionOutfitGenerator()]
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

system_message = SystemMessage(content="""
You are a fashion outfit generator chatbot. You need to create a complete single fashion outfit for the user and then find appropriate products for all outfits from flipkart.
Return the product name and the price for it.
You must take feedback from the user about the suggested outfit and make changes accordingly.
""" )

agent = initialize_agent(tools,llm,agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                         verbose=True,memory=memory,
                         handle_parsing_errors=True)

def func(query):
    response = agent.run(input=query)
    return response