from .config import *
from .recommender import build_fashion_outfit_generator_tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import SystemMessage,  HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
import langchain
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(LANGCHAIN_CACHE_SQLITE_PATH)


def build_agent() -> AgentExecutor:
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY,verbose=VERBOSE)
    tools = [build_fashion_outfit_generator_tool()]

    systemMessage = SystemMessage(content="""
You are a friendly human-like conversational clothing shopping assistant. \
Use the 'fashion_outfit_generator' tool to help the shopper create an outfit look and find outfit/products from a shopping website. \
Output must provide all the details such as name, price, if it is discounted etc. \
Always be descriptive in your answers. Be very human like.""")
    
    prompt = ChatPromptTemplate.from_messages([
        systemMessage,
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
        HumanMessage(content="Tip: Provide a detailed answer."),
        AIMessage(content="Detailed Answer:")
    ])

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS, 
        verbose=True,
        memory=memory,
        handle_parsing_errors=True, 
        prompt = prompt
    )
    return agent

def test():
    agent = build_agent()
    while True:
        response = agent.run(input("Human: "))
        print(response)

if __name__=="__main__":
    test()