import warnings
warnings.filterwarnings("ignore")

from recommender_tool import FashionOutfitGenerator
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from config import *
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from query_refiner import QueryRefiner


def get_agent(llm, memory):
    tools = [FashionOutfitGenerator()]

    system_message = SystemMessage(content="""
    You are a fashion outfit generator chatbot.
    Output format: 
    List the product names and prices for all the products in the outfit as output """)
    sys_message = """You are a fashion outfit generator chatbot.
    Output format: 
    List the product names and prices for all the products in the outfit as output"""
    prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injectd
    ])
    # main_mode = LLMChain(
    #         llm=llm,
    #         prompt=prompt,
    #         verbose=VERBOSE,
    #         memory=memory,
    #     )
    agent = initialize_agent(tools,llm,agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                            verbose=VERBOSE,
                            memory=memory,
                            handle_parsing_errors=True, prompt = prompt, 
                            agent_kwargs={"system_message": sys_message})
    return agent

def main():
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    queryRefiner = QueryRefiner(llm,memory)
    agent = get_agent(llm,memory)
    #llmchain = LLMChain(llm,condense_question_prompt,verbose=VERBOSE)
    print("Ready to chat!")
    while True:
        inp = queryRefiner(input())
        response = agent.run(input = inp)
        print(response)
        print("---------END OF RESPONSE---------")

if __name__=="__main__":
    main()