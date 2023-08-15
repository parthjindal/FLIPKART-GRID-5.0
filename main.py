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
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

template = """Given the following chat history and a follow up question or request, rephrase the follow up input request to be a standalone request.
Or end the conversation if it seems like it's done.
Chat History:\"""
{chat_history}
\"""
Follow Up Input: \"""
{human_input}
\"""
Standalone request:"""
 
condense_question_prompt = ChatPromptTemplate.from_template(template)

def main():
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    tools = [FashionOutfitGenerator()]
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    system_message = SystemMessage(content="""
    You are a fashion outfit generator chatbot. You need to create a complete single fashion outfit for the user and then find appropriate products for all outfits from flipkart.
    Return the product name and the price for it.
    You must take feedback from the user about the suggested outfit and make changes accordingly.
    """ )

    prompt = ChatPromptTemplate.from_messages([
            # system_message,
            MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injectd
    ])
    agent = initialize_agent(tools,llm,agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                            verbose=VERBOSE,
                            memory=memory,
                            handle_parsing_errors=True,prompt=prompt)
    
    print("Ready to chat!")
    while True:
        inp = input()
        response = agent.run(input = inp)
        print(response)
        print("---------END OF RESPONSE---------")

if __name__=="__main__":
    main()