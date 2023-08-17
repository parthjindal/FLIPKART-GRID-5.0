import warnings
warnings.filterwarnings("ignore")

from recommender_tool import FashionOutfitGenerator, SearchCache
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


def get_agent(llm,memory=None, searchCache=None):
    tools = [FashionOutfitGenerator(searchCache)]

    template ="""
    You are a friendly, conversational clothing shopping assistant.
    Use the following context to recommend the outfit with product names, product   price, descriptions, and
    keywords to show the shopper whats available, help find what they want, and answer any questions. 
    Be descriptive in your answers.
    Context:
    \"""
    {chat_history}
    \"""
    Human Input:\"
    {input}
    \"""
    Helpful answer:
    """

    prompt = PromptTemplate.from_template(template)
    agent = initialize_agent(tools,llm,agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                            verbose=VERBOSE,
                            memory=memory,
                            handle_parsing_errors=True, prompt = prompt)
    return agent

def main():
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    agent = get_agent(llm,memory)
    queryRefiner = QueryRefiner(llm,memory)
    #llmchain = LLMChain(llm,condense_question_prompt,verbose=VERBOSE)
    print("Ready to chat!")
    chat_history = []
    while True:
        inp = queryRefiner(input())
        response = agent.run(input = inp)
        print(response)
       # chat_history.append(f"Human: {inp}, AI: {response}")
        #llmchain.run(input("Ask a question from the chat history: "),chat_history)
        print("---------END OF RESPONSE---------")

if __name__=="__main__":
    main()