from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from config import *

TEMPLATE = """
Given the following chat history between a User and Fashion Outfit Recommending AI,
Rephrase the user input to be a standalone input that does not require the whole chat history or end the conversation if it seems like it's done.
You may also include a summary of the chat history to provide context to the user input.
Chat History:
\"""
{chat_history}
\"""
User input:
\"""
{input}
\"""

Standalone input:
"""

class QueryRefiner():
    def __init__(self, baseLLM, memory:ConversationBufferMemory, *args, **kwargs):
        promptTemplate = PromptTemplate(template=TEMPLATE,input_variables=["chat_history","input"])
        self.llm_chain = LLMChain(llm=baseLLM,prompt=promptTemplate,**kwargs)
        self.memory = memory

    def run(self, input):
        chat_history = self.memory.buffer_as_str
        return self.llm_chain.run(input=input,
                                 chat_history=chat_history)
    
    def __call__(self, input):
        return self.run(input)
    
    def __repr__(self):
        return f"QueryRefiner(llm_chain={self.llm_chain}, memory={self.memory})"
    
def test():
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    llm = ChatOpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate.from_template("You are an AI Assistant. Human: {input}, Your reply:")
    llm_chain = LLMChain(llm=llm,memory=memory,prompt=prompt)
    qr = QueryRefiner(baseLLM=llm,memory=memory)
    query = qr("Who is sachin tendulkar ?")
    print(query)
    resp = llm_chain.run(input=query)
    print(resp)
    query = qr("Is he a batsman ?")
    print(query)

if __name__=="__main__":
    test()