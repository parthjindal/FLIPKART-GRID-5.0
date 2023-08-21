from config import *
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage,HumanMessage
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import requests
import os

def create_social_media_trends():
        prompt_template = """You are given a paragraph which contains details about some trending fashion. You have to extract fashion keywords and summarize the paragraph in max 5 sentences:
        "{text}"
        """
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm = OpenAI(OPENAI_API_KEY=OPENAI_API_KEY1,temperature=0, model_name="gpt-3.5-turbo-16k")
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        with open(f"./vogueSummary.txt", "w") as f:
            arr = os.listdir('./vogueMen')
            for file in arr:
                loader = TextLoader(f"./vogueMen/{file}")
                docs = loader.load()
                file_summary1 = stuff_chain.run(docs)
                print(file_summary1)
                f.write(file_summary1)
            arr2 = os.listdir('./vogueWomen')
            for file in arr2:
                loader = TextLoader(f"./vogueWomen/{file}")
                docs = loader.load()
                file_summary2 = stuff_chain.run(docs)
                print(file_summary2)
                f.write(file_summary2)


create_social_media_trends()