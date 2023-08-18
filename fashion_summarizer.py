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

def create_social_media_trends():
        prompt_template = """You are given a paragraph which contains details about some trending fashion. You have to extract fashion keywords and summarize the paragraph in max 5 sentences:
        "{text}"
        CURRENT SOCIAL MEDIA FASHION TRENDS:"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm = OpenAI(openai_api_key=OPENAI_API_KEY,temperature=0, model_name="gpt-3.5-turbo-16k")
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        loader = TextLoader("./vogueContent.txt")
        docs = loader.load()
        return (stuff_chain.run(docs))

print(create_social_media_trends())