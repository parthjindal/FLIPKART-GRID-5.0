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

def create_social_media_trends(query):
        prompt_template = f"""You are given the current social media fashion trends in the below text.
        Provide a summary of the fashion trends including the color trends, outfit combinations, pattern trends etc. by 
        matching the closest fashion trend from the text with the given User Query:
        Fashion Trends: \"""
        {{text}}
        \"""

        User Query: \"""
        "{query}"
        \"""

        CURRENT SOCIAL MEDIA FASHION TRENDS:"""

        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm = OpenAI(openai_api_key=OPENAI_API_KEY3,temperature=0, model_name="gpt-3.5-turbo-16k")
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        loader = TextLoader("./vogueSummary.txt")
        docs = loader.load()
        return (stuff_chain.run(docs))

summary = create_social_media_trends("Give me an outfit for a music concert.")
print(summary)