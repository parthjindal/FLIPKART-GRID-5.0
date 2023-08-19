from config import *
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.llms.base import BaseLLM
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage,HumanMessage
from langchain.tools import BaseTool, tool
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder,SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
import requests
import datetime
import os
import logging

class SearchCache:
    def __init__(self):
        self.cache = []

    def getAll(self):
        return [item for item in self.cache]
    
    def add(self, item):
        self.cache.append(item)

    def clear(self):
        self.cache.clear()


class FashionOutfitGeneratorTool():
    name = "fashion_outfit_generator"
    description = """
    This tool is a fashion outfit generator Chat. Use this tool to generate a fashion outfit consisting of clothing,accessories and footwear. 
    or make changes to an existing outfit or an outfit piece.
    If the requirement is to create a new outfit, then:
        Use it to generate complete outfit aligning with user profile and purchase history.
    else:  
        Use it to make changes to the current outfit based on the user ask. 
        In the input, mention the the pieces that need to be changed.
    Input must specify if the request is to create a new outfit or to make changes to an existing outfit along with the user requirements.
    """

    def __init__(self, baseLLM: BaseLLM, searchCache: SearchCache, *args, **kwargs) -> None:
        
        self.baseLLM = baseLLM
        self.searchCache = searchCache   
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.logger = logging.getLogger()
        
        userProfile = self.createUserProfile()
        self.logger.debug(f"User Profile: {userProfile}")

        fetchHistory = kwargs.get("fetch_history", True)
        userPurchaseHistory = ""
        if fetchHistory:
            userPurchaseHistory = self.createUserPurchaseHistory()
            self.logger.debug(f"User Purchase History: {userPurchaseHistory}")

        self.promptTemplate = PromptTemplate.from_template(f"""
System:
You are a fashion outfit generator. On the basis of the input prompt,
If the input prompt is to change some of the items in the current outfit, then recommend changes to the given outfit based on the user ask.
else recommend a complete outfit consisting of clothing, accessories and footwear aligning with user profile, user purchase history and current fashion trends.
The item names may mention the colors and patterns as well. Always adhere to the output format.

Output format:
List the items in a python list format.
Example: ["red shirt", "blue jeans", "black shoes"]

User profile:
\"""
'{userProfile}'
\"""
User purchase History: 
\"""
'{userPurchaseHistory}'
\"""
Current fashion trends: 
\"""
'{fashionTrends}'
\"""

Human: {{human_input}}
""")


    def run(self, prompt: str):
        llmChain = LLMChain(
            llm=self.baseLLM,
            prompt=self.promptTemplate,
            verbose=VERBOSE,
            memory=self.memory,
        )


    def createUserProfile(self):
        self.logger.debug("Creating user profile")
        # loader = TextLoader("./user_profile.txt")
        # documents = loader.load()
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # texts = text_splitter.split_documents(documents)

        # embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
        # docsearch = Chroma.from_documents(texts, embeddings)

        # qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key= OPENAI_API_KEY), chain_type="stuff", retriever=docsearch.as_retriever())
        user_profile_local = "The profile of the user is a 22 year old male from Gujarat"
        self.logger.debug("User profile created")
        # return qa.run("What is the profile of the user?")
        return user_profile_local
        
    def createUserPurchaseHistory(self):
        self.logger.debug("Creating user purchase history")
        prompt_template = """You are given the purchase history of a user as a list of products in the below text. You have to summarize and find the preferences of the user such as color, pattern, type etc.:
        "{text}"
        USER PREFERENCES:"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm = OpenAI(openai_api_key=OPENAI_API_KEY3,temperature=0, model_name="gpt-3.5-turbo-16k")
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        loader = TextLoader("./user_purchase_history.txt")
        docs = loader.load()
        history = stuff_chain.run(docs)
        self.logger.debug("User purchase history created")
        
        return history


class FashionOutfitGenerator(BaseTool):
    

    baseLLM: BaseLLM = None
    searchCache: SearchCache = None
    memory: ConversationBufferMemory = None
    userProfile: str = ""
    userPurchaseHistory: str = ""
    
    def __init__(self, ):
        super().__init__()
        memory = ConversationBufferMemory(memory_key="chat_history",
                                               return_messages=True)
        userProfile = self.createUserProfile()
        userPurchaseHistory = self.createUserPurchaseHistory()
        fashionTrends = ""
        promptTemplate = PromptTemplate.from_template(f"""
System:
You are a fashion outfit generator. On the basis of the input prompt,
If the input prompt is to change some of the items in the current outfit, then recommend changes to the given outfit based on the user ask.
else recommend a complete outfit consisting of clothing, accessories and footwear aligning with user profile, user purchase history and current fashion trends.
The item names may mention the colors and patterns as well. Always adhere to the output format.

Output format:
List the items in a python list format.
Example: ["red shirt", "blue jeans", "black shoes"]

User profile:
\"""
'{userProfile}'
\"""
User purchase History: 
\"""
'{userPurchaseHistory}'
\"""
Current fashion trends: 
\"""
'{fashionTrends}'
\"""

Human: {{human_input}}
""")
        
        self.chatLLMChain = LLMChain(
            llm=baseLLM,
            prompt=promptTemplate,
            verbose=VERBOSE,
            memory=memory,
        )
        self.searchCache = searchCache

    def createUserProfile(self):
        # loader = TextLoader("./user_profile.txt")
        # documents = loader.load()
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # texts = text_splitter.split_documents(documents)

        # embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
        # docsearch = Chroma.from_documents(texts, embeddings)

        # qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key= OPENAI_API_KEY), chain_type="stuff", retriever=docsearch.as_retriever())
        user_profile_local = "The profile of the user is a 22 year old male from Gujarat"
        # return qa.run("What is the profile of the user?")
        return user_profile_local

    def createUserPurchaseHistory(self):
        # prompt_template = """You are given the purchase history of a user as a list of products in the below text. You have to summarize and find the preferences of the user such as color, pattern, type etc.:
        # "{text}"
        # USER PREFERENCES:"""
        # prompt = PromptTemplate.from_template(prompt_template)

        # # Define LLM chain
        # llm = OpenAI(openai_api_key=OPENAI_API_KEY3,temperature=0, model_name="gpt-3.5-turbo-16k")
        # llm_chain = LLMChain(llm=llm, prompt=prompt)

        # # Define StuffDocumentsChain
        # stuff_chain = StuffDocumentsChain(
        #     llm_chain=llm_chain, document_variable_name="text"
        # )
        # loader = TextLoader("./user_purchase_history.txt")
        # docs = loader.load()
        # return (stuff_chain.run(docs))
        return ""

    def create_social_media_trends(self, query):
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

    def _run(self,prompt: str):
        print(prompt)
        # fashion_trends = self.create_social_media_trends(prompt)
        # fashion_trends = ""
        # self.chatLLMChain.prompt.partial(fashion_trends=fashion_trends)
        output = self.chatLLMChain.run(human_input=prompt)

        # validate output
        if self.validate_output(output) == False:
            raise Exception(f"Output is not a python list: {output}")
        
        outfit_list = eval(output)
        outfit: str = ""
       
        self.searchCache.clear()
        for item in outfit_list:
            search_result = self.search_flipkart(item)
            name = search_result[0]["name"]
            price = search_result[0]["current_price"]
            outfit += f"{item}: {name}, Rs. {price} \n"
            self.searchCache.add(search_result[0])
        
        return outfit
    
    def search_flipkart(self, search_term):
        URL = f"https://flipkart-scraper-api.dvishal485.workers.dev/search/{search_term}"
        response = requests.get(URL).json()
        response = response["result"]
        return response[:1]

    def validate_output(self, output: str):
        # try to check if output is a python list
        try:
            eval(output)
            return True
        except:
            return False

    def _arun(self, prompt: str):
        raise NotImplementedError("This tool does not support async")

def main():
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY3)

    fashion_outfit_generator = FashionOutfitGeneratorTool(llm, SearchCache())



    # fashion_outfit_generator.create_user_purchase_history()
    # print(fashion_outfit_generator._run("Create an outfit for a 20 year old girl for a  birthday party"))
    #print(fashion_outfit_generator._run("Give products only with rating > 4*"))
# print(fashion_outfit_generator._run("recommend something else instead of flip flops."))
# print(fashion_outfit_generator._run("No stick with the flip flops."))


if __name__ == "__main__":
    logger = logging.getLogger()
    # currentTime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(level=logging.DEBUG, 
                        format="[%(asctime)s:%(msecs)03d %(levelname)s]: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",filename=f"./logs/logfile.log", filemode="w")
    main()
