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
import json
import requests


class FashionOutfitGenerator(BaseTool):
    name = "fashion_outfit_generator"
    description = """
    This tool is a fashion outfit generator Chat. Use this tool to generate a fashion outfit consisting of clothing,accessories and footwear. 
    or make changes to an existing outfit or an outfit piece.
    If the requirement is to create a new outfit, then:
        Use it to generate complete outfit aligning with user profile and purchase history.
    else:  
        Use it to make changes to the current outfit based on the user ask. 
        In the input, mention the the pieces that need to be changed.
    Input must specify if the request is to create a new outfit or to make changes to an existing outfit.
    """
    
    chat_llm_chain: LLMChain = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        memory = ConversationBufferMemory(memory_key="chat_history",
                                               return_messages=True)
        outfit_model = ChatOpenAI(openai_api_key = OPENAI_API_KEY,temperature = 0,verbose=VERBOSE)

        user_profile = self.create_user_profile()
        user_purchase_history = self.create_user_purchase_history()
        fashion_trends = self.create_social_media_trends()
        #fashion_trends = "The current fashion trends is to use floral print and vibrant colored outfits."
        systemMessage = SystemMessage(
            content=f"""
                You are a fashion outfit generator. On the basis of the input prompt,
                If the input prompt is to change some of the items in the current outfit, then:
                    Recommend changes to the given outfit based on the user ask.
                else:  
                    Recommend a complete outfit consisting of clothing,accessories and footwear aligning with user profile, 
                    user purchase history and current fashion trends.
                
                The item names may mention the colors and patterns as well.
                Output format:
                A python list of items in the outfit

                User profile: '{user_profile}'
                User purchase History: '{user_purchase_history}'
                Current fashion trends: '{fashion_trends}
                """
            )
        prompt = ChatPromptTemplate.from_messages([
            systemMessage,
            MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injectd
        ])

        self.chat_llm_chain = LLMChain(
            llm=outfit_model,
            prompt=prompt,
            verbose=VERBOSE,
            memory=memory,
        )

    def create_user_profile(self):
        # loader = TextLoader("./user_profile.txt")
        # documents = loader.load()
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # texts = text_splitter.split_documents(documents)

        # embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_ARYAN_KEY)
        # docsearch = Chroma.from_documents(texts, embeddings)

        # qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key= OPENAI_ARYAN_KEY), chain_type="stuff", retriever=docsearch.as_retriever())
        user_profile_local = "The profile of the user is a 22 year old male from Gujarat"
        # return qa.run("What is the profile of the user?")
        return user_profile_local

    def create_user_purchase_history(self):
        prompt_template = """You are given the purchase history of a user as a list of products in the below text. You have to summarize and find the preferences of the user such as color, pattern, type etc.:
        "{text}"
        USER PREFERENCES:"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm = OpenAI(openai_api_key=OPENAI_API_KEY,temperature=0, model_name="gpt-3.5-turbo-16k")
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        loader = TextLoader("./user_purchase_history.txt")
        docs = loader.load()
        return (stuff_chain.run(docs))

    def create_social_media_trends(self):
        prompt_template = """You are given the current social media fashion trends in the below text. You need to provide a summary of the fashion trends and give a summary of the color trends, outfit combinations, pattern trends etc:
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
        loader = TextLoader("./imageCaptions.txt")
        docs = loader.load()
        return (stuff_chain.run(docs))

    def _run(self,prompt: str):
        output = self.chat_llm_chain.run(prompt)
        # validate output
        if self.validate_output(output) == False:
            raise Exception(f"Output is not in proper format: {output}")
        
        outfit_list =  eval(output)
        outfit: str = ""

        for item in outfit_list:
            search_result = self.search_flipkart(item)
            name = search_result[0]["name"]
            price = search_result[0]["current_price"]
            outfit += f"{item}: {name}, Rs. {price} \n"
        return outfit
    
    def search_flipkart(self, search_term):
        URL = f"https://9d5f-13-233-172-64.ngrok-free.app/search/{search_term}"
        response = requests.get(URL).json()
        response = response["result"]
        filtered_response = []
        for resp in response:
            filtered_response.append({
                "name": resp["name"],
                "current_price": resp["current_price"],
            })
        return filtered_response[:1]

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
    fashion_outfit_generator = FashionOutfitGenerator()
    print(fashion_outfit_generator._run("Create an outfit for a hindu wedding"))

if __name__ == "__main__":
    main()