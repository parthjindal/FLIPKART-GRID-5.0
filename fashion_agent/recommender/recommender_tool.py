from ..config import *
from ..search import SearchEngine, SearchEngineWithLLM
from ..logger import logger
from .summarizer import (
    TrendSummarizerLLMChain, 
    PurchasePatternSummarizerLLMChain,
    UserProfileSummarizer
)
from .recommender_chain import FashionOutfitRecommenderLLMChain
from .validator import OutputValidatorLLMChain
from ..models.products import session

from langchain.llms.base import BaseLLM
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from typing import Callable
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
import langchain


class SearchCache:
    def __init__(self):
        self.cache = dict()
   
    def add(self,key,item):
        self.cache[key] = item

    def clear(self):
        self.cache.clear()

    def get(self,key):
        return self.cache.get(key,0)
    
    def get_all(self):
        return self.cache.values()


class FashionOutfitGeneratorTool(BaseTool):
    name = "fashion_outfit_generator"
    description = """
Use this tool to generate a fashion outfit consisting of clothing, accessories and footwear from a shopping website \
or make changes to an existing outfit or an outfit piece. \
Input: `prompt`(str) A first person query about creating an outfit or making changes to \
an existing outfit or an outfit piece based on the user request along with any user preferences about price, popularity, rating etc.
Input must only be a single string 'prompt'!.
"""
    memory: ConversationBufferMemory = None
    llm: BaseLLM = None
    recommender_getter: Callable = None
    user_profile_summarizer: UserProfileSummarizer = None
    trend_summarizer: LLMChain = None
    purchase_pattern_summarizer: LLMChain = None
    output_validator: LLMChain = None
    search_cache1: SearchCache = None
    search_cache2: SearchCache = None
    search_engine: SearchEngineWithLLM = None

    def __init__(
        self,
        memory: ConversationBufferMemory,
        llm: BaseLLM,
        recommender_getter: Callable,
        user_profile_summarizer: UserProfileSummarizer,
        trend_summarizer: LLMChain,
        purchase_pattern_summarizer: LLMChain,
        output_validator: LLMChain,
        search_engine: SearchEngineWithLLM,
        search_cache1: SearchCache,
        search_cache2: SearchCache,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = memory
        self.llm = llm
        self.recommender_getter = recommender_getter
        self.user_profile_summarizer = user_profile_summarizer
        self.trend_summarizer = trend_summarizer
        self.purchase_pattern_summarizer = purchase_pattern_summarizer
        self.output_validator = output_validator
        self.search_engine = search_engine
        self.search_cache1 = search_cache1
        self.search_cache2 = search_cache2
    
    def _run(self, prompt: str):
        """Use the tool."""
        user_profile = self.user_profile_summarizer.run()
        trend_summary = self.trend_summarizer.run(prompt)
        purchase_pattern_summary = self.purchase_pattern_summarizer.run()
        recommender = self.recommender_getter(
            llm=self.llm,
            memory=self.memory,
            user_profile=user_profile,
            user_purchase_history=purchase_pattern_summary,
            fashion_trends=trend_summary,
        )
        response = recommender.run(prompt)
        outfit = response["items"]
        attributes = response["attributes"]

        changed_items = []
        print(outfit, attributes)
        for item in outfit:
            if self.search_cache1.get(item) == 0:
                changed_items.append(item)

        search_results = self.search_engine.search(changed_items, attributes)
        
        self.search_cache2.clear()
        recommendation_results = ""
        for search_term, search_result in zip(changed_items, search_results):
            if search_result == "NAN":
                
                continue
            self.search_cache1.add(search_term, search_result)
            name = search_result["name"]
            current_price = search_result["current_price"]
            original_price = search_result["original_price"]
            recommendation_results += f"{search_term}: {name}, Rs. {current_price} (Original Price Rs. {original_price})>\n"
        
        temp_cache = SearchCache()
        for item in outfit:
            temp_cache.add(item, self.search_cache1.get(item))
        self.search_cache1.cache = temp_cache.cache
        
        for item in changed_items:
            self.search_cache2.add(item, self.search_cache1.get(item))

        if recommendation_results == "":
            recommendation_results = "No recommendations found"
        
        return recommendation_results

def build_fashion_outfit_generator_tool():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    trend_summarizer_llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k", 
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    trend_summarizer = TrendSummarizerLLMChain(
        llm=trend_summarizer_llm,
        trends_store_path=TRENDS_STORE_PATH,
    )
    user_profile_summarizer = UserProfileSummarizer(
        age="23",
        city="Delhi",
        gender="Male",
    )
    purchase_pattern_summarizer_llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k", 
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    purchase_pattern_summarizer = PurchasePatternSummarizerLLMChain(
        purchase_pattern_summarizer_llm,
        PURCHASE_HISTORY_PATH,
    )

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY1)
    vectorstore = Redis.from_existing_index(index_name=INDEX_NAME,
                                             redis_url=REDIS_URL,
                                             embedding=embeddings)
    search_engine = SearchEngine(
        embeddings=embeddings,
        vectorstore=vectorstore,
        session=session,    
        max_documents=MAX_DOCUMENTS,
        similarity_thresh=SIMILARITY_THRESH,
    )

    search_llm = ChatOpenAI(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        openai_api_key=OPENAI_API_KEY1,
    )
    search_engine_with_llm = SearchEngineWithLLM(
        base_llm=search_llm,
        search_engine=search_engine,
    )

    recommender_llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        openai_api_key=OPENAI_API_KEY1,
    )
    
    search_cache1 = SearchCache()
    search_cache2 = SearchCache()
    output_validator = OutputValidatorLLMChain(recommender_llm) # currently turned off
    return FashionOutfitGeneratorTool(
        memory=memory,
        llm=recommender_llm,
        recommender_getter=FashionOutfitRecommenderLLMChain,
        user_profile_summarizer=user_profile_summarizer,
        trend_summarizer=trend_summarizer,
        purchase_pattern_summarizer=purchase_pattern_summarizer,
        output_validator=output_validator,
        search_engine=search_engine_with_llm,
        search_cache1=search_cache1,
        search_cache2=search_cache2,
    )


def test():
    tool = build_fashion_outfit_generator_tool()
    print(tool._run("Create an outfit for a christian ring ceremony"))

if __name__ ==  "__main__":
    test()