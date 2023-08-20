from .search_engine import (
    SearchEngine, 
    OrderBy,
    Order,
    RatingFilter,
    PopularityFilter,
    PriceFilter
)
from ..config import *
from langchain.vectorstores.redis import Redis
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessage, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from typing import List,Dict
from ..models import session

SEARCH_ENGINE_ARGS_SCHEMA = {
    "title": "SearchEngineArgs",
    "description": "Identify the arguments for the search engine",
    "type": "object",
    "properties": {
        "order_by": {"title": "order_by", "description": "Order the results by", "type": "string", "enum": ["similarity", "current_price", "popularity"]},
        "is_descending": {"title": "is_descending", "description": "Order in descending order or not", "type": "boolean"},
        "rating_filter": {"title": "rating_filter", "description": "Filter by rating", "type": "integer","minimum": 1, "maximum": 5},
        "popularity_filter": {"title": "popularity_filter", "description": "Filter by popularity", "type": "number", "minimum": 0, "maximum": 1},
        "min_price": {"title": "min_price", "description": "Filter by minimum price", "type": "number"},
        "max_price": {"title": "max_price", "description": "Filter by maximum price", "type": "number"},
    },
    "required": [],
}


class SearchEngineWithLLM():
    """
    A wrapper around the search engine that uses a language model to extract the arguments for the search engine
    """
    def __init__(
        self, 
        base_llm: BaseLLM,
        search_engine: SearchEngine,
        *args, 
        **kwargs
    ) -> None:
        """
        Args:
            base_llm (BaseLLM): The language model to use for extracting arguments
            search_engine (SearchEngine): The search engine to use for searching
        """
        self.search_engine = search_engine
        self.base_llm = base_llm
        prompt_msgs = [
            SystemMessage(
                content="You are a world class algorithm for extracting information in structured formats."
            ),
            HumanMessage(content="Use the given format to extract information from the following input:"),
            HumanMessagePromptTemplate.from_template("Input: {input}"),
            HumanMessage(content="Tips: Make sure to answer in the correct format"),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)
        self.llm_chain = create_structured_output_chain(SEARCH_ENGINE_ARGS_SCHEMA, base_llm, prompt, verbose=True)

    def construct_se_kwargs(self, kwargs: Dict) -> Dict:
        se_kwargs = {"filters": []}
        if kwargs.get("order_by"):
            se_kwargs["order_by"] = OrderBy(kwargs["order_by"])
        if kwargs.get("is_descending"):
            se_kwargs["order"] = Order.DESC if kwargs["is_descending"] else Order.ASC
        if kwargs.get("rating_filter"):
            se_kwargs["filters"].append(RatingFilter(kwargs["rating_filter"]))            
        if kwargs.get("popularity_filter"):
            se_kwargs["filters"].append(PopularityFilter(kwargs["popularity_filter"]))
        if kwargs.get("min_price") or kwargs.get("max_price"):
            se_kwargs["filters"].append(PriceFilter((kwargs.get("min_price",0),kwargs.get("max_price",0))))
        return se_kwargs
    
    def search(self, search_terms, input_prompt) -> List:
        search_results = []
        kwargs = self.llm_chain.run(input=input_prompt)
        se_kwargs = self.construct_se_kwargs(kwargs)

        for search_term in search_terms:
            se_kwargs["search_term"] = search_term
            search_result = self.search_engine.search(**se_kwargs)
            if len(search_result) == 0:
                search_results.append("NAN")
                continue
            search_results.append(search_result[0])
    
        return search_results

def test():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY1)
    vectorstore = Redis.from_existing_index(index_name=INDEX_NAME,
                                             redis_url=REDIS_URL,
                                             embedding=embeddings)
    se = SearchEngine(
        embeddings=embeddings,
        vectorstore=vectorstore,
        session=session,
        max_documents=100,
        similarity_thresh=0.6
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0,openai_api_key=OPENAI_API_KEY)
    seV2 = SearchEngineWithLLM(base_llm=llm, search_engine=se)
    print(seV2.search(["Maroon kurta", "Gold Bangles"], "rating greater than 4 and price less than 400 and ordered by price in descending order"))


if __name__ == "__main__":
    test()
