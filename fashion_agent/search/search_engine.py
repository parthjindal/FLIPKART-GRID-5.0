from ..config import *
from langchain.vectorstores import VectorStore
from langchain.vectorstores.redis import Redis
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import List,Dict, Tuple
import logging
from enum import Enum
from abc import ABC, abstractmethod
from sqlalchemy.orm import Session
from .. import logger

class OrderBy(Enum):
    SIMILARITY = "similarity"
    CURRENT_PRICE = "current_price"

class Order(Enum):
    ASC = "asc"
    DESC = "desc"

class FilterType(Enum):
    RATING = "rating"
    POPULARITY = "popularity"
    PRICE = "price"

class Filter(ABC):
    def __init__(self, filter_type: FilterType, *args, **kwargs):
        self.filter_type = filter_type

    @abstractmethod
    def __repr__(self):
        pass

class RatingFilter(Filter):
    def __init__(self, rating: float
                 , *args, **kwargs):
        super().__init__(filter_type=FilterType.RATING, *args, **kwargs)
        self.rating = rating

    def __repr__(self):
        return f"rating >= {self.rating}"

class PopularityFilter(Filter):
    def __init__(self, popularity: float
                 , *args, **kwargs):
        super().__init__(filter_type=FilterType.POPULARITY, *args, **kwargs)
        self.popularity = popularity

    def __repr__(self):
        return f"popularity >= {self.popularity}"
    

class PriceFilter(Filter):
    def __init__(self, price_range: Tuple
                 , *args, **kwargs):
        """
        Arguments:
            price_range: Tuple of (min_price, max_price)
            if min_price == 0, then min_price is not considered
            if max_price == 0, then max_price is not considered
        """
        super().__init__(filter_type=FilterType.PRICE, *args, **kwargs)
        self.price_range = price_range

    def __repr__(self):
        if self.price_range[0] == self.price_range[1]:
            return f"current_price == {self.price_range[0]}"
        if self.price_range[0] == 0:
            return f"current_price <= {self.price_range[1]}"
        if self.price_range[1] == 0:
            return f"current_price >= {self.price_range[0]}"
        return f"current_price >= {self.price_range[0]} AND current_price <= {self.price_range[1]}"

class SearchEngine():
    """
    Search Engine is an interface that contains methods to search products
    on Flipkart using a vectorstore and sql database containing product information.
    To search a term, the term is first embedded using the OpenAIEmbeddings class
    and then the vectorstore is searched for similar vectors. The results are then
    filtered using the sql database to get the products that are most similar to the
    search term. This combines the power of vector search with the power of sql databases 
    allowing for meaningful search results along with ability to sort and filter the results.
    """
    def __init__(self,
                 embeddings: OpenAIEmbeddings,
                 vectorstore: Redis,
                 session: Session,
                 max_documents: int = 10,
                 similarity_thresh: float = 0.45) -> None:
        """
        Arguments:
            embeddings: OpenAIEmbeddings object
            vectorstore: VectorStore object
            max_documents: Maximum number of documents to search
            similarity_thresh: Threshold for similarity
        """
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.session = session
        self.max_documents = max_documents
        self.similarity_thresh = similarity_thresh

    def _construct_search_query(
        self, 
        uuids: List[str],
        order_by: str,
        order: str,
        limit: int,
        filters: List[Filter],
    ) -> str:
        uuids_str = ""
        for uuid in uuids:
            uuids_str += f"'{uuid}',"
        uuids_str = uuids_str[:-1]
        uuids_str = "(" + uuids_str + ")"
        if order_by == "similarity":
            suffix = f"LIMIT {limit}"
        else:
            suffix = f"ORDER BY {order_by} {order} LIMIT {limit}"
        query = f"""
SELECT * FROM products WHERE id IN {uuids_str}"""
        for filter in filters:
            query += f"\n   AND {filter}"
        
        query += f" {suffix}"
        query += ";"
        return query
    
    def search(
        self,
        search_term: str,
        order_by: OrderBy = OrderBy.SIMILARITY,
        order: Order = Order.DESC,
        limit: int = 10,
        filters: List[Filter] = []
    ) -> List[Dict]:
        """
        Arguments:
            search_term: Search term
            order_by: Order by which to sort the results
            order: Order of the results
            limit: Maximum number of results to return
            filters: List of filters to apply to the results

        """
        logging.debug("Searching in vectorstore")
        results = self.vectorstore.search(
            search_term,
            search_type="similarity",
            k=self.max_documents,
            threshold=self.similarity_thresh,
        )
        retriever = self.vectorstore.as_retriever()
        retriever.get_relevant_documents
        logging.debug("Search in vectorstore complete")
        uuids = [result.metadata["uuid"] for result in results]
        
        query = self._construct_search_query(
            uuids=uuids,
            order_by=order_by.value,
            order=order.value,
            limit=limit,
            filters=filters,
        )
        logging.debug(f"Search Query: {query}")
        try:
            resp = self.session.execute(query)
            products: List = resp.fetchall()
        except Exception as e:
            logging.error(f"Error while executing query {query}: {e}")
            raise e
 
        productsDict = []
        for product in products:
            productsDict.append({
                "name": product.name,
                "link": product.link,
                "current_price": product.current_price,
                "original_price": product.original_price,
                "discounted": product.discounted,
                "thumbnail": product.thumbnail,
            })
        logging.debug(f"Search results: {productsDict}")
        return productsDict


def test():
    from ..models.products import session
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY1)
    vectorstore = Redis.from_existing_index(index_name=INDEX_NAME,
                                             redis_url=REDIS_URL,
                                             embedding=embeddings)
    se = SearchEngine(embeddings=embeddings,
                      vectorstore=vectorstore,
                      session=session,
                      max_documents=100, similarity_thresh=0.6)
    print(se.search("jumpsuit",
                    order_by=OrderBy.SIMILARITY, 
                    filters=[RatingFilter(2), PopularityFilter(0.9), PriceFilter((200, 1000))]))


if __name__ == "__main__":
    test()