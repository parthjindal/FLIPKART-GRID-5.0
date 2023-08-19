from config import *
from langchain.vectorstores.redis import Redis
from langchain.embeddings.openai import OpenAIEmbeddings
import requests
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid
from typing import List
import logging

# Define the SQLAlchemy Base class
Base = declarative_base() 
class Products(Base):
    __tablename__ = 'products'
    id = Column(String(36), primary_key=True)
    name = Column(String, nullable=False)
    link = Column(String, nullable=False)
    current_price = Column(Integer, nullable=False)
    original_price = Column(Integer, nullable=False)
    discounted = Column(Integer)
    thumbnail = Column(String)
    query_url = Column(String)

def SQLSessionGetter():
    engine = create_engine(f'sqlite:///{SQL_DB_PATH}')
    Base.metadata.create_all(engine) # type: ignore
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def SearchOnFlipKart(search_term):
    URL = f"https://9d5f-13-233-172-64.ngrok-free.app/search/{search_term}"
    response = requests.get(URL).json()
    responses = response["result"]
    return responses


class SearchEngine():
    def __init__(self, is_init: bool = False, k: int = 10, threshold: float = 0.45):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY1)
        self.vectorstore = None
        if is_init == False:
            self.vectorstore = Redis.from_existing_index(index_name=INDEX_NAME,
                                                         redis_url=REDIS_URL,
                                                         embedding=self.embeddings)
        self.session = SQLSessionGetter()
        self.k = k
        self.threshold = threshold

    def addToRedis(self, texts, metadatas):
        if self.vectorstore is not None:
            embeddings = self.embeddings.embed_documents(texts)
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas, embeddings=embeddings)
        else:
            self.vectorstore = Redis.from_texts(
                texts=texts,
                metadatas=metadatas,
                index_name=INDEX_NAME,
                embedding=self.embeddings,
                redis_url=REDIS_URL,
            )
        return self.vectorstore

    def add(self, productJSON: dict):
        thumbnail = "" 
        if len(productJSON["thumbnail"]):
            thumbnail = productJSON["thumbnail"][0]

        product = Products(
            id = str(uuid.uuid4()),
            name=productJSON["name"],
            link=productJSON["link"],
            current_price=productJSON["current_price"],
            original_price=productJSON["original_price"],
            discounted=productJSON["discounted"],
            thumbnail=thumbnail,
            query_url=productJSON["query_url"]
        ) # type: ignore
        self.session.add(product)
        self.session.commit()
        
        return product.id

    def addAll(self, productsJSON: List[dict]):
        texts = []
        metadatas = []
        for productJSON in productsJSON:
            uniqueID = self.add(productJSON)
            texts.append(productJSON["name"])
            metadatas.append({"uuid": uniqueID})
        self.vectorstore = self.addToRedis(texts=texts, metadatas=metadatas)

    def _construct_search_query(self, uuids, order_by: str = "current_price", order: str = "asc", limit: int = 10):
        uuids_str = ""
        for uuid in uuids:
            uuids_str += f"'{uuid}',"
        uuids_str = uuids_str[:-1]
        uuids_str = "(" + uuids_str + ")"
        query = f"""
        SELECT * FROM products WHERE id IN {uuids_str} ORDER BY {order_by} {order} LIMIT {limit};
        """
        return query

    def search(self, query: str, order_by: str = "current_price", order: str = "asc", limit: int = 10):
        logging.debug(f"Searching for {query}")
        results = self.vectorstore.search(query,search_type="similarity",k=self.k,threshold=self.threshold)
        uuids = [result.metadata["uuid"] for result in results]

        query = self._construct_search_query(uuids=uuids, order_by=order_by, order=order, limit=limit)
        resp = self.session.execute(query)
        products: List[Products] = resp.fetchall()
        productsJSON = []
        
        logging.debug(f"Found {len(products)} products")
        for product in products:
            productsJSON.append({
                "name": product.name,
                "link": product.link,
                "current_price": product.current_price,
                "original_price": product.original_price,
                "discounted": product.discounted,
                "thumbnail": product.thumbnail,
                "query_url": product.query_url
            })
        return productsJSON


def search_flipkart( search_term):
    URL = f"https://9d5f-13-233-172-64.ngrok-free.app/search/{search_term}"
    response = requests.get(URL).json()
    responses = response["result"]
    return responses

se = SearchEngine(is_init=False,k = 10, threshold=0.4)
search_terms = [
    # 'lehenga',
    # 'saree',
    # 't-shirt',
    # 'shirt',
    # 'jeans',
    # 'shorts',
    # 'crop-tops',
    # 'jackets',
    # 'sweatshirts',
    # 'plazzo',
    # 'salwar',
    # 'churidar',
    # 'dhoti',
    # 'jumpsuit',
    # 'blazer',
    # 'coat',
    # 'sweater',
    # 'shrug',
    'Silver sandals',
    'Embroidered clutch',
    'Gold jhumka earrings',
    'Gold bangles',
]

# allProducts = []
# for search_term in search_terms:
#     products = search_flipkart(search_term)
#     allProducts.extend(products)
# se.addAll(allProducts)
# print(se.search(input("Enter search term:")))