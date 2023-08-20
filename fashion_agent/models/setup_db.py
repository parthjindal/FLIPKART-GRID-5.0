import requests
from .products import Products, session
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from ..config import *
import random

search_terms = []

with open("fashion_agent/models/terms.txt", mode="r") as f:
    search_terms = f.read().splitlines()

def search_flipkart( search_term):
    URL = f"https://flipkart-scraper-api.dvishal485.workers.dev/search/{search_term}"
    response = requests.get(URL).json()
    responses = response["result"]
    return responses

def add_to_db(products, vectorstore: Redis, embeddings: OpenAIEmbeddings):
    print("Adding to SQL database")
    texts = []
    metadatas = []
    for product_dict in products:
        product = Products(
            name = product_dict["name"],
            link = product_dict["link"],
            current_price = product_dict["current_price"],
            original_price = product_dict["original_price"],
            discounted = product_dict["discounted"],
            thumbnail = product_dict["thumbnail"],
            rating = random.uniform(3.5, 5),
            popularity = random.uniform(0.5, 0.99),
        ) # type: ignore
        product.add(session=session)
        texts.append(product.name)
        metadatas.append({"uuid": product.id})
        
    print("Adding to vectorstore")
    embedding_texts = embeddings.embed_documents(texts)
    vectorstore.add_texts(texts=texts,embeddings=embedding_texts,metadatas=metadatas)

def main():
    products = []
    print("Fetching products")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    redis = Redis(
        index_name=INDEX_NAME,
        redis_url=REDIS_URL,
        embedding_function=embeddings.embed_query,
    )
    redis._create_index()
    redis = Redis.from_existing_index(
        index_name=INDEX_NAME,
        redis_url=REDIS_URL,
        embedding=embeddings,
    )
    for search_term in search_terms:
        print(f"Fetching products for {search_term}")
        products.extend(search_flipkart(search_term))
        if len(products) >= 500:
            add_to_db(products, redis, embeddings)
            products = []
    
    if len(products) > 0:
        add_to_db(products, redis, embeddings)

if __name__ == "__main__":
    main()