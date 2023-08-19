import requests
from products import Products, session
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from config import *
import random


search_terms = [
    'lehenga',
    'saree',
    't-shirt',
    'shirt',
    'jeans',
    'shorts',
    'crop-tops',
    'jackets',
    'sweatshirts',
    'plazzo',
    'salwar',
    'churidar',
    'dhoti',
    'jumpsuit',
    'blazer',
    'coat',
    'sweater',
    'shrug',
    'Silver sandals',
    'Embroidered clutch',
    'Gold jhumka earrings',
    'Gold bangles',
    'brown mojari',
    'Nehru Jacket',
]

def search_flipkart( search_term):
    URL = f"https://9d5f-13-233-172-64.ngrok-free.app/search/{search_term}"
    response = requests.get(URL).json()
    responses = response["result"]
    return responses

def main():
    products = []
    print("Fetching products")
    for search_term in search_terms:
        print(f"Fetching products for {search_term}")
        products.extend(search_flipkart(search_term))
    texts = []
    metadatas = []
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY1)

    print("Adding to SQL database")
    for product_dict in products:
        thumbnails = product_dict["thumbnail"]
        if len(thumbnails) > 0:
            thumbnail = thumbnails[0]
        else:
            thumbnail = ""
        product = Products(
            name = product_dict["name"],
            link = product_dict["link"],
            current_price = product_dict["current_price"],
            original_price = product_dict["original_price"],
            discounted = product_dict["discounted"],
            thumbnail = thumbnail,
            rating = random.uniform(1, 5),
            popularity = random.uniform(0, 1),
        ) # type: ignore
        product.add(session=session)
        texts.append(product.name)
        metadatas.append({"uuid": product.id})
        
    print("Adding to vectorstore")
    vectorstore = Redis.from_texts(
        texts=texts,
        metadatas=metadatas,
        index_name=INDEX_NAME,
        embedding=embeddings,
        redis_url=REDIS_URL,
    )


if __name__ == "__main__":
    main()