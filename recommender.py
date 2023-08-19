from config import *
from langchain.vectorstores.redis import Redis
from langchain.embeddings.openai import OpenAIEmbeddings
import os
os.environ["openai_api_key"] = OPENAI_API_KEY3


def RedisVectorStoreRetrieverGetter(search_type="similarity"):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY3)
    vectorstore = Redis(
        index_name=INDEX_NAME,
        embedding_function=embeddings.embed_query,
        redis_url=REDIS_URL,
    )
    return vectorstore.as_retriever(search_type=search_type)