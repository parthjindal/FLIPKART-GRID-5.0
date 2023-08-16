import requests
from config import *
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool

outfit_model = ChatOpenAI(openai_api_key = OPENAI_API_KEY1,
                          temperature = 0,
                          verbose=VERBOSE)

class FlipkartSearchTool(BaseTool):
    def __init__(self) -> None:
        super().__init__()
        self.name = "flipkart_search_tool"
        self.description = "Use this tool to search for products on flipkart."
        self.memory = dict()

    def search_flipkart(self, search_term):
        URL = f"https://9d5f-13-233-172-64.ngrok-free.app/search/{search_term}"
        response = requests.get(URL).json()
        response = response["result"]
        filtered_response = []
        for resp in response:
            self.memory[resp["name"]] = resp
            filtered_response.append({
                "name": resp["name"],
                "current_price": resp["current_price"],
            })
        return filtered_response[:1]

    def get_product_details(self, product_name):
        return self.memory[product_name]

class IntelligentSearchTool(BaseTool):
    name = "intelligent_search_tool"
    description = "Use this tool to generate intelli"

def search_flipkart(self, search_terms, attributes):
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