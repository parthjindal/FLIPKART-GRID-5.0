from langchain.prompts.chat import SystemMessage,  HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import BaseLLM
from langchain.chains.llm import LLMChain
from typing import Dict
from langchain.chat_models import ChatOpenAI
from ..config import *

class FashionOutfitRecommenderLLMChain():
    def __init__(
        self,
        llm: BaseLLM,
        memory: ConversationBufferMemory,
        user_profile: str,
        user_purchase_history: str,
        fashion_trends: str,
        *args,
        **kwargs,
    ):
        system_message = SystemMessage(content=f"""
You are a fashion outfit recommender. Based on the input prompt, \
Either create a complete outfit consisting of clothing, accessories and footwear or make changes to an existing outfit or an outfit piece. \
You may also use the user profile, purchase history and current fashion trends to make the recommendations more personalized. \
The item names must be detailed and can mention colors, patterns and other search terms.
                                       
User profile: '{user_profile}'
User purchase History: '{user_purchase_history}'
Current fashion trends: '{fashion_trends}'

Output JSON format:
{{
    'items': ['item1', 'item2', 'item3'], # A python list containing only the names of the items.
}}

Example:
{{
    'items': ['men red graphic shirt', 'men blue ripped jeans', 'men black shoes'] 
}}
""")
        
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injectd
            HumanMessage(content="Tips: Make sure to stick to the output JSON format."),
            AIMessage(content="Output JSON: ")
        ])
        self.llm_chain = LLMChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            **kwargs,
        )

    def run(self, prompt) -> Dict:
        print(prompt)
        response = self.llm_chain.run(prompt)
        try:
            response_dict = eval(response)
            if "items" not in response_dict:
                raise Exception(f"Output JSON does not contain 'items' key: {response}")
        except:
            raise Exception(f"Output is not a valid JSON: {response}")
        return response_dict

    @staticmethod
    def test():
        fashion_trends = "The current fashion trends is to use floral print and vibrant colored outfits."
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0,OPENAI_API_KEY=OPENAI_API_KEY)
        memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
        recommender = FashionOutfitRecommenderLLMChain(llm,memory,"","",fashion_trends,verbose=VERBOSE)
        print(recommender.run("Generate outfit for a hindu wedding."))
        print(recommender.run("recommend something else instead of gold sandals with rating > 4"))
        print(recommender.run("recommend a completely new outfit. Make sure the price of each item is less than 1000 and the rating is greater than 3"))


if __name__ == "__main__":
    FashionOutfitRecommenderLLMChain.test()