from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from ..config import OPENAI_API_KEY, VERBOSE

class TrendSummarizerLLMChain():
    def __init__(self, llm, trends_store_path):
        """
        Trend Summarizer uses a LLM to summarize the current fashion trends
        based on the user prompt.
        Args:
            llm (BaseLLM): The language model to use for summarizing
            trends_store_path (str): The path to the file containing the current fashion trends
        """
        prompt_template = """
You are given the current social media fashion trends in the below text.
Provide a summary of the fashion trends including the color trends, outfit combinations, pattern trends etc. by 
matching the closest fashion trend from the text with the given User Query in max 3 sentences.
Fashion Trends: \"""
{text}
\"""
User Query: \"""
{query}
\"""
CURRENT SOCIAL MEDIA FASHION TRENDS:"""
        llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
        prompt_template
        self.summarizer = StuffDocumentsChain(llm_chain=llm_chain,
                                              document_variable_name="text", verbose = VERBOSE)
        loader = TextLoader(trends_store_path)
        self.documents = loader.load()

    def run(self, query):
        return self.summarizer.run(input_documents=self.documents, query=query)

    @staticmethod
    def test():        
        llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0,openai_api_key=OPENAI_API_KEY)
        summarizer = TrendSummarizerLLMChain(llm, "./grid/vogueSummary.txt")
        print(summarizer.run("What are the current fashion trends for weddings?"))


class PurchasePatternSummarizerLLMChain():
    PROMPT_TEMPLATE = """
You are given the purchase history of a user as a list of products in the below text. \
You have to summarize and find the preferences of the user such as color, pattern, type etc.:
"{text}"
USER PREFERENCES:"""
    
    def __init__(self, llm, purchase_history_path):
        """
        Purchase Pattern Summarizer uses a LLM to summarize the purchase patterns of a user
        based on the user's purchase history, browsing history etc.
        Args:
            llm (BaseLLM): The language model to use for summarizing
            purchase_history_path (str): The path to the file containing the purchase history of the user
        """

        llm_chain = LLMChain(
            llm=llm, 
            prompt=PromptTemplate.from_template(self.PROMPT_TEMPLATE)
        )
        self.summarizer = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="text",
            verbose = VERBOSE,
        )
        loader = TextLoader(purchase_history_path)
        self.documents = loader.load()
        self.purchase_pattern_summary = None

    def run(self):
        # cache it since doesn't change for a user
        if self.purchase_pattern_summary is None:
            self.purchase_pattern_summary = self.summarizer.run(
                input_documents=self.documents)
        return self.purchase_pattern_summary
    
class UserProfileSummarizer():
    def __init__(self, age, city, gender):
        """
        User Profile Summarizer returns a summary of the user profile
        """
        self.user_profile = f"The user is {age} year old {gender} from {city}"

    def run(self):
        return self.user_profile