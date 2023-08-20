from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate


class OutputValidatorLLMChain():
    def __init__(self, llm):
        prompt = """
You are given the following output for a fashion outfit recommendation task
along with the input prompt. You have to validate the output based on the description, name of the product and price. \
If the output is not valid, Say 'Couldn't find a valid outfit. Please try again.' Otherwise simply return the output. 
Input prompt:
\"""
{prompt}
\"""
Output:
\"""
{output}
\"""
Validated Output:
"""
        self.llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))

    def run(self, prompt, output):
        return self.llm_chain.run(prompt=prompt, output=output)
    
    