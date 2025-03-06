import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
api_key = os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="Gemma-7b-It")

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

sample = sample.txt

from langchain.chains import LLMChain
from langchain import PromptTemplate

generictemplate = """
Write a summary of the following text:
Text: {text}
Translate the following text to {language}:
"""

prompt = PromptTemplate(
    input_variables=["text", "language"],
    template=generictemplate

)

complete_prmopt = prompt.format(text=sample, language="French")

llm_chain = LLMChain(llm, prompt)
summary = llm_chain.invoke(complete_prmopt)