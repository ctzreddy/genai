import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("50_page_sample.pdf")
template = """"Write a summary of the following text:
Text: {text}
"""
prompt=PromptTemplate(
    input_variables=["text"], template=template)

from langchain.chains.summarize import load_summarize_chain
chain=load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
output=chain.invoke({"text": loader.load()})
print(output.content)

# Map reducer
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
final_docs = text_splitter.split_documents(text_splitter)

chunksprompt = PromptTemplate(
    input_variables=["text"], template=template
)

summary_chain = load_summarize_chain(
    llm, chain_type="map_reduce",
    map_prompt=chunksprompt)

final_prompt = """
Provide a final summary fo the entire document based on the following summaries:
{summaries}
"""

final_prompt_template = PromptTemplate(
    input_variables=["summaries"], 
    template=final_prompt
)

final_summary_chain = load_summarize_chain(
    llm, 
    chain_type="map-reduce",
    map_prompt=chunksprompt, 
    combine_prompt=final_prompt_template,
    verbose=True
)

output = final_summary_chain.invoke({"text": final_docs})

