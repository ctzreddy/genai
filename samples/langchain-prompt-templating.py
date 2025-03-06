## Chaining languages with prompt templating

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

## Load text document
loader = TextLoader('sample.txt')
textDocument = loader.load()

## Split text document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)
chunks = text_splitter.split_documents(textDocument)

# Convert text document to embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Store embeddings in a FAISS vector store
vectordb = FAISS.from_documents(chunks, embeddings)

## querying vector store
query = "What is Lorem Ipsum?"
docs = vectordb.similarity_search(query)

# Retrieval Chain, Document Chain
from langchain.chains.combin_documnets import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(model="gpt-4o")

prompt=ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")

document_chain=create_stuff_documents_chain(llm, prompt)




