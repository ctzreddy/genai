## Hugging face embedding for langchain
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ['OPENAI_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')
