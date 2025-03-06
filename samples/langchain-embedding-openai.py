## OpenAI create embeddings for langchain
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

text = "This is a test document"
query_result = embeddings.embed_query(text)
print(query_result)
