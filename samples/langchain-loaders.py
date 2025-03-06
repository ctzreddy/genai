# Text loader for langchain
from langchain_community.document_loaders import TextLoader
loader = TextLoader('sample.txt')
textDocument = loader.load()
print(textDocument)


## PDf loader for langchain
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('DataSource\somatosensory.pdf')
pdfDocument = loader.load()
print(pdfDocument)

# Web loader for langchain
from langchain_community.document_loaders import WebBaseLoader 
loader = WebBaseLoader('https://python.langchain.com/docs/introduction/')
webDocument = loader.load()
print(webDocument)

