from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader('DataSource\somatosensory.pdf')
pdfDocument = loader.load()
print(pdfDocument)

text_chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        length_function=len
        )


chunks = text_chunks.split_documents(pdfDocument)
print(chunks)