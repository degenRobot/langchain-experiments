from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# Option to load from a directory of PDFs & get text to feed into DB 
def processDocs(embedding, loadDir = './docs', persist_directory = 'chroma'  ) : 
    loader = DirectoryLoader('./docs/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(documents=texts, 
                                    embedding=embedding,
                                    collection_name = "stateOfTheWorld",
                                    persist_directory=persist_directory)

