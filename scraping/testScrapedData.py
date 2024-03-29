
from langchain.vectorstores import Chroma
import chromadb

from config import embedding

persist_directory = 'chroma'
persistent_client = chromadb.PersistentClient()


_retriveChroma = Chroma(
    client=persistent_client,
    collection_name="stateOfTheWorld",
    embedding_function=embedding,
)

def convert_to_normalized_sentence(multiline_string):
    words = multiline_string.split()
    return ' '.join(words)


while True : 
    
    _newQuery = input("Query Input : ")
    _results = _retriveChroma.similarity_search(_newQuery)
    nResults = len(_results) 
    for i in range(nResults) : 
        _out = convert_to_normalized_sentence(_results[i].page_content)
        print("Result " + _out)
    #print("Results : " + str(_results))
