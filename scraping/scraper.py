from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# Topics to loop through and scrape data from 
topics = {
    'Defi Market' : 'Defi Market',
    'Defi Regulation' : 'Defi Regulation',


}

# Loop through topics search & write to DB 
def scrapeTopics(embedding, topics = topics, useNews=True, persist_directory = 'chroma') : 
    if useNews : 
        search = GoogleSerperAPIWrapper(type="news",tbs="qdr:d")
    else : 
        search = GoogleSerperAPIWrapper(tbs="qdr:d")
    

    texts = []

    for topic in topics.keys() :
        print("Topic : " + topic)
        print("Question : " + topics[topic])
        res = search.results(topics[topic])
        # Clean up results to write to Chroma
        if useNews :
            results = res['news']
        else : 
            results = res['organic']
        #results = res['news']

        for i in range(len(results)) : 
            #outList.append(organicResults[i]['title'] + " " + organicResults[i]['snippet'])
            out = results[i]['title'] + " " + results[i]['snippet']
            print(out)
            doc = Document(page_content=out)
            doc.metadata = {
                'topic' : topic
            }
            texts.append(doc)
        #searchResults[topic] = outList

    # Write to DB 
        
    vectordb = Chroma.from_documents(documents=texts, 
                                    collection_name = "stateOfTheWorld",
                                    embedding=embedding,
                                    persist_directory=persist_directory)

