import ollama
import time
from helpers import *

from config import configuration

from langchain.vectorstores import Chroma
import chromadb

persistent_client = chromadb.PersistentClient()


prompt = """### System:
You are a chatbot that assists users in answering questions about cryptocurrency & defi (decentralized finance).

You may use the below context to assist in answering questions.
{context}

Log of interaction so far :
{conversation}

Help the user answer their question as best as possible - you can use the context provided above to assist in answering the question.

### User:
{message}
"""


summaryPrompt = """### System:
Help summarise the conversation so far between the user and the chatbot.
{summaryInput}

### Assistant:
"""


def loadChromaCollections():

    sowChroma = Chroma(
        client=persistent_client,
        collection_name="stateOfWorld",
        embedding_function=configuration["embedding"],
    )

    return sowChroma


def getOllamaResponse(
    prompt, modelName="", trackTime=True, model=configuration["localModel"]
):
    start = time.time()
    _out = ollama.generate(model=model, prompt=prompt)
    _response = _out["response"]
    end = time.time()
    if trackTime == True:
        if configuration["enableLogging"]:
            print("Time to get response from Ollama : " + modelName + str(end - start))

    return _response


def manageChatLength(convList, maxConervationLength=configuration["maxConervationLength"]):
    conv_str = "".join(convList)


    if countTokens(conv_str) > configuration["maxConervationLength"]:
        n = len(conv_str)
        nToRemove = int(n * configuration["summaryPercent"])
        summaryInput = ""
        i = 0
        while len(summaryInput) < nToRemove:
            summaryInput += convList[i]
            summaryInput += convList[i + 1]
            i += 2

        _summaryPrompt = summaryPrompt.format(summaryInput=summaryInput)
        summary = getOllamaResponse(_summaryPrompt)

        newConvList = []
        newConvList.append(summary)

        for i in range(n) : 
            if (i > nToRemove) : 
                newConvList.append(convList[i])
        return newConvList
    
    else : 
        return convList

def chat() : 
    sowChroma = loadChromaCollections()
    conversation = ""
    convList = []
    while True : 
        

        context = ""
        userMessage = input("Your Message : ")
        convList.append( "User : " + userMessage)
        _results = sowChroma.similarity_search(userMessage)
        nResults = len(_results) 
        for i in range(nResults) : 
            _out = convert_to_normalized_sentence(_results[i].page_content)
            print("Result " + _out)
            context += _out + "\n"
        _response = getOllamaResponse(prompt.format(context=context, conversation=conversation, message=userMessage))
        print("Response : " + _response)

        convList.append("System : " + _response)
        context += _response + "\n"
        convList = manageChatLength(convList)
        conversation = "".join(convList)


chat()