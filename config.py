from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

localEmbedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
openAiEmbedding = OpenAIEmbeddings(model="text-embedding-ada-002")

useOpenAIEmbedding = False

if useOpenAIEmbedding :
    embedding = openAiEmbedding
else :
    embedding = localEmbedding

localModel = "tinyllama"

maxConervationLength = 700 
summaryPercent = 0.6 # what percent of the conversation to strip out & summarise mid conversation

maxTokensOut = 300 # max tokens to generate from the model

configuration = {
    "localModel" : localModel,
    "maxConervationLength" : maxConervationLength,
    "summaryPercent" : summaryPercent,
    "maxTokensOut" : maxTokensOut,
    "useOllama" : True,
    "embedding" : embedding,
    "enableLogging" : False,
}