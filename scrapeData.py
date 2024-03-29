from models import embedding

from scraping.scraper import scrapeTopics, topics
from scraping.processDocs import processDocs

# Scrape topics and write to DB
toProcessDocs = False

scrapeTopics(embedding, topics = topics, useNews=True, persist_directory = 'chroma')

if toProcessDocs : 
    processDocs(embedding, loadDir = './docs', persist_directory = 'chroma'  )