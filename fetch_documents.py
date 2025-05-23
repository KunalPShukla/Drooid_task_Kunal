import os
import sys
from pymongo import MongoClient
import certifi
from langchain.docstore.document import Document
import re  # Required for whitespace cleanup


# MongoDB connection
MONGO_DB_URL = "mongodb+srv://kpshukla3:QAElI81frJLKq3eo@drooidinputfile.feshorw.mongodb.net/?retryWrites=true&w=majority&appName=DrooidInputFile"

class MongoArticleFetcher:
    def __init__(self, db_name: str, collection_name: str):
        try:
            self.client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
            self.collection = self.client[db_name][collection_name]
        except Exception as e:
            raise Exception(f"MongoDB connection failed: {e}")


    def fetch_documents(self):
        try:
            # Fetch all documents with articleBody present
            docs = list(self.collection.find({"articleBody": {"$exists": True}}))

            langchain_docs = [
                Document(
                    page_content=re.sub(r'\s+', ' ', doc["articleBody"]).strip(),
                    metadata={
                        "source": doc.get("source", ""),
                        "title": doc.get("title", "")
                    }
                )
                for doc in docs
            ]
            return langchain_docs
        except Exception as e:
            raise Exception(f"Error fetching documents: {e}")


if __name__ == "__main__":
    fetcher = MongoArticleFetcher("qa_data", "news_articles")
    articles = fetcher.fetch_documents()
    print(f"Fetched {len(articles)} articles as LangChain Documents.")
    print("\nSample article:\n", articles[0].page_content[:500])  # Show a preview
