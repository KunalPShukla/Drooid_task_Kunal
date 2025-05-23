import os
import sys
import json
import pymongo
import certifi


MONGO_DB_URL = "mongodb+srv://kpshukla3:QAElI81frJLKq3eo@drooidinputfile.feshorw.mongodb.net/?retryWrites=true&w=majority&appName=DrooidInputFile"

class NewsDataUploader:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
        except Exception as e:
            raise (e, sys)

    def load_json(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return data
        except Exception as e:
            raise (e, sys)

    def insert_data_mongodb(self, records: list, database: str, collection: str):
        try:
            db = self.client[database]
            col = db[collection]
            col.insert_many(records)
            return len(records)
        except Exception as e:
            raise (e, sys)

if __name__ == '__main__':
    try:
        FILE_PATH = "newsarticle.json"
        DATABASE = "qa_data"
        COLLECTION = "news_articles"

        uploader = NewsDataUploader()
        records = uploader.load_json(FILE_PATH)
        inserted = uploader.insert_data_mongodb(records, DATABASE, COLLECTION)
        print(f"Inserted {inserted} articles into MongoDB.")
    except Exception as e:
        print(e)
