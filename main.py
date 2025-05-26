from pymongo import MongoClient
import os
from pathlib import Path
from get_image_embedding import get_image_embedding

client = MongoClient(os.getenv("DATABASE_URL"))
db = client["vectorDb"]
collection = db["imageVectors"]

image_paths=[]
folder = Path(os.getenv("DATASET_PATH"))
for file_path in folder.rglob('*'):
    if file_path.is_file():
            image_paths.append(file_path)

for path in image_paths:
    emb = get_image_embedding(path)
    collection.insert_one({
    "name": str(path),
    "embedding":emb.tolist()
    })

def vector_search(query_embedding):
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 1, 
                "k": 3  
            }
        }
    ])
    for doc in results:
        print(doc["text"])