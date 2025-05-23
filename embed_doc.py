# embed_full_articles.py
from fetch_documents import MongoArticleFetcher
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from tqdm import tqdm
import faiss
import numpy as np
import pickle

# Step 1: Load full articles from MongoDB
fetcher = MongoArticleFetcher("qa_data", "news_articles")
raw_documents = fetcher.fetch_documents()  # This returns List[Document]
texts = [doc.page_content for doc in raw_documents]  # Each text is a full article

print(f"Fetched {len(texts)} full articles")

# Step 2: Load sentence transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Encode full articles (batching for performance)
batch_size = 64
embeddings = []

for i in tqdm(range(0, len(texts), batch_size), desc="Embedding full articles"):
    batch = texts[i:i + batch_size]
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    embeddings.extend(batch_embeddings)

# Step 4: Create FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Step 5: Save FAISS index and articles
faiss.write_index(index, "faiss_full_articles.index")

# Save original articles for retrieval
with open("full_articles.pkl", "wb") as f:
    pickle.dump(raw_documents, f)

print("FAISS index and article list saved.")
