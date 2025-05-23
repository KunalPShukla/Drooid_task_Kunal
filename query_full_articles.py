# query_full_articles.py

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Step 1: Load FAISS index and full articles
index = faiss.read_index("faiss_full_articles.index")
with open("full_articles.pkl", "rb") as f:
    documents = pickle.load(f)

# Step 2: Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Embed the query
query = input("Enter your question: ")
query_embedding = embedder.encode([query]).astype("float32")

# Step 4: Retrieve top-k full articles
k = 3
D, I = index.search(query_embedding, k)
retrieved_docs = [documents[i].page_content for i in I[0]]
titles = [documents[i].metadata.get("title", "Unknown") for i in I[0]]

# Step 5: Optional - limit to top-1 document or full context
use_top1_only = False  # Toggle this if needed
if use_top1_only:
    combined_context = retrieved_docs[0]
else:
    combined_context = "\n\n---\n\n".join(retrieved_docs)

# Step 6: Build better prompt
template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a highly knowledgeable investigative assistant. Answer the question with clarity and completeness using only the context below.

Context:
{context}

Question:
{question}

Answer:"""
)

prompt = template.format(context=combined_context, question=query)

# Step 7: Load model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Step 8: Generate answer with beam search
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=4,
        early_stopping=True
    )
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 9: Output result
print("\nTop Source Titles:")
for t in titles:
    print("-", t)

print("\nAnswer:")
print(response)
