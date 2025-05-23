import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Page configuration
st.set_page_config(page_title="News QA Assistant", layout="wide")
st.title("News Question Answering Assistant")

# Load precomputed FAISS index and documents (uploaded via Streamlit Cloud)
@st.cache_resource
def load_index_and_documents():
    try:
        index = faiss.read_index("faiss_full_articles.index")
    except Exception as e:
        st.error("Failed to load FAISS index. Make sure it is uploaded to Streamlit Cloud.")
        st.stop()
    
    try:
        with open("full_articles.pkl", "rb") as f:
            documents = pickle.load(f)
    except Exception as e:
        st.error("Failed to load article documents. Make sure full_articles.pkl is uploaded.")
        st.stop()

    return index, documents

# Load embedding model and language model (Flan-T5)
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embedder, tokenizer, model

# Load resources
index, documents = load_index_and_documents()
embedder, tokenizer, model = load_models()

# Input box
query = st.text_input("Ask a question based on recent news articles (e.g. What did Sanjay Raut say?)")

# Run pipeline
if query:
    # Embed query and search in FAISS
    query_embedding = embedder.encode([query]).astype("float32")
    D, I = index.search(query_embedding, k=3)
    retrieved_docs = [documents[i].page_content for i in I[0]]
    titles = [documents[i].metadata.get("title", "Unknown") for i in I[0]]

    # Construct context and prompt
    context = "\n\n---\n\n".join(retrieved_docs)
    prompt = f"""You are a helpful assistant. Answer the question using the context below.

Context:
{context}

Question:
{query}

Answer:"""

    # Generate answer using Flan-T5
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,
            early_stopping=True
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display results
    st.markdown("### Source Titles:")
    for t in titles:
        st.write("- " + t)

    st.markdown("### Answer:")
    st.success(answer)
