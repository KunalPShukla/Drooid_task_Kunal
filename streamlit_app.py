import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Page configuration
st.set_page_config(page_title="News QA Assistant", layout="wide")
st.title("🗞️ News Question Answering Assistant")

# Load FAISS index and articles
@st.cache_resource
def load_index_and_documents():
    try:
        index = faiss.read_index("faiss_full_articles.index")
        with open("full_articles.pkl", "rb") as f:
            documents = pickle.load(f)
        return index, documents
    except Exception as e:
        st.error(f"Error loading FAISS index or documents: {e}")
        st.stop()

# Load embedding model and FLAN-T5 model
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embedder, tokenizer, model

# Load resources
index, documents = load_index_and_documents()
embedder, tokenizer, model = load_models()

# Input query
query = st.text_input("Ask a question from the news articles:")
use_top1_only = st.checkbox("Use only top-1 most relevant article", value=False)

if query:
    # Embed query and search top-k documents
    query_embedding = embedder.encode([query]).astype("float32")
    D, I = index.search(query_embedding, k=3)

    # Retrieve articles and titles
    retrieved_docs = [documents[i].page_content for i in I[0]]
    titles = [documents[i].metadata.get("title", "Unknown") for i in I[0]]

    # Choose context: top 1 or full context
    if use_top1_only:
        combined_context = retrieved_docs[0]
    else:
        combined_context = "\n\n---\n\n".join(retrieved_docs)

    # Format prompt manually
    prompt = f"""You are a highly knowledgeable investigative assistant. Answer the question with clarity and completeness using only the context below.

Context:
{combined_context}

Question:
{query}

Answer:"""

    # Generate answer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,
            early_stopping=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display source titles
    st.markdown("### Top Source Titles:")
    for t in titles:
        st.markdown(f"- {t}")

    # Display answer
    st.markdown("### Answer:")
    st.success(response)
