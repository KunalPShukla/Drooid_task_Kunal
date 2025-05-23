# streamlit_app.py

import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="News QA Assistant", layout="wide")
st.title("News Question Answering Assistant")

# Load FAISS index and documents
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

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embedder, tokenizer, model

index, documents = load_index_and_documents()
embedder, tokenizer, model = load_models()

query = st.text_input("Ask a question based on recent news articles (e.g. What did Sanjay Raut say?)")

if query:
    query_embedding = embedder.encode([query]).astype("float32")
    D, I = index.search(query_embedding, k=3)

    retrieved_docs = [documents[i].page_content for i in I[0]]
    titles = [documents[i].metadata.get("title", "Unknown") for i in I[0]]

    # Use all 3 documents for better context
    combined_context = "\n\n---\n\n".join(retrieved_docs)

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

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,
            early_stopping=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.markdown("### Top Source Titles:")
    for title in titles:
        st.write(f"- {title}")

    st.markdown("### Answer:")
    st.success(response)
