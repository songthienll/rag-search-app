import streamlit as st
import PyPDF2
import requests
from bs4 import BeautifulSoup
import numpy as np
import ollama
from io import BytesIO
import time
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Set page config
st.set_page_config(
    page_title="RAG Search App",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []
if 'processed' not in st.session_state:
    st.session_state.processed = False


def extract_pdf_text(uploaded_file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def extract_url_text(url):
    """Extract text from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        return f"Error: {str(e)}"


def chunk_text(text, max_length=500):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def generate_embedding(text):
    """Generate embedding using Ollama"""
    try:
        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        return np.array(response['embedding'])
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None


def get_available_models():
    """Get available Ollama models"""
    try:
        response = ollama.list()

        models = response.get('models', [])
        model_names = []
        for model in models:
            model_name = model.get('name') or model.get('model_name') or model.get('model')
            if model_name:
                if 'embed' not in model_name.lower() and 'nomic-embed-text' not in model_name:
                    model_names.append(model_name)

        if not model_names:
            model_names = ['llama3.2']  # Fallback if list empty

        return model_names
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return ['llama3.2']  # Default fallback



def search_similar_documents(query, top_k=3):
    """Search for similar documents using cosine similarity"""
    if not st.session_state.embeddings:
        return []

    query_embedding = generate_embedding(query)
    if query_embedding is None:
        return []

    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], st.session_state.embeddings)[0]

    # Get top k similar documents
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'content': st.session_state.documents[idx],
            'similarity': similarities[idx]
        })

    return results


def query_ollama(prompt, model='llama3.2'):
    """Query Ollama model"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error querying model: {str(e)}"


# Main UI
st.title("📚 RAG Search App")

# Sidebar for model selection
with st.sidebar:
    st.header("⚙️ Settings")

    # Get available models
    available_models = get_available_models()
    selected_model = st.selectbox(
        "Select LLM Model",
        available_models,
        index=0 if available_models else 0
    )

    st.markdown("---")
    st.markdown("### 📝 How to use:")
    st.markdown("1. Upload a PDF or enter a URL")
    st.markdown("2. Click 'Process' to extract and embed text")
    st.markdown("3. Ask questions about the document")

    if st.session_state.processed:
        st.success(f"✅ Processed {len(st.session_state.documents)} text chunks")

# Main content area
st.header("📤 Upload Document or URL")

# File upload
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=['pdf'],
    help="Upload a PDF file to extract text from"
)

st.markdown("**Or**")

# URL input
url_input = st.text_input(
    "Enter URL",
    placeholder="https://example.com/article.html",
    help="Enter a URL to extract text from a webpage"
)

# Process button
if st.button("🔄 Process", type="primary"):
    if uploaded_file is not None or url_input:
        with st.spinner("Processing document..."):
            start_time = time.time()

            try:
                # Extract text
                if uploaded_file is not None:
                    text = extract_pdf_text(uploaded_file)
                    source = f"PDF: {uploaded_file.name}"
                else:
                    text = extract_url_text(url_input)
                    source = f"URL: {url_input}"

                if text.startswith("Error:"):
                    st.error(text)
                else:
                    # Chunk text
                    chunks = chunk_text(text)

                    # Generate embeddings
                    embeddings = []
                    progress_bar = st.progress(0)

                    for i, chunk in enumerate(chunks):
                        embedding = generate_embedding(chunk)
                        if embedding is not None:
                            embeddings.append(embedding)
                        else:
                            st.error(f"Failed to generate embedding for chunk {i + 1}")
                            break

                        progress_bar.progress((i + 1) / len(chunks))

                    if len(embeddings) == len(chunks):
                        # Store in session state
                        st.session_state.documents = chunks
                        st.session_state.embeddings = embeddings
                        st.session_state.processed = True

                        time_taken = time.time() - start_time
                        st.success(f"✅ Document processed successfully!")
                        st.info(f"⏱️ Time taken: {time_taken:.2f} seconds")
                        st.info(f"📊 Generated {len(chunks)} text chunks")

                        # Show sample chunk
                        with st.expander("Preview first text chunk"):
                            st.write(chunks[0][:500] + "..." if len(chunks[0]) > 500 else chunks[0])

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    else:
        st.warning("Please upload a PDF file or enter a URL")

# Separator
st.markdown("---")

# Search section
st.header("🔍 Search Document")

if not st.session_state.processed:
    st.warning("⚠️ Please process a document first using the upload section above")
else:
    # Query input
    query = st.text_area(
        "Enter your question",
        placeholder="Ask a question",
        height=100
    )

    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching and generating answer..."):
                start_time = time.time()

                try:
                    # Search similar documents
                    results = search_similar_documents(query)

                    if results:
                        # Prepare context
                        context = "\n".join([result['content'] for result in results])

                        # Create prompt
                        prompt = f"""
You are an AI assistant with expertise in reading documents and answering questions based on them.
Context: {context}

Question: {query}

Answer the question comprehensively based on the context above. If the context doesn't contain enough information to answer the question, please say so."""

                        # Query model
                        answer = query_ollama(prompt, selected_model)

                        time_taken = time.time() - start_time

                        # Display results
                        st.subheader("🎯 Answer:")
                        st.write(answer)

                        st.subheader("📄 Top Context Chunks:")
                        for i, result in enumerate(results):
                            with st.expander(f"Relevant chunk {i + 1} (similarity: {result['similarity']:.3f})"):
                                st.write(result['content'])

                        st.info(f"⏱️ Time taken: {time_taken:.2f} seconds")
                    else:
                        st.error("No relevant documents found")

                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
        else:
            st.warning("Please enter a question")

