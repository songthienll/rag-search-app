# 📚 RAG Search App

This is a **Retrieval-Augmented Generation (RAG) Q&A application** built with **Streamlit** and **Ollama**. It allows users to:

✅ Upload a **PDF document** or enter a **URL**  
✅ Extract and chunk text into manageable pieces  
✅ Generate embeddings for each chunk  
✅ Search for relevant text chunks based on user queries using **cosine similarity**  
✅ Use **Large Language Models (LLMs)** from Ollama to generate answers grounded on retrieved context

---

## 🚀 Features

- Upload **PDF files** or extract text from **web URLs**
- Generate embeddings using **nomic-embed-text** model
- Search and retrieve top relevant chunks
- Query LLM (e.g. **llama3.2**) to produce final answers
- Streamlit UI with upload, search tabs, and progress indicators

---

## 🛠️ Requirements

- **Ollama** installed with required models:
  - `nomic-embed-text`
  - `llama3.2` (or your preferred LLM model)
- **Python packages**:
  - `streamlit`
  - `PyPDF2`
  - `requests`
  - `beautifulsoup4`
  - `numpy`
  - `scikit-learn`
  - `ollama`

---

## 📥 Installation

1. **Clone this repository**

```bash
git clone https://github.com/songthienll/rag-search-app.git
cd rag-search-app
```

2. **Install Python packages**
```bash
pip install requirements.txt
```
3. **Install Ollama and pull required models**
Visit https://ollama.ai for Ollama installation instructions.
```bash
ollama pull nomic-embed-text
ollama pull llama3.2  # or any other model you want to use
``` 
4. **Run the Streamlit app**
```bash
streamlit run app.py
```

---

## 💡Usage
1. **Open the app in your browser after running Streamlit**

2. **Upload a PDF or enter a URL in the Upload tab**

3. **Click "Process" to extract and embed text**

4. **Switch to Search tab and ask any question related to the document**

5. **The app will retrieve relevant chunks and generate a grounded answer using the selected LLM**
