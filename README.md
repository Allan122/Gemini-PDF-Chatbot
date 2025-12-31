# 🤖 Gemini RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Gemini API](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

A powerful **RAG (Retrieval-Augmented Generation)** application that allows users to chat with their PDF documents using Google's latest **Gemini 2.5 Flash** model.

## 📸 Demo
*(Upload a screenshot of the app answering a question here)*

## 🚀 Key Features
* **🧠 Advanced RAG Architecture:** Uses FAISS vector store to retrieve exact context from documents.
* **🛡️ Smart Rate Limiting:** Automatic retry logic handles API traffic and prevents crashes.
* **⚡ High Performance:** Powered by the `gemini-2.5-flash` model for sub-second responses.
* **📄 Multi-File Support:** capable of processing multiple complex PDFs simultaneously.

## 🛠️ Technical Architecture
* **Language:** Python 3.13
* **Framework:** Streamlit (UI), LangChain (Orchestration)
* **LLM:** Google Gemini 2.5 Flash
* **Embeddings:** Google Text-Embedding-004
* **Vector DB:** FAISS (Facebook AI Similarity Search)

## 📦 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Allan122/Gemini-PDF-Chatbot.git](https://github.com/Allan122/Gemini-PDF-Chatbot.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up API Keys:**
    * Create a `.env` file in the root directory.
    * Add your key: `GOOGLE_API_KEY=your_api_key_here`
4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

---
*Allan Cheerakunnil Alex - Data Analyst Portfolio*
