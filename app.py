import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv
import os
import time

# 1. Load API Keys
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- CORE FUNCTIONS ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Using the standard embedding model (Safe for 2025)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    vector_store = None
    batch_size = 5 
    
    total_chunks = len(text_chunks)
    progress_text = "Embedding in progress. Please wait..."
    my_bar = st.progress(0, text=progress_text)

    for i in range(0, total_chunks, batch_size):
        batch = text_chunks[i : i + batch_size]
        success = False
        while not success:
            try:
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embedding=embeddings)
                else:
                    new_vectors = FAISS.from_texts(batch, embedding=embeddings)
                    vector_store.merge_from(new_vectors)
                success = True 
            except Exception as e:
                if "429" in str(e):
                    st.warning(f"Rate limit hit. Waiting 30s... (Batch {i})")
                    time.sleep(30)
                else:
                    st.error(f"Error: {e}")
                    return

        percent_complete = min((i + batch_size) / total_chunks, 1.0)
        my_bar.progress(percent_complete, text=f"Processing batch {i//batch_size + 1}...")
        time.sleep(1)

    if vector_store:
        vector_store.save_local("faiss_index")
    my_bar.empty()

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say "The answer is not available in the context", don't provide the wrong answer.
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    
    # 🛠️ FIXED: Using the model confirmed in your list
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Retry loop for searches
        docs = None
        retries = 3
        for attempt in range(retries):
            try:
                docs = new_db.similarity_search(user_question)
                break 
            except Exception as e:
                if "429" in str(e):
                    st.warning(f"Traffic high. Retrying in 5 seconds... (Attempt {attempt+1})")
                    time.sleep(5)
                else:
                    raise e 

        if docs:
            chain = get_conversational_chain()
            response = chain.invoke(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            st.write("Reply: ", response["output_text"])
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# --- FRONTEND (UI) ---

def main():
    st.set_page_config("Chat PDF")
    st.header("🤖 Chat with PDF (Gemini Powered)")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                if text_chunks:
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.warning("No text found in PDF.")

if __name__ == "__main__":
    main()