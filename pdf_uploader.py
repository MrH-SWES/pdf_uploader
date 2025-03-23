import streamlit as st
import os
import time
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.documents import Document

# Page configuration
st.set_page_config(
    page_title="PDF Uploader for Ms. GPT",
    page_icon=None,
    layout="centered"
)

# Custom CSS to match Ms. GPT style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&display=swap');

body, .main, .stApp {
    background-color: #fdf8f0 !important;
    font-family: 'Lexend', sans-serif;
    color: #2D2926;
}

.stApp h1 {
    font-family: 'Lexend', sans-serif !important;
    font-weight: 600;
    font-size: 28px;
    color: #022E66;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
    letter-spacing: -0.015em;
}

.stButton > button {
    background-color: #8B6E4E !important;
    color: #FFFFFF !important;
    font-family: 'Lexend', sans-serif;
    padding: 8px 16px !important;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    height: auto !important;
    min-height: 0px !important;
    white-space: nowrap;
    transition: background-color 0.3s ease;
}

.stButton > button:hover {
    background-color: #5D4037 !important;
    color: #FFFFFF !important;
}

.stProgress .st-bo {
    background-color: #8B6E4E;
}

.upload-header {
    margin-top: 2rem;
    margin-bottom: 1rem;
    padding: 1rem;
    background-color: #F5E8D6;
    border-radius: 10px;
    border-left: 4px solid #8B6E4E;
}

.result-summary {
    margin-top: 2rem;
    padding: 1rem;
    background-color: #e7f5e8;
    border-radius: 10px;
    border-left: 4px solid #8B6E4E;
}

/* File uploader styling */
[data-testid="stFileUploader"] {
    width: 100%;
}

[data-testid="stFileUploader"] section {
    padding: 1rem;
    border-radius: 10px;
    border: 1px dashed #8B6E4E;
    background-color: #F5E8D6;
}

/* Radio buttons */
.stRadio [data-testid="stMarkdownContainer"] p {
    font-family: 'Lexend', sans-serif;
    font-size: 15px;
    color: #2D2926;
}
</style>
""", unsafe_allow_html=True)

# App title
st.title("PDF Uploader for Ms. GPT")
st.markdown("Upload PDFs to expand Catherine's knowledge base")

# Sidebar with instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Choose your OpenAI API key option
    2. Upload one or more PDF files
    3. Click the 'Process PDFs' button
    4. Wait for processing to complete
    
    The PDFs will be processed and added to Catherine's knowledge base. She'll be able to access this information immediately.
    """)
    
    st.header("About")
    st.markdown("""
    This tool helps you expand Catherine's knowledge by processing PDF documents and adding them to her Pinecone vector database.
    
    The documents are split into chunks, embedded using OpenAI, and stored in Pinecone for semantic retrieval.
    """)

# API Key input
api_key_option = st.radio(
    "OpenAI API Key",
    ["Use default key", "Enter my own key"]
)

if api_key_option == "Enter my own key":
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API Key to continue")
else:
    api_key = st.secrets["OPENAI_API_KEY"]
    
# PDF uploader
st.markdown('<div class="upload-header">', unsafe_allow_html=True)
st.subheader("Upload PDFs")
st.markdown("Select one or more PDF files to process and add to the knowledge base.")
st.markdown('</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Pinecone settings
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
PINECONE_HOST = st.secrets.get("PINECONE_HOST", "")  # Get host if available

# Process PDFs button
if uploaded_files and st.button("Process PDFs"):
    # Set up progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize embeddings
    os.environ["OPENAI_API_KEY"] = api_key
    embeddings = OpenAIEmbeddings()
    
    # Connect to Pinecone
    status_text.text("Connecting to Pinecone...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        
        # Initialize vector store
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Process each uploaded file
        total_files = len(uploaded_files)
        total_pages = 0
        total_chunks = 0
        file_results = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress_bar.progress((i / total_files) * 0.5)
            status_text.text(f"Processing file {i+1}/{total_files}: {uploaded_file.name}")
            
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
            
            try:
                # Load PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                if not documents:
                    file_results.append({
                        "name": uploaded_file.name,
                        "status": "Skipped - No text extracted",
                        "pages": 0,
                        "chunks": 0
                    })
                    continue
                
                pages = len(documents)
                total_pages += pages
                
                # Split into chunks
                chunked_documents = text_splitter.split_documents(documents)
                chunks = len(chunked_documents)
                total_chunks += chunks
                
                # Add to Pinecone
                status_text.text(f"Adding {chunks} chunks from {uploaded_file.name} to Pinecone...")
                
                # Process in smaller batches
                batch_size = 50
                for j in range(0, chunks, batch_size):
                    end_idx = min(j + batch_size, chunks)
                    batch = chunked_documents[j:end_idx]
                    
                    # Set metadata using formats that work with your app
                    for doc in batch:
                        # Use the exact format that works with direct fetch
                        doc.metadata["source"] = f"cleaned_pdfs\\{uploaded_file.name}"
                        # Add additional metadata to help with filtering
                        doc.metadata["type"] = "pdf_resource"
                        doc.metadata["upload_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Add documents to Pinecone
                    vector_store.add_documents(batch)
                    
                    # Update progress within file
                    file_progress = 0.5 + ((i + (end_idx / chunks)) / total_files) * 0.5
                    progress_bar.progress(min(file_progress, 1.0))
                    time.sleep(0.5)  # Small delay to avoid rate limits
                
                file_results.append({
                    "name": uploaded_file.name,
                    "status": "Success",
                    "pages": pages,
                    "chunks": chunks
                })
                
            except Exception as e:
                file_results.append({
                    "name": uploaded_file.name,
                    "status": f"Error: {str(e)}",
                    "pages": 0,
                    "chunks": 0
                })
            
            finally:
                # Clean up the temporary file
                os.unlink(pdf_path)
        
        # Completed
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Show results
        st.markdown('<div class="result-summary">', unsafe_allow_html=True)
        st.subheader("Processing Results")
        st.markdown(f"**Files processed:** {total_files}")
        st.markdown(f"**Total pages:** {total_pages}")
        st.markdown(f"**Total chunks added:** {total_chunks}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show details for each file
        st.subheader("File Details")
        for result in file_results:
            with st.expander(f"{result['name']} - {result['status']}"):
                st.markdown(f"**Pages:** {result['pages']}")
                st.markdown(f"**Chunks:** {result['chunks']}")
                st.markdown(f"**Source in Pinecone:** cleaned_pdfs\\{result['name']}")
        
        st.success("All files have been processed and added to Catherine's knowledge base!")
        st.info("Please go back to Ms. GPT and click the 'Refresh' button in the Resource Library to see your newly added PDFs.")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
