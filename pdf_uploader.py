import streamlit as st
import os
import time
import tempfile
import re
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Page configuration
st.set_page_config(
    page_title="PDF Uploader for Ms. GPT",
    page_icon="📚",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f4f7f9;
    }
    .stApp h1 {
        color: #2d6a8f;
    }
    .stButton>button {
        background-color: #2d6a8f;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 500;
    }
    .stProgress .st-bo {
        background-color: #2d6a8f;
    }
    .upload-header {
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 1rem;
        background-color: #e6f0f7;
        border-radius: 10px;
        border-left: 4px solid #2d6a8f;
    }
    .result-summary {
        margin-top: 2rem;
        padding: 1rem;
        background-color: #e7f5e8;
        border-radius: 10px;
        border-left: 4px solid #2d9954;
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
    1. Upload one or more PDF files
    2. Click the 'Process PDFs' button
    3. Wait for processing to complete
    
    The PDFs will be processed and added to Catherine's knowledge base. She'll be able to access this information immediately.
    """)
    
    st.header("About")
    st.markdown("""
    This tool helps you expand Catherine's knowledge by processing PDF documents and adding them to her Pinecone vector database.
    
    The documents are split into chunks, embedded using OpenAI, and stored in Pinecone for semantic retrieval.
    
    **NEW**: Page numbers are now preserved in the metadata, allowing queries for specific pages.
    """)

# PDF uploader
st.markdown('<div class="upload-header">', unsafe_allow_html=True)
st.subheader("Upload PDFs")
st.markdown("Select one or more PDF files to process and add to the knowledge base.")
st.markdown('</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# API Keys from secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
    PINECONE_HOST = st.secrets.get("PINECONE_HOST", "")  # Optional, might not be needed depending on your Pinecone setup
except Exception as e:
    st.error(f"Error accessing secrets: {str(e)}")
    st.warning("Please make sure your API keys are set in Streamlit secrets.")
    OPENAI_API_KEY = ""
    PINECONE_API_KEY = ""
    PINECONE_INDEX_NAME = ""
    PINECONE_HOST = ""

# Add option to clear index
with st.expander("⚠️ Advanced Settings", expanded=False):
    st.warning("Warning: Clearing the index will remove ALL data stored in Pinecone.")
    clear_index = st.checkbox("I want to clear the index before uploading")
    if clear_index:
        st.info("All existing data will be deleted before uploading new files.")

# Process PDFs button
if uploaded_files and st.button("Process PDFs"):
    # Verify API keys are available
    if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        st.error("Missing API keys. Please check your Streamlit secrets configuration.")
        st.stop()
    
    # Set up progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize embeddings
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    embeddings = OpenAIEmbeddings()
    
    # Connect to Pinecone
    status_text.text("Connecting to Pinecone...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_params = {"name": PINECONE_INDEX_NAME}
        if PINECONE_HOST:
            index_params["host"] = PINECONE_HOST
        
        index = pc.Index(**index_params)
        
        # Clear index if requested
        if clear_index:
            status_text.text("Clearing Pinecone index...")
            index.delete(delete_all=True)
            status_text.text("Index cleared successfully!")
            time.sleep(1)
        
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
                # Load PDF - PyPDFLoader already includes page numbers in metadata
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
                
               # Ensure proper metadata is preserved
               for idx, doc in enumerate(chunked_documents):
                   # Set document type
                   doc.metadata["type"] = "pdf_resource"
    
                   # Page numbers in PyPDFLoader are 0-indexed, let's make them 1-indexed
                   if "page" in doc.metadata:
                       # Convert to int to ensure it's a number
                       page_num = int(doc.metadata["page"])
                       # Store as 1-indexed for more natural queries
                       doc.metadata["page"] = page_num + 1
                   else:
                       # Default to page 1 if missing
                       doc.metadata["page"] = 1
    
                   # IMPORTANT: Set the original filename as the source
                   # This overrides any temporary filename that might have been set
                   doc.metadata["source"] = uploaded_file.name
    
                   # Add debug info to the status message
                   status_text.text(f"Processing chunk {idx+1}/{len(chunked_documents)} from {uploaded_file.name}")
                
                chunks = len(chunked_documents)
                total_chunks += chunks
                
                # Add to Pinecone
                status_text.text(f"Adding {chunks} chunks from {uploaded_file.name} to Pinecone...")
                
                # Process in smaller batches
                batch_size = 50
                for j in range(0, chunks, batch_size):
                    end_idx = min(j + batch_size, chunks)
                    batch = chunked_documents[j:end_idx]
                    vector_store.add_documents(batch)
                    # Update progress within file
                    file_progress = 0.5 + ((i + (end_idx / chunks)) / total_files) * 0.5
                    progress_bar.progress(min(file_progress, 1.0))
                    time.sleep(1)  # Small delay to avoid rate limits
                
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
        st.markdown("**Page metadata:** Preserved for all documents")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show details for each file
        st.subheader("File Details")
        for result in file_results:
            with st.expander(f"{result['name']} - {result['status']}"):
                st.markdown(f"**Pages:** {result['pages']}")
                st.markdown(f"**Chunks:** {result['chunks']}")
        
        st.success("All files have been processed and added to Catherine's knowledge base!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
