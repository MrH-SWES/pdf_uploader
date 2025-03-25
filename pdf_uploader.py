import streamlit as st
import os
import time
import tempfile
import re
import traceback # Added for potentially more detailed error logging
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# --- Configuration ---

# Page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="PDF Uploader for Ms. GPT",
    page_icon="📚",
    layout="centered"
)

# Attempt to load keys from Streamlit secrets
# IMPORTANT: Ensure you have a .streamlit/secrets.toml file in your GitHub repo
# with your actual keys, like:
# PINECONE_API_KEY = "your_pinecone_key_here"
# OPENAI_API_KEY = "your_openai_key_here"
try:
    DEFAULT_OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except KeyError as e:
    st.error(f"Error: Missing secret key: {e}. Please ensure PINECONE_API_KEY and OPENAI_API_KEY are set in Streamlit secrets.", icon="🚨")
    st.stop() # Stop execution if secrets are missing

INDEX_NAME = "science-of-reading" # Your Pinecone index name

# --- UI Setup ---

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
    1. Choose your OpenAI API key option (default uses the configured secret key).
    2. Upload one or more PDF files.
    3. Click the 'Process PDFs' button.
    4. Wait for processing to complete.

    The PDFs will be processed and added to Catherine's knowledge base. She'll be able to access this information immediately.
    """)

    st.header("About")
    st.markdown("""
    This tool helps you expand Catherine's knowledge by processing PDF documents and adding them to her Pinecone vector database.

    The documents are split into chunks, embedded using OpenAI, and stored in Pinecone for semantic retrieval.

    **NEW**: Page numbers are now preserved in the metadata, allowing queries for specific pages.
    """)

# --- API Key Selection ---

# Initialize api_key variable
api_key = None

api_key_option = st.radio(
    "OpenAI API Key Source",
    ["Use stored default key", "Enter my own key"]
)

if api_key_option == "Enter my own key":
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API Key to continue processing.")
    elif not api_key.startswith("sk-"):
         st.warning("Please enter a valid OpenAI API Key starting with 'sk-'.", icon="⚠️")
else:
    # Use the key loaded from secrets
    api_key = DEFAULT_OPENAI_API_KEY
    if not api_key:
        # This case should ideally be caught by the initial secret check, but belt-and-suspenders
        st.error("Default OpenAI API Key not found in secrets.", icon="🚨")
    else:
        st.success("Using stored default OpenAI API key.", icon="✅")


# --- File Uploader and Settings ---

st.markdown('<div class="upload-header">', unsafe_allow_html=True)
st.subheader("Upload PDFs")
st.markdown("Select one or more PDF files to process and add to the knowledge base.")
st.markdown('</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Add option to clear index
with st.expander("⚠️ Advanced Settings", expanded=False):
    st.warning("Warning: Clearing the index will remove ALL data stored in Pinecone.")
    clear_index = st.checkbox("I want to clear the index before uploading")
    if clear_index:
        st.info("All existing data will be deleted before uploading new files.")

# --- Processing Logic ---

# Process PDFs button - disable if no files are uploaded or if using own key and it's missing/invalid
process_button_disabled = not uploaded_files or (api_key_option == "Enter my own key" and (not api_key or not api_key.startswith("sk-")))

if st.button("Process PDFs", disabled=process_button_disabled):
    if not api_key:
        st.error("Cannot process: OpenAI API Key is missing or invalid.", icon="🚫")
    else:
        # Set up progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Set OpenAI API key environment variable for Langchain components
            os.environ["OPENAI_API_KEY"] = api_key
            # Initialize embeddings (will use the key from os.environ)
            embeddings = OpenAIEmbeddings()

            # Connect to Pinecone using the key from secrets
            status_text.text("Connecting to Pinecone...")
            # Use the PINECONE_API_KEY loaded from secrets at the start
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(INDEX_NAME)
            status_text.text("Connected to Pinecone index successfully!")
            time.sleep(1) # Give user time to see the message

            # Clear index if requested
            if clear_index:
                status_text.text(f"Clearing Pinecone index '{INDEX_NAME}'...")
                index.delete(delete_all=True)
                # Wait briefly for deletion to propagate, might need longer for large indexes
                time.sleep(5)
                status_text.text("Index cleared successfully!")
                time.sleep(1)

            # Initialize vector store
            vector_store = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text", # Ensure this matches how data was stored if not clearing
                namespace=""
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
                file_progress_start = (i / total_files) * 0.5 # Progress before upserting
                progress_bar.progress(file_progress_start)
                status_text.text(f"Processing file {i+1}/{total_files}: {uploaded_file.name}")

                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name

                try:
                    # Load PDF
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load() # Each item is a page

                    if not documents:
                        st.warning(f"File '{uploaded_file.name}' yielded no documents (is it empty or unreadable?). Skipping.", icon="⚠️")
                        file_results.append({
                            "name": uploaded_file.name,
                            "status": "Skipped - No text extracted",
                            "pages": 0, "chunks": 0
                        })
                        continue # Skip to the next file

                    pages = len(documents)
                    total_pages += pages

                    # Split into chunks
                    chunked_documents = text_splitter.split_documents(documents)

                    # Clean/Prepare metadata before upserting
                    for doc in chunked_documents:
                        doc.metadata["type"] = "pdf_resource" # Add custom type
                        # Ensure page number is 1-indexed integer
                        if "page" in doc.metadata:
                            try:
                                doc.metadata["page"] = int(doc.metadata["page"]) + 1
                            except ValueError:
                                doc.metadata["page"] = 1 # Default if conversion fails
                        else:
                            doc.metadata["page"] = 1 # Default if key missing

                        # Clean up source filename
                        if "source" in doc.metadata:
                            doc.metadata["source"] = os.path.basename(uploaded_file.name) # Use original filename
                        else:
                             doc.metadata["source"] = uploaded_file.name # Add original filename if missing


                    chunks = len(chunked_documents)
                    total_chunks += chunks

                    if chunks == 0:
                        st.warning(f"File '{uploaded_file.name}' yielded no text chunks after splitting. Skipping upsert.", icon="⚠️")
                        file_results.append({
                            "name": uploaded_file.name,
                            "status": "Skipped - No chunks generated",
                            "pages": pages, "chunks": 0
                        })
                        continue # Skip to the next file


                    # Add to Pinecone in batches
                    status_text.text(f"Adding {chunks} chunks from {uploaded_file.name} to Pinecone...")
                    batch_size = 50 # Adjust as needed based on performance/rate limits
                    for j in range(0, chunks, batch_size):
                        end_idx = min(j + batch_size, chunks)
                        batch = chunked_documents[j:end_idx]
                        vector_store.add_documents(batch)

                        # Update progress within file processing (after upserting)
                        batch_progress = ((end_idx / chunks) * 0.5) / total_files
                        progress_bar.progress(file_progress_start + batch_progress)
                        time.sleep(0.5) # Small delay between batches

                    file_results.append({
                        "name": uploaded_file.name, "status": "Success",
                        "pages": pages, "chunks": chunks
                    })

                except Exception as file_e:
                    st.error(f"Error processing file {uploaded_file.name}: {str(file_e)}", icon="🚫")
                    file_results.append({
                        "name": uploaded_file.name, "status": f"Error: {str(file_e)}",
                        "pages": 0, "chunks": 0
                    })
                    # Optionally log full traceback to Streamlit logs for detailed debugging
                    # print(f"Traceback for file {uploaded_file.name}:")
                    # print(traceback.format_exc())

                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(pdf_path)
                    except OSError as unlink_e:
                         st.warning(f"Could not delete temporary file {pdf_path}: {unlink_e}", icon="⚠️")


            # Completed all files
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")

            # --- Show Results ---
            st.markdown('<div class="result-summary">', unsafe_allow_html=True)
            st.subheader("Processing Results")
            st.markdown(f"**Files processed:** {total_files}")
            st.markdown(f"**Total pages extracted:** {total_pages}")
            st.markdown(f"**Total chunks added/updated in Pinecone:** {total_chunks}")
            st.markdown("**Page metadata:** Preserved (1-indexed)")
            st.markdown('</div>', unsafe_allow_html=True)

            # Show details for each file
            st.subheader("File Details")
            for result in file_results:
                expander_title = f"{result['name']} - {result['status']}"
                if "Error" in result['status']:
                    expander_title = f"🚫 {expander_title}"
                elif "Skipped" in result['status']:
                     expander_title = f"⚠️ {expander_title}"
                else:
                    expander_title = f"✅ {expander_title}"

                with st.expander(expander_title):
                    st.markdown(f"**Pages Extracted:** {result['pages']}")
                    st.markdown(f"**Chunks Added:** {result['chunks']}")
                    if "Error" in result['status']:
                         st.error(f"Details: {result['status']}")

            st.success("All files have been processed!")
            st.balloons()

        except Exception as e:
            st.error(f"A critical error occurred during processing: {str(e)}", icon="🚨")
            # Log full traceback to Streamlit logs for detailed debugging
            print(f"Critical Error Traceback:")
            print(traceback.format_exc())
            status_text.text("Processing failed.") # Update status

# --- Footer or additional info ---
st.markdown("---")
st.caption("Ensure your API keys are correctly configured in Streamlit secrets.")
