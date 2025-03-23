import streamlit as st
import os
import time
import tempfile
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.documents import Document

# Page configuration
st.set_page_config(
    page_title="PDF Uploader for Ms. GPT (Diagnostic)",
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

.diagnostic-info {
    margin-top: 1rem;
    padding: 1rem;
    background-color: #f0f0f0;
    border-radius: 10px;
    border-left: 4px solid #555555;
    font-family: monospace;
    white-space: pre-wrap;
    overflow-x: auto;
}

pre {
    white-space: pre-wrap;
    background-color: #f5f5f5;
    padding: 10px;
    border-radius: 5px;
    overflow-x: auto;
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
st.title("PDF Uploader for Ms. GPT (Diagnostic)")
st.markdown("Upload PDFs to expand Catherine's knowledge base")

# Diagnostic mode toggle
with st.expander("Diagnostic Options", expanded=True):
    st.markdown("These options help identify why your PDFs aren't showing up in the Resource Library.")
    
    format_option = st.radio(
        "Source format in metadata:",
        [
            "cleaned_pdfs\\filename.pdf", 
            "cleaned_pdfs/filename.pdf", 
            "filename.pdf (no prefix)",
            "All formats (try all)"
        ],
        index=3  # Default to trying all formats
    )
    
    add_extra_doc = st.checkbox("Add test document with unique source name", value=True)
    
    test_query = st.checkbox("Test query after upload", value=True)
    
    show_pinecone_stats = st.checkbox("Show Pinecone index stats", value=True)

# Sidebar with instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Choose your OpenAI API key option
    2. Set diagnostic options
    3. Upload one or more PDF files
    4. Click the 'Process PDFs' button
    5. Review the diagnostic information
    
    The diagnostic information will help identify why your PDFs aren't appearing in the resource library.
    """)
    
    st.header("About")
    st.markdown("""
    This diagnostic tool helps troubleshoot issues with the PDF upload process.
    
    It provides detailed information about the Pinecone index and how documents are stored and retrieved.
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

# Get format strings based on selection
def get_format_strings(option, filename):
    if option == "cleaned_pdfs\\filename.pdf":
        return [f"cleaned_pdfs\\{filename}"]
    elif option == "cleaned_pdfs/filename.pdf":
        return [f"cleaned_pdfs/{filename}"]
    elif option == "filename.pdf (no prefix)":
        return [filename]
    else:  # All formats
        return [
            f"cleaned_pdfs\\{filename}",
            f"cleaned_pdfs/{filename}",
            filename
        ]

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
        
        # Display Pinecone index stats if requested
        if show_pinecone_stats:
            try:
                stats = index.describe_index_stats()
                st.markdown("### Pinecone Index Stats (Before Upload)")
                st.markdown('<div class="diagnostic-info">', unsafe_allow_html=True)
                st.json(stats)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as stats_e:
                st.warning(f"Could not retrieve Pinecone stats: {str(stats_e)}")
        
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
        all_source_formats = []
        
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
                        "chunks": 0,
                        "sources": []
                    })
                    continue
                
                pages = len(documents)
                total_pages += pages
                
                # Split into chunks
                chunked_documents = text_splitter.split_documents(documents)
                chunks = len(chunked_documents)
                total_chunks += chunks
                
                # Get source formats to try
                source_formats = get_format_strings(format_option, uploaded_file.name)
                all_source_formats.extend(source_formats)
                
                # Add to Pinecone with different source formats
                status_text.text(f"Adding {chunks} chunks from {uploaded_file.name} to Pinecone...")
                
                # Process in smaller batches
                batch_size = 50
                batch_results = []
                
                for source_format in source_formats:
                    for j in range(0, chunks, batch_size):
                        end_idx = min(j + batch_size, chunks)
                        batch = chunked_documents[j:end_idx].copy()
                        
                        # Set source metadata in format being tested
                        for doc in batch:
                            doc.metadata["source"] = source_format
                            doc.metadata["type"] = "pdf_resource"
                            doc.metadata["upload_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                            doc.metadata["diagnostic"] = "true"
                        
                        # Add documents to Pinecone
                        vector_store.add_documents(batch)
                        batch_results.append({
                            "format": source_format,
                            "documents": len(batch)
                        })
                        
                        # Update progress
                        file_progress = 0.5 + ((i + (end_idx / chunks)) / total_files) * 0.5
                        progress_bar.progress(min(file_progress, 1.0))
                        time.sleep(0.5)  # Small delay to avoid rate limits
                
                file_results.append({
                    "name": uploaded_file.name,
                    "status": "Success",
                    "pages": pages,
                    "chunks": chunks,
                    "sources": source_formats,
                    "batches": batch_results
                })
                
            except Exception as e:
                file_results.append({
                    "name": uploaded_file.name,
                    "status": f"Error: {str(e)}",
                    "pages": 0,
                    "chunks": 0,
                    "sources": []
                })
            
            finally:
                # Clean up the temporary file
                os.unlink(pdf_path)
        
        # Add test document if requested
        if add_extra_doc:
            test_timestamp = time.strftime("%Y%m%d%H%M%S")
            test_doc_source = f"TEST_DOCUMENT_{test_timestamp}.pdf"
            
            test_document = Document(
                page_content=f"This is a test document created at {test_timestamp} to verify the Pinecone index is working correctly and documents can be retrieved by the resource library.",
                metadata={
                    "source": test_doc_source,
                    "type": "test_document",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "diagnostic": "true"
                }
            )
            
            status_text.text("Adding test document to Pinecone...")
            vector_store.add_documents([test_document])
            
            all_source_formats.append(test_doc_source)
            
            st.markdown("### Test Document")
            st.markdown(f"Added test document with source: `{test_doc_source}`")
            st.markdown("Check if this appears in your resource library after refreshing.")
        
        # Verify documents were added if requested
        if test_query:
            status_text.text("Testing document retrieval...")
            
            st.markdown("### Document Retrieval Test")
            
            # Test direct fetch first
            try:
                st.markdown("#### 1. Testing direct Pinecone fetch")
                direct_results = []
                
                for source in all_source_formats:
                    try:
                        # Try to fetch by metadata filter
                        results = vector_store.similarity_search(
                            "test",
                            k=2,
                            filter={"source": source}
                        )
                        
                        if results:
                            st.success(f"✅ Successfully retrieved documents with source: `{source}`")
                            direct_results.append(source)
                        else:
                            st.warning(f"❌ No documents found with source: `{source}`")
                    except Exception as search_e:
                        st.error(f"Error querying for source `{source}`: {str(search_e)}")
                
                # Test more general type filter
                try:
                    results = vector_store.similarity_search(
                        "test",
                        k=5,
                        filter={"type": "pdf_resource"}
                    )
                    
                    if results:
                        st.success(f"✅ Successfully retrieved {len(results)} documents with type: 'pdf_resource'")
                    else:
                        st.warning("❌ No documents found with type: 'pdf_resource'")
                        
                except Exception as type_e:
                    st.error(f"Error querying by type filter: {str(type_e)}")
                
                # Test fallback queries (similar to main app)
                st.markdown("#### 2. Testing fallback query methods")
                
                # Test queries used by main app
                main_app_queries = [
                    "education pdf documents", 
                    "teaching resources",
                    "literacy curriculum"
                ]
                
                query_results = []
                for query in main_app_queries:
                    try:
                        results = vector_store.similarity_search(
                            query, 
                            k=5
                        )
                        
                        found_sources = set([doc.metadata.get("source", "") for doc in results])
                        query_results.append({
                            "query": query,
                            "results": len(results),
                            "sources": list(found_sources)
                        })
                        
                        if any(source in found_sources for source in all_source_formats):
                            st.success(f"✅ Query '{query}' found uploaded documents")
                        else:
                            st.warning(f"❌ Query '{query}' did not find uploaded documents")
                            
                    except Exception as query_e:
                        st.error(f"Error with query '{query}': {str(query_e)}")
                
                st.markdown("#### 3. Recommended actions")
                if not direct_results:
                    st.error("None of the tested source formats were found. This suggests problems with the index or embeddings.")
                    st.markdown("""
                    **Possible solutions:**
                    - Check if your Pinecone index has namespaces enabled and if so, make sure both apps use the same namespace
                    - Verify that your OpenAI API key is working correctly in both apps
                    - Try modifying the main app's `get_pdf_library()` function to use a broader search
                    """)
                else:
                    st.success(f"Documents with these source formats were found: {', '.join(direct_results)}")
                    st.markdown(f"""
                    **Recommendations:**
                    1. Modify the main app's `get_pdf_library()` function to look for source: '{direct_results[0]}'
                    2. Check if the test document '{test_doc_source if add_extra_doc else '[not added]'}' appears in the resource library
                    3. If still not working, try restarting the main app to clear its cache
                    """)
                
            except Exception as verification_e:
                st.error(f"Error during verification: {str(verification_e)}")
        
        # Display updated Pinecone stats if requested
        if show_pinecone_stats:
            try:
                stats = index.describe_index_stats()
                st.markdown("### Pinecone Index Stats (After Upload)")
                st.markdown('<div class="diagnostic-info">', unsafe_allow_html=True)
                st.json(stats)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as stats_e:
                st.warning(f"Could not retrieve updated Pinecone stats: {str(stats_e)}")
        
        # Completed
        progress_bar.progress(1.0)
        status_text.text("Processing and diagnostics complete!")
        
        # Show results
        st.markdown('<div class="result-summary">', unsafe_allow_html=True)
        st.subheader("Processing Results")
        st.markdown(f"**Files processed:** {total_files}")
        st.markdown(f"**Total pages:** {total_pages}")
        st.markdown(f"**Total chunks added:** {total_chunks}")
        st.markdown(f"**Source formats tested:** {len(all_source_formats)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show details for each file
        st.subheader("File Details")
        for result in file_results:
            with st.expander(f"{result['name']} - {result['status']}"):
                st.markdown(f"**Pages:** {result['pages']}")
                st.markdown(f"**Chunks:** {result['chunks']}")
                if "sources" in result and result["sources"]:
                    st.markdown("**Source formats tested:**")
                    for src in result["sources"]:
                        st.markdown(f"- `{src}`")
        
        # Show main app code suggestion
        st.subheader("Fix for Main App")
        st.markdown("Based on diagnostics, here's a suggested modification for the `get_pdf_library()` function in your main app:")
        
        suggested_code = """
def get_pdf_library():
    try:
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        index = connect_to_pinecone()
        if not index:
            return []
        
        # Try direct metadata fetch (enhanced to check multiple formats)
        try:
            stats = index.describe_index_stats()
            total_vector_count = stats.get('total_vector_count', 0)
            
            # Get all documents in batches
            all_ids = []
            batch_size = 1000
            
            for i in range(0, total_vector_count, batch_size):
                id_batch = index.fetch(ids=None, limit=batch_size, offset=i)
                if id_batch and 'vectors' in id_batch:
                    all_ids.extend(list(id_batch['vectors'].keys()))
            
            all_sources = set()
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i+batch_size]
                batch_data = index.fetch(ids=batch_ids)
                
                if batch_data and 'vectors' in batch_data:
                    for vector_id, vector_data in batch_data['vectors'].items():
                        if 'metadata' in vector_data:
                            source = vector_data['metadata'].get('source', '')
                            if source and source != '':
                                # Remove any "conversation_" prefixes
                                if not source.startswith('conversation_'):
                                    all_sources.add(source)
            
            if all_sources:
                return sorted(list(all_sources))
                
        except Exception as e:
            print(f"⚠️ Direct metadata fetch failed: {e}")
        
        # Fallback: Use broader similarity search with multiple queries
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        all_sources = set()
        queries = [
            "education pdf documents", 
            "teaching resources",
            "literacy curriculum",
            "math education resources",
            "assessment strategies",
            "pdf document",  # Simple generic query
            "education",     # Very generic query
            "test document"  # For test documents
        ]
        
        # Try with type filter first
        try:
            for query in queries[:3]:  # Try just a few queries with filter
                results = vector_store.similarity_search(
                    query, 
                    k=100,
                    filter={"type": {"$ne": "conversation_memory"}}
                )
                
                for doc in results:
                    source = doc.metadata.get("source", "")
                    if source and not source.startswith('conversation_'):
                        all_sources.add(source)
                        
            if all_sources:
                return sorted(list(all_sources))
        except Exception:
            pass  # Silently continue to unfiltered search
            
        # Fall back to unfiltered search
        for query in queries:
            try:
                results = vector_store.similarity_search(query, k=100)
                
                for doc in results:
                    source = doc.metadata.get("source", "")
                    if source and not source.startswith('conversation_'):
                        all_sources.add(source)
            except Exception as query_e:
                print(f"Query '{query}' failed: {query_e}")
        
        return sorted(list(all_sources))
    except Exception as e:
        print(f"❌ Error retrieving PDF library: {e}")
        return []
"""

        st.code(suggested_code, language="python")
        
        st.success("Diagnostic complete! Use the information above to troubleshoot why your PDFs aren't appearing in the resource library.")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
