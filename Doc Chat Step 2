import os
import uuid
from tika import parser
import spacy
from collections import defaultdict
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import time
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter # Added for splitting

# --- Configuration ---
# Use SentenceTransformer for embeddings
MODEL_NAME = "all-MiniLM-L6-v2"
# ChromaDB setup
CHROMA_PATH = "chroma_db" # Directory to store ChromaDB data
COLLECTION_NAME_PREFIX = "doc_chat_"

def get_device():
    """Gets the appropriate device for computation (CUDA or CPU)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
st.info(f"Using device: {device}")

@st.cache_resource
def get_embedding_model():
    """Loads the SentenceTransformer model (cached)."""
    print(f"Loading embedding model: {MODEL_NAME} onto device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("Embedding model loaded.")
    return model

@st.cache_resource
def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client."""
    print(f"Initializing ChromaDB client at path: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    print("ChromaDB client initialized.")
    return client

# --- Core Functions ---

def extract_text_from_file(uploaded_file):
    """Extracts text content from an uploaded file using Tika."""
    try:
        print(f"Extracting text from: {uploaded_file.name}")
        parsed = parser.from_buffer(uploaded_file.read())
        content = parsed.get('content')
        if not content:
            st.error("Could not extract text content from the file.")
            return None
        print(f"Successfully extracted text (length: {len(content)}).")
        # Limit content size for performance if needed
        # MAX_LEN = 5_000_000
        # if len(content) > MAX_LEN:
        #     st.warning(f"Content is very long ({len(content)} chars). Truncating to {MAX_LEN} chars.")
        #     content = content[:MAX_LEN]
        return content.strip()
    except Exception as e:
        st.error(f"Error parsing file with Tika: {str(e)}")
        return None

def split_text(text):
    """Splits text into manageable chunks."""
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    print(f"Split text into {len(chunks)} chunks.")
    return chunks

def setup_vector_store(chunks, collection_name):
    """Creates embeddings and stores chunks in ChromaDB."""
    start_time = time.time()
    print(f"Setting up vector store for collection: {collection_name}")
    chroma_client = get_chroma_client()
    embedding_model = get_embedding_model()

    # Use SentenceTransformer directly with ChromaDB
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME, device=device
    )

    # Check if collection exists, delete if it does (for simplicity in workshop)
    try:
        if collection_name in [c.name for c in chroma_client.list_collections()]:
            print(f"Deleting existing collection: {collection_name}")
            chroma_client.delete_collection(name=collection_name)
    except Exception as e:
        st.warning(f"Could not delete existing collection {collection_name}: {e}")
        # Continue anyway, maybe creation will fail if it truly exists and couldn't be deleted

    # Create collection
    try:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef,
            metadata={"hnsw:space": "cosine"} # Use cosine distance
        )
        print(f"Created new collection: {collection_name}")
    except Exception as e:
         # Handle potential race condition or other creation error
        try:
            print(f"Collection creation failed, attempting to get existing collection: {collection_name}. Error: {e}")
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef
            )
            print(f"Successfully got existing collection: {collection_name}")
        except Exception as get_e:
             st.error(f"Failed to create or get collection {collection_name}: {get_e}")
             return None


    # Add documents in batches (optional, but good practice for large docs)
    batch_size = 100
    num_chunks = len(chunks)
    print(f"Adding {num_chunks} chunks to collection '{collection_name}'...")
    with st.spinner(f"Embedding and storing {num_chunks} document chunks..."):
        for i in range(0, num_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))] # Unique IDs

            try:
                 # Embeddings are handled by ChromaDB when using embedding_function
                collection.add(
                    documents=batch_chunks,
                    ids=ids
                )
                progress = min((i + batch_size) / num_chunks, 1.0)
                # st.progress(progress, text=f"Processing batch {i//batch_size + 1}...") # Optional progress bar
            except Exception as e:
                st.error(f"Error adding batch {i//batch_size + 1} to ChromaDB: {e}")
                # Decide whether to stop or continue
                return None # Stop on error

    end_time = time.time()
    print(f"Finished adding chunks to vector store. Time taken: {end_time - start_time:.2f} seconds")
    st.success(f"Document processed and stored in vector database (Collection: {collection_name}). Time: {end_time - start_time:.2f}s")
    return collection_name # Return name instead of object for state management

def query_vector_store(question, collection_name, n_results=3):
    """Queries the vector store for relevant chunks based on the question."""
    if not collection_name:
        st.error("Vector store not initialized.")
        return []

    start_time = time.time()
    print(f"Querying collection '{collection_name}' for question: '{question}'")
    chroma_client = get_chroma_client()
    embedding_model = get_embedding_model() # Needed for the embedding function reference

    # Use SentenceTransformer directly with ChromaDB
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME, device=device
    )

    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef # Provide EF for consistency
        )
    except Exception as e:
        st.error(f"Could not retrieve collection '{collection_name}': {e}")
        # Attempt to list collections to see if it exists
        try:
            st.info(f"Available collections: {[c.name for c in chroma_client.list_collections()]}")
        except Exception as list_e:
            st.warning(f"Could not list collections: {list_e}")
        return []


    try:
        results = collection.query(
            query_texts=[question],
            n_results=n_results,
            include=['documents', 'distances'] # Include distances to show relevance
        )
        end_time = time.time()
        print(f"Query completed in {end_time - start_time:.2f} seconds.")
        print(f"Retrieved {len(results.get('documents', [[]])[0])} chunks.")
        # print(f"Results: {results}") # Debugging
        return results # Return the full results dictionary
    except Exception as e:
        st.error(f"Error querying ChromaDB collection '{collection_name}': {e}")
        return []


# --- Streamlit UI ---
st.title("Chat with your Document")
st.write("Upload a document (PDF, DOCX, TXT) and ask questions about its content.")

# File Uploader
uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'doc', 'docx'])

# Initialize session state
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

# Process uploaded file
if uploaded_file is not None:
    # Check if it's a new file or the same one
    if st.session_state.uploaded_filename != uploaded_file.name:
        st.session_state.processing_done = False # Reset processing flag for new file
        st.session_state.collection_name = None
        st.session_state.uploaded_filename = uploaded_file.name

        with st.status("Processing document...", expanded=True) as status:
            st.write("Extracting text...")
            text_content = extract_text_from_file(uploaded_file)

            if text_content:
                st.write("Splitting text into chunks...")
                chunks = split_text(text_content)

                if chunks:
                    # Create a unique collection name based on filename (and maybe timestamp/uuid)
                    # Simple approach: replace non-alphanumeric with underscore
                    safe_filename = "".join(c if c.isalnum() else "_" for c in uploaded_file.name)
                    collection_name = f"{COLLECTION_NAME_PREFIX}{safe_filename}_{uuid.uuid4().hex[:8]}"
                    st.write(f"Creating vector store (Collection: {collection_name})...")
                    created_collection_name = setup_vector_store(chunks, collection_name)

                    if created_collection_name:
                        st.session_state.collection_name = created_collection_name
                        st.session_state.processing_done = True
                        status.update(label="Document processing complete!", state="complete", expanded=False)
                    else:
                        st.error("Failed to set up vector store.")
                        status.update(label="Processing failed", state="error")
                else:
                    st.error("Failed to split text into chunks.")
                    status.update(label="Processing failed", state="error")
            else:
                st.error("Failed to extract text from the file.")
                status.update(label="Processing failed", state="error")
    else:
        # Same file uploaded again, check if processing was already done
        if st.session_state.processing_done:
            st.success(f"Document '{uploaded_file.name}' is already processed and ready for questions (Collection: {st.session_state.collection_name}).")
        else:
            # This case shouldn't happen often with the logic above, but handle it just in case
            st.warning("Processing was interrupted or failed previously. Please re-upload or refresh.")


# Chat Interface (only if processing is done)
if st.session_state.processing_done and st.session_state.collection_name:
    st.divider()
    st.header("Ask a Question")

    user_question = st.text_input("Enter your question about the document:")

    if user_question:
        with st.spinner("Searching for relevant information..."):
            results = query_vector_store(user_question, st.session_state.collection_name, n_results=3)

            if results and results.get('documents') and results['documents'][0]:
                st.subheader("Relevant Information Found:")
                for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
                    st.markdown(f"**Chunk {i+1} (Distance: {dist:.4f}):**")
                    st.text_area(f"chunk_{i+1}", doc, height=150, key=f"chunk_display_{i}_{user_question}") # Use unique key
                    st.divider()
                st.success("Found relevant chunks. Review the text above for your answer.")
            else:
                st.warning("Could not find relevant information in the document for your question.")

elif uploaded_file is None:
    st.info("Please upload a document to begin.")

