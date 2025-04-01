import os
import uuid
from tika import parser
# import spacy # No longer needed for this approach
# from collections import defaultdict # No longer needed
import streamlit as st
# import pandas as pd # No longer needed
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import time
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Langchain Imports ---
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline # Added for LLM

# --- Configuration ---
# Use SentenceTransformer for embeddings
MODEL_NAME = "all-MiniLM-L6-v2"
# LLM Model for Response Generation
LLM_MODEL_ID = "google/flan-t5-base" # Or "google/flan-t5-small" for less memory
# ChromaDB setup
CHROMA_PATH = "chroma_db_rag" # Use a separate DB path for this exercise if desired
COLLECTION_NAME_PREFIX = "rag_chat_"

def get_device():
    """Gets the appropriate device for computation (CUDA, MPS, or CPU)."""
    # Check for MPS (Apple Silicon) first, then CUDA, then CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
         # Bug: is_available() can be true but mps backend might not be built.
         # Check if built explicitly. See https://github.com/pytorch/pytorch/issues/89727
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

device = get_device()
st.info(f"Using device: {device}")

@st.cache_resource
def get_embedding_model():
    """Loads the SentenceTransformer model (cached)."""
    print(f"Loading embedding model: {MODEL_NAME} onto device: {device}")
    # Specify trust_remote_code=True if necessary for the model
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("Embedding model loaded.")
    return model

@st.cache_resource
def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client."""
    print(f"Initializing ChromaDB client at path: {CHROMA_PATH}")
    # Create the directory if it doesn't exist
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    print("ChromaDB client initialized.")
    return client

# --- LLM Setup ---
@st.cache_resource
def get_llm_pipeline():
    """Loads the Hugging Face pipeline for the LLM (cached)."""
    print(f"Loading LLM pipeline for model: {LLM_MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        # Try loading with device_map='auto' for automatic distribution (good for larger models)
        # If it fails or causes issues, fall back to specific device loading
        try:
             model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_ID, device_map='auto', torch_dtype=torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32)
             print("Loaded model with device_map='auto'.")
        except Exception as e_auto:
             print(f"device_map='auto' failed ({e_auto}), attempting explicit device placement.")
             model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_ID).to(device) # Explicitly move to device

        # Determine the correct device index for the pipeline if not using device_map
        pipeline_device = -1 # Default to CPU
        if device == 'cuda':
             pipeline_device = 0 # Use GPU 0 if CUDA
        # Note: device='mps' in pipeline might be problematic, often better to let device_map handle or manually .to('mps') the model

        print(f"Setting up pipeline for device: {device} (pipeline device index: {pipeline_device})")

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.6, # Slightly lower temp for more factual RAG
            top_p=0.95,
            repetition_penalty=1.15,
            # Only specify device index if NOT using device_map='auto' effectively
            # If device_map worked, model is already on correct devices.
            # If device_map failed and we used .to(device), specify pipeline_device
            device=pipeline_device if model.device.type != 'meta' else -1 # Check if model is on meta device (device_map)
        )

        hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        print("LLM pipeline loaded successfully.")
        return hf_pipeline
    except Exception as e:
        st.error(f"Error loading LLM pipeline: {e}")
        if "Triton is not available" in str(e):
             st.warning("Triton kernel not available. Performance may be impacted. This is often fine.")
             # Attempt to load without device_map as a fallback
             try:
                 print("Retrying LLM load without device_map...")
                 model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_ID).to(device)
                 pipeline_device = 0 if device == 'cuda' else -1
                 pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512, device=pipeline_device)
                 hf_pipeline = HuggingFacePipeline(pipeline=pipe)
                 print("LLM pipeline loaded successfully on retry.")
                 return hf_pipeline
             except Exception as e_retry:
                 st.error(f"LLM pipeline retry failed: {e_retry}")
                 return None
        # Provide more specific advice based on common errors
        elif "expected device cpu but got cuda" in str(e).lower() or "expected device cuda but got cpu" in str(e).lower():
             st.warning("Device mismatch error. Try restarting the kernel/app. Ensure PyTorch and Transformers have compatible versions.")
        elif "mps" in str(e).lower():
             st.warning("Potential issue with MPS backend. Consider falling back to CPU by modifying get_device() or pipeline setup.")
        return None


# --- Core Functions ---

def extract_text_from_file(uploaded_file):
    """Extracts text content from an uploaded file using Tika."""
    try:
        print(f"Extracting text from: {uploaded_file.name}")
        # Read into memory once
        file_bytes = uploaded_file.getvalue()
        parsed = parser.from_buffer(file_bytes)
        content = parsed.get('content')
        if not content:
            st.error("Could not extract text content from the file.")
            return None
        print(f"Successfully extracted text (length: {len(content)}).")
        return content.strip()
    except Exception as e:
        st.error(f"Error parsing file with Tika: {str(e)}")
        # Add specific check for Tika server
        if "Tika server" in str(e):
             st.error("Ensure the Tika server is running. Download tika-server-standard jar and run: java -jar tika-server-standard-x.x.jar")
        return None

def split_text(text):
    """Splits text into manageable chunks."""
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Standard chunk size
        chunk_overlap=200, # Overlap helps maintain context between chunks
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
    embedding_model = get_embedding_model() # Ensure model is loaded

    # Use SentenceTransformer directly with ChromaDB
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME, device=device
    )

    # Check if collection exists, delete if it does (for simplicity in workshop)
    try:
        print("Checking for existing collection...")
        list_of_collections = [c.name for c in chroma_client.list_collections()]
        print(f"Existing collections: {list_of_collections}")
        if collection_name in list_of_collections:
            print(f"Deleting existing collection: {collection_name}")
            chroma_client.delete_collection(name=collection_name)
            time.sleep(0.5) # Short pause after deletion
    except Exception as e:
        st.warning(f"Could not list or delete collection {collection_name}: {e}")
        # Continue anyway, creation might handle it or fail informatively

    # Create collection
    try:
        print(f"Creating new collection: {collection_name}")
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef,
            metadata={"hnsw:space": "cosine"} # Use cosine distance
        )
        print(f"Successfully created collection: {collection_name}")
    except Exception as e:
         # Handle potential race condition or error if deletion failed but collection exists
        try:
            print(f"Collection creation failed ('{e}'), attempting to get existing collection: {collection_name}.")
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef
            )
            print(f"Successfully got existing collection: {collection_name}")
        except Exception as get_e:
             st.error(f"Fatal error: Failed to create or get collection {collection_name}: {get_e}")
             return None


    # Add documents in batches
    batch_size = 166 # ChromaDB batch limit can be around 166 for this model embedding size
    num_chunks = len(chunks)
    print(f"Adding {num_chunks} chunks to collection '{collection_name}' in batches of {batch_size}...")
    with st.spinner(f"Embedding and storing {num_chunks} document chunks..."):
        total_added = 0
        for i in range(0, num_chunks, batch_size):
            batch_start_time = time.time()
            batch_chunks = chunks[i:min(i + batch_size, num_chunks)] # Ensure not to exceed list bounds
            ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))] # Unique IDs

            if not batch_chunks:
                print(f"Skipping empty batch at index {i}")
                continue

            try:
                 print(f"Adding batch {i//batch_size + 1}/{(num_chunks + batch_size - 1)//batch_size} (size: {len(batch_chunks)})...")
                 # Embeddings are handled by ChromaDB when using embedding_function
                 collection.add(
                     documents=batch_chunks,
                     ids=ids
                 )
                 total_added += len(batch_chunks)
                 batch_end_time = time.time()
                 print(f"Batch {i//batch_size + 1} added in {batch_end_time - batch_start_time:.2f}s")
                 # Optional progress update in Streamlit
                 # progress = min(total_added / num_chunks, 1.0)
                 # st.progress(progress, text=f"Processed {total_added}/{num_chunks} chunks...")
            except Exception as e:
                st.error(f"Error adding batch starting at index {i} to ChromaDB: {e}")
                # Consider whether to stop or continue; stopping is safer.
                return None # Stop on error

    end_time = time.time()
    print(f"Finished adding {total_added} chunks to vector store. Time taken: {end_time - start_time:.2f} seconds")
    if total_added == num_chunks:
        st.success(f"Document processed and stored in vector database (Collection: {collection_name}). Time: {end_time - start_time:.2f}s")
        return collection_name # Return name for state management
    else:
        st.error(f"Failed to add all chunks. Added {total_added}/{num_chunks}.")
        return None

def query_vector_store(question, collection_name, n_results=3):
    """Queries the vector store for relevant chunks based on the question."""
    if not collection_name:
        st.error("Vector store collection name is not set.")
        return None

    start_time = time.time()
    print(f"Querying collection '{collection_name}' for question: '{question}'")
    chroma_client = get_chroma_client()

    # Get the embedding function associated with the collection
    # This ensures consistency, especially if EF details change
    try:
        collection = chroma_client.get_collection(name=collection_name) # No need to pass EF here if already set
        print(f"Retrieved collection '{collection_name}'")
    except Exception as e:
        st.error(f"Could not retrieve collection '{collection_name}': {e}")
        # Attempt to list collections to see if it exists
        try:
            st.info(f"Available collections: {[c.name for c in chroma_client.list_collections()]}")
        except Exception as list_e:
            st.warning(f"Could not list collections: {list_e}")
        return None # Indicate error or collection not found


    try:
        results = collection.query(
            query_texts=[question],
            n_results=n_results,
            include=['documents', 'distances'] # Include distances for potential filtering/display
        )
        end_time = time.time()
        print(f"Query completed in {end_time - start_time:.2f} seconds.")

        # Check if results are valid and documents exist
        if results and results.get('documents') and results['documents'][0]:
             retrieved_docs = results['documents'][0]
             retrieved_distances = results['distances'][0]
             print(f"Retrieved {len(retrieved_docs)} chunks.")
             # print(f"Distances: {retrieved_distances}") # Debugging
             # print(f"Docs: {retrieved_docs}") # Debugging
             return retrieved_docs # Return only the document texts
        else:
             print("Query returned no documents.")
             return [] # Return empty list if no documents found

    except Exception as e:
        st.error(f"Error querying ChromaDB collection '{collection_name}': {e}")
        return None # Indicate error

# --- RAG Chain Setup ---
def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    if not docs:
        return "No relevant context found."
    # Simple concatenation, separated by newlines and markers
    return "\n\n---\n\n".join(doc for doc in docs)

@st.cache_resource # Cache the chain setup based on the LLM instance
def setup_rag_chain(_llm): # Pass llm as argument
    """Sets up the Langchain RAG chain."""
    if _llm is None:
        st.error("LLM Pipeline not available. Cannot setup RAG chain.")
        return None

    print("Setting up RAG chain...")
    # Define the prompt template - Adjust based on model's instruction following ability
    template = """Based *only* on the following context, answer the question.
If the context doesn't contain the answer, state that you cannot answer from the provided context.
Do not use any prior knowledge. Be concise.

Context:
{context}

Question: {question}

Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    # Define the RAG chain using LCEL (Langchain Expression Language)
    rag_chain = (
        # The input to this chain is expected to be a dictionary: {"question": "...", "context": [...list of docs...]}
        {
            "context": RunnableLambda(lambda x: format_docs(x["context"])), # Format docs first
            "question": RunnablePassthrough() # Pass the question through
        }
        | RunnableLambda(lambda x: custom_rag_prompt.format(context=x["context"], question=x["question"]["question"])) # Format the prompt string
        | _llm # The LLM pipeline
        | StrOutputParser() # Parses the LLM output into a string
    )
    print("RAG chain setup complete.")
    return rag_chain


# --- Streamlit UI ---
st.set_page_config(layout="wide") # Use wider layout
st.title("📄 RAG Chatbot: Chat with your Document")
st.write("Upload a document (PDF, DOCX, TXT), ask questions, and get answers generated *only* from its content.")

# File Uploader
uploaded_file = st.file_uploader("1. Upload your document", type=['txt', 'pdf', 'doc', 'docx'])

# Initialize session state more robustly
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None # Initialize rag_chain state
if 'llm_pipeline' not in st.session_state:
     st.session_state.llm_pipeline = None # Initialize

# Load LLM pipeline once and cache it in session state
if st.session_state.llm_pipeline is None:
     with st.spinner("Loading Language Model (this may take a moment)..."):
         st.session_state.llm_pipeline = get_llm_pipeline() # Load LLM on startup/refresh

# Display LLM status
if st.session_state.llm_pipeline:
    st.sidebar.success(f"LLM Ready: {LLM_MODEL_ID}")
else:
    st.sidebar.error("LLM failed to load. Chat functionality disabled.")

# --- Document Processing Logic ---
if uploaded_file is not None:
    current_filename = uploaded_file.name
    # Check if it's a new file or if processing wasn't completed for the current file
    if st.session_state.uploaded_filename != current_filename or not st.session_state.processing_done:
        st.info(f"Processing new file: {current_filename}...")
        st.session_state.processing_done = False # Reset flag
        st.session_state.collection_name = None
        st.session_state.uploaded_filename = current_filename
        st.session_state.rag_chain = None # Reset chain when file changes

        with st.status("Processing document...", expanded=True) as status:
            st.write("1. Extracting text...")
            text_content = extract_text_from_file(uploaded_file)

            if text_content:
                st.write("2. Splitting text into chunks...")
                chunks = split_text(text_content)

                if chunks:
                    # Create a unique collection name
                    safe_filename = "".join(c if c.isalnum() else "_" for c in current_filename)
                    collection_name = f"{COLLECTION_NAME_PREFIX}{safe_filename}_{uuid.uuid4().hex[:8]}"
                    st.write(f"3. Creating vector store (Collection: {collection_name})...")
                    created_collection_name = setup_vector_store(chunks, collection_name)

                    if created_collection_name:
                        st.session_state.collection_name = created_collection_name
                        st.session_state.processing_done = True
                        # Setup RAG chain now that processing is done and LLM is available
                        if st.session_state.llm_pipeline:
                             # Pass the loaded LLM pipeline to the setup function
                             st.session_state.rag_chain = setup_rag_chain(st.session_state.llm_pipeline)
                             if st.session_state.rag_chain:
                                 status.update(label="Document processing complete!", state="complete", expanded=False)
                                 st.success(f"Document '{current_filename}' processed. Ready for questions.")
                             else:
                                 st.error("Failed to set up RAG chain after processing.")
                                 status.update(label="Processing failed (RAG Chain Setup)", state="error")
                                 st.session_state.processing_done = False # Mark as not ready
                        else:
                             st.error("LLM Pipeline not available. Cannot generate answers.")
                             status.update(label="Processing failed (LLM Load Error)", state="error")
                             st.session_state.processing_done = False # Mark as not ready if LLM failed
                    else:
                        st.error("Failed to set up vector store.")
                        status.update(label="Processing failed (Vector Store)", state="error")
                else:
                    st.error("Failed to split text into chunks.")
                    status.update(label="Processing failed (Text Splitting)", state="error")
            else:
                st.error("Failed to extract text from the file.")
                status.update(label="Processing failed (Text Extraction)", state="error")
    else:
        # Same file uploaded again, processing already done
        st.success(f"Document '{current_filename}' is already processed (Collection: {st.session_state.collection_name}). Ready for questions.")
        # Ensure RAG chain is set up if it wasn't (e.g., after refresh but LLM loaded later)
        if st.session_state.rag_chain is None and st.session_state.llm_pipeline:
             st.session_state.rag_chain = setup_rag_chain(st.session_state.llm_pipeline)
             if st.session_state.rag_chain:
                 st.info("RAG chain re-initialized.")
             else:
                 st.error("Failed to re-initialize RAG chain.")


# --- Chat Interface (only if processing is done and RAG chain is ready) ---
st.divider()
if st.session_state.processing_done and st.session_state.collection_name and st.session_state.rag_chain:
    st.header("2. Ask a Question")
    user_question = st.text_input("Enter your question about the document:", key="user_question_input")

    if user_question:
        st.markdown("### Answer:")
        with st.spinner("Thinking... (Querying vector store and generating answer)"):
            # 1. Query the vector store
            retrieved_docs = query_vector_store(user_question, st.session_state.collection_name, n_results=3) # Get top 3 chunks

            if retrieved_docs is None:
                st.error("Failed to query the vector store. Cannot generate answer.")
            elif not retrieved_docs:
                st.warning("Could not find relevant information in the document for your question.")
                st.markdown("> No relevant context found to answer the question.")
            else:
                # 2. Prepare input for the RAG chain
                chain_input = {"question": user_question, "context": retrieved_docs}

                # 3. Invoke the RAG chain
                try:
                    start_llm_time = time.time()
                    response = st.session_state.rag_chain.invoke(chain_input)
                    end_llm_time = time.time()

                    # 4. Display the response
                    st.markdown(response)
                    st.info(f"Answer generated in {end_llm_time - start_llm_time:.2f} seconds.")

                    # 5. Optionally display retrieved chunks in an expander
                    with st.expander("Show Retrieved Context Chunks"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text_area(f"chunk_{i+1}", doc, height=100, key=f"chunk_display_{i}_{user_question}", disabled=True)
                            st.divider()

                except Exception as e:
                    st.error(f"Error generating answer with LLM: {e}")
                    # Fallback or further debugging info
                    st.warning("Could not generate a final answer using the LLM.")
                    # Display raw chunks as a fallback
                    st.markdown("---")
                    st.subheader("Raw Retrieved Information:")
                    for i, doc in enumerate(retrieved_docs):
                         st.markdown(f"**Chunk {i+1}:**")
                         st.text_area(f"chunk_{i+1}_fallback", doc, height=100, key=f"chunk_display_fallback_{i}_{user_question}", disabled=True)
                         st.divider()


elif uploaded_file is None:
    st.info("Please upload a document to begin.")
elif not st.session_state.llm_pipeline:
     st.warning("LLM is not loaded. Please wait or check for errors.")
elif not st.session_state.processing_done:
     st.warning("Document processing is not complete or failed. Please check the status above.")

# Add footer or sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info:**")
st.sidebar.markdown(f"- Embedding: `{MODEL_NAME}`")
st.sidebar.markdown(f"- LLM: `{LLM_MODEL_ID}`")
st.sidebar.markdown(f"- Device: `{device.upper()}`")
if st.session_state.collection_name:
    st.sidebar.markdown(f"- DB Collection: `{st.session_state.collection_name}`") 