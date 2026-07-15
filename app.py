import streamlit as st
import logging

# Silence Streamlit's local_sources_watcher logs to avoid console spam from optional dependencies (like torchvision)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# ==========================================
# RAG PIPELINE LIBRARIES & DEPENDENCIES
# ==========================================
from PyPDF2 import PdfReader                                    # For reading and extracting text from uploaded PDF files
from langchain_cohere import ChatCohere                          # Cohere's official Chat API integration for generation
from langchain_community.vectorstores import Chroma             # ChromaDB vector database to store and query text embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings # Local open-source models for generating text embeddings
from langchain_core.runnables import RunnablePassthrough        # Helper to pass original inputs through the LCEL pipeline
from langchain_core.output_parsers import StrOutputParser        # Parser to clean and format LLM output into raw strings
from langchain_core.prompts import PromptTemplate               # Template to structure queries and retrieved context for the LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter # Splitter to divide text into smaller semantic chunks

# ==========================================
# WINDOWS COMPATIBILITY WORKAROUND (SQLITE3)
# ==========================================
# ChromaDB requires a newer version of SQLite than what is default on some systems (especially Linux).
# On Windows, we catch the ImportError and use Python's built-in sqlite3, which is already up to date.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass


# ==========================================
# STREAMLIT UI CONFIGURATION & STYLING
# ==========================================
# Configure Streamlit's page properties
st.set_page_config(page_title="Cohere RAG", layout="wide")

# Inject custom CSS to design a sleek, premium container layout and style widgets
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .stApp {
        max-width: 1500px;
        margin: 0 auto;
    }
    .st-bw {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR CONFIGURATION (INPUTS & UPLOADS)
# ==========================================
with st.sidebar:
    # Display logo/header image
    st.image("https://cdn.sanity.io/images/rjtqmwfu/production/5a374837aab376bb677b3a968c337532ea16f6cb-800x600.png?rect=0,90,800,420&w=1200&h=630", width=200)
    st.title("PDF Upload & Settings")
    
    # User inputs for PDF file and Cohere API credentials
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    cohere_api_key = st.text_input("Enter your Cohere API key", type="password")

    # Status indicators
    if uploaded_file:
        st.success("PDF uploaded successfully!")
    if cohere_api_key:
        st.success("API key entered!")


# ==========================================
# MAIN INTERFACE HEADERS & SESSION STATE
# ==========================================
st.title("Cohere RAG APP")
st.markdown("---")

# Initialize session state variables to cache processed data across reruns
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'cohere_llm' not in st.session_state:
    st.session_state.cohere_llm = None

# Initialize ChatCohere when the user provides an API key
if cohere_api_key and st.session_state.cohere_llm is None:
    # Using 'command-r-08-2024', which is Cohere's flagship model optimized for RAG and tool use
    st.session_state.cohere_llm = ChatCohere(model="command-r-08-2024", temperature=0.1, cohere_api_key=cohere_api_key)


# ==========================================
# RAG PIPELINE: DOCUMENT PROCESSING
# ==========================================
# Check if a PDF is uploaded and credentials are provided
if uploaded_file is not None and cohere_api_key and not st.session_state.pdf_processed:
    with st.spinner("Processing PDF... This may take a moment."):
        
        # 1. DEFINE EMBEDDINGS MODEL
        # We use an open-source sentence-transformers model from HuggingFace to convert text chunks into vector embeddings.
        # 'all-MiniLM-L6-v2' maps sentences & paragraphs to a 384-dimensional dense vector space.
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # 2. READ PDF AND EXTRACT TEXT
        # Read the raw binary stream of the uploaded PDF and combine text from all pages.
        pdf_reader = PdfReader(uploaded_file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        # 3. SPLIT TEXT INTO SEMANTIC CHUNKS
        # Large documents exceed LLM context windows. We split the text into manageable chunks.
        # chunk_size: Max characters per chunk.
        # chunk_overlap: Overlapping characters between adjacent chunks to maintain context continuity.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=['\n', '\n\n', ' ', '']
        )
        chunks = text_splitter.split_text(text=pdf_text)

        # 4. CREATE VECTORSTORE (CHROMADB)
        # Store the extracted text chunks and their embeddings in ChromaDB, a lightweight vector database.
        # During search, we will query this database to retrieve chunks that are semantically similar to the user's question.
        vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="./chroma_db")
        
        # Cache the initialized vector database and mark processing as complete
        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_processed = True
    
    st.success("PDF processed successfully!")


# ==========================================
# RAG PIPELINE: QUESTION & ANSWER GENERATION
# ==========================================
# Get user question input
question = st.text_input("What would you like to know about the PDF content?", placeholder="Enter your question here...")

if st.session_state.pdf_processed and question and st.session_state.cohere_llm:
    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            # Load cached vectorstore
            vectorstore = st.session_state.vectorstore
            
            # 1. DEFINE RETRIEVER
            # Configure retriever to fetch the top 3 (k=3) most semantically similar chunks.
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

            # 2. DEFINE SYSTEM PROMPT
            # Anchor the LLM's response strictly to the retrieved context to minimize hallucinations.
            prompt_template = """You are a helpful AI assistant. Answer the question as accurately and precisely as possible using only the provided context.
                                If the answer is not contained in the context, respond with "answer not available in context."
                                Context:{context}
                                Question:{question}
                                Answer:"""

            prompt = PromptTemplate.from_template(template=prompt_template)

            # Helper function to format retrieved documents into a single block of context text
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # 3. BUILD THE RAG CHAIN (LCEL PIPELINE)
            # - retriever | format_docs: Automatically fetches and formats context from ChromaDB.
            # - RunnablePassthrough(): Passes the user's question unchanged to the prompt.
            # - prompt: Combines context and question into the prompt template.
            # - cohere_llm: Sends the prompt to Cohere's Chat API.
            # - StrOutputParser: Extracts and returns the final response string.
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | st.session_state.cohere_llm
                | StrOutputParser()
            )

            # Invoke the pipeline and display the result
            answer = rag_chain.invoke(question)
            st.markdown("### Answer:")
            st.info(answer)

# Handle UI feedback messages based on application state
elif not st.session_state.pdf_processed:
    st.warning("Please upload a PDF file and enter your Cohere API key first.")
elif st.session_state.pdf_processed and not question:
    st.info("PDF processed. Please enter a question to get an answer.")
elif st.session_state.pdf_processed and question and not st.session_state.cohere_llm:
    st.error("Cohere LLM not initialized. Please check your API key.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Cohere.")
