
import streamlit as st
import os
from datetime import datetime # Import datetime

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# Authentication and DB
from auth_db import init_db, register_user, verify_user, save_conversation, get_conversations

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="MeetingMind AI Agent",
    layout="centered", # Changed to 'centered' for a more focused look, 'wide' is also an option
    initial_sidebar_state="expanded"
)

# Custom CSS for a more polished look
st.markdown("""
    <style>
    .stApp { /* Main app container */
        background-color: #1e1e1e; /* Dark gray background */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #f0f2f6; /* Light text for dark background */
    }
    .st-emotion-cache-z5fcl4 { /* Target the main block container if layout is 'centered' */
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6 { /* All headers */
        color: #2F80ED; /* A nice blue color */
        text-align: center;
    }
    .stButton>button { /* Styling for buttons */
        background-color: #2F80ED;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2667CC;
    }
    .stFileUploader {
        border: 2px dashed #2F80ED;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stTextInput>div>div>input { /* Input field styling */
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        background-color: #282828; /* Darker input background */
        color: #f0f2f6; /* Light text */
    }
    .stTextArea>div>div>textarea { /* Text area styling */
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        background-color: #282828; /* Darker textarea background */
        color: #f0f2f6; /* Light text */
    }
    .css-1fv8s86.eqr7zpz4 { /* Target the info box specifically */
        background-color: #004d66; /* Darker cyan */
        border-left: 5px solid #00bcd4; /* Cyan bar */
        color: #e0f7fa; /* Lighter text */
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 MeetingMind AI Agent")
st.markdown("<h3 style='text-align: center; color: #555;'>Your Intelligent Meeting Assistant</h3>", unsafe_allow_html=True)

# Initialize the database
init_db()

# Session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None


def show_auth_page():
    """Displays the login/signup forms."""
    st.markdown("""
    Welcome! Please Login or Sign Up to use the MeetingMind AI Agent.
    """)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            user_id = verify_user(login_username, login_password)
            if user_id:
                st.session_state.authenticated = True
                st.session_state.user_id = user_id
                st.session_state.username = login_username
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        st.subheader("Sign Up")
        new_username = st.text_input("New Username", key="new_username")
        new_password = st.text_input("New Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        if st.button("Register"): # This button now applies to the modified CSS
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            elif len(new_username) < 3 or len(new_password) < 6:
                st.error("Username must be at least 3 characters and password at least 6 characters long.")
            else:
                if register_user(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists or registration failed.")


def logout():
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.vectorstore = None # Clear vectorstore on logout
    st.session_state.pdf_uploaded = False
    st.rerun()


if not st.session_state.authenticated:
    show_auth_page()
else:
    # --- Sidebar for Instructions and About ---
    with st.sidebar:
        st.header(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"): # This button now applies to the modified CSS
            logout()
        st.markdown("--- ")
        st.header("📚 How to Use")
        with st.expander("Click here for detailed instructions", expanded=True):
            st.markdown("""
            1.  **Upload PDF**: Use the file uploader below to select your meeting transcript in PDF format. The app will automatically process it.
            2.  **Ask Questions**: Once processed, type your questions about the meeting content into the text area.
            3.  **Get Answers**: Click 'Get Answer' to receive a concise, accurate response generated by Google Gemini, based *only* on your uploaded document.
            """)
        st.markdown("--- ")
        st.header("💡 About This App")
        st.info("""
        MeetingMind AI Agent helps you cut through lengthy meeting notes to find key information fast. It uses advanced AI to summarize, extract decisions, and answer questions directly from your documents.

        Developed as a capstone project utilizing LangChain, ChromaDB, HuggingFace Embeddings, and Google Gemini.
        """)

        st.markdown("--- ")
        st.header("Past Conversations")
        user_conversations = get_conversations(st.session_state.user_id)
        if user_conversations:
            for i, (timestamp, question, answer) in enumerate(user_conversations):
                with st.expander(f"**{datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')}**: {question[:50]}..."):
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")
        else:
            st.info("No past conversations found.")

    # Initialize session state variables if they don't exist
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False

    # Gemini API Key input (using st.secrets for security)
    gemini_api_key = st.secrets.get("GOOGLE_API_KEY", "")

    # --- Main Application Logic ---

    # Upload Section
    with st.container(border=True):
        st.subheader("Upload Your Meeting Transcript (PDF)")
        uploaded_file = st.file_uploader("Choose a PDF file to analyze", type="pdf", label_visibility="collapsed")

        if uploaded_file is not None:
            st.session_state.pdf_uploaded = True
            st.success("PDF uploaded successfully! Now processing...")

            # Save the uploaded file temporarily
            with open("uploaded_meeting.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            doc_filename = "uploaded_meeting.pdf"

            st.write("#### Document Processing Status")

            # Load the document
            with st.spinner("Loading PDF..."):
                loader = PyPDFLoader(doc_filename)
                documents = loader.load()
                st.success(f"Loaded {len(documents)} pages.")

            # Split into chunks
            with st.spinner("Splitting document into chunks..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100
                )
                chunks = text_splitter.split_documents(documents)
                st.success(f"Created {len(chunks)} chunks.")

            # Embed chunks and store in ChromaDB
            with st.spinner("Embedding chunks and creating vector store (this may take a moment)... "):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                st.success("Vector store created successfully!")

    # Q&A Section
    if st.session_state.pdf_uploaded and st.session_state.vectorstore is not None:
        st.markdown("--- ")
        with st.container(border=True):
            st.subheader("Ask a Question About the Meeting")
            question_input = st.text_area("Enter your question here:", key="question_area", height=100)

            if st.button("Get Answer"): # This button now applies to the modified CSS
                if not gemini_api_key:
                    st.error("Google API Key not found in Streamlit Secrets. Please ensure it's set.")
                else:
                    # Setup LLM
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.5-flash",
                        google_api_key=gemini_api_key,
                        temperature=0
                    )

                    def ask_document(question):
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 5}
                        )

                        retrieved_chunks = retriever.invoke(question)

                        unique_chunks_content = set()
                        deduplicated_chunks = []
                        for chunk in retrieved_chunks:
                            if chunk.page_content not in unique_chunks_content:
                                unique_chunks_content.add(chunk.page_content)
                                deduplicated_chunks.append(chunk)

                        context = "\n\n---\n\n".join(
                            [chunk.page_content for chunk in deduplicated_chunks]
                        )

                        user_prompt = f"""You are an AI assistant designed to extract and summarize information from meeting transcripts.\nBased on the following CONTEXT from the meeting, please answer the QUESTION.\nYour answer should be as comprehensive and informative as possible, drawing all relevant details and making logical inferences *only* from the provided CONTEXT.\nIf the CONTEXT does not contain enough information to provide a direct answer, or if the information is entirely absent, explain what information is missing or state that the answer cannot be found within the document.\nDo not use any external knowledge.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n"""

                        response = llm.invoke(user_prompt)
                        answer = response.content[0]["text"]
                        return answer, deduplicated_chunks

                    with st.spinner("Generating answer with Gemini..."):
                        answer, source_chunks = ask_document(question_input)

                        st.success("Answer Generated!")
                        st.markdown("#### Answer:")
                        st.write(answer)

                        # Save conversation
                        save_conversation(st.session_state.user_id, question_input, answer)

                        with st.expander("View Source Chunks Used (for context)"):
                            for i, chunk in enumerate(source_chunks):
                                st.markdown(f"**Chunk {i+1}:**")
                                st.info(chunk.page_content)
    else:
        st.info("Please upload a PDF document to begin asking questions.")
