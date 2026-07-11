
import streamlit as st
import os

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Streamlit UI ---
st.set_page_config(page_title="MeetingMind AI Agent", layout="wide")
st.title("🧠 MeetingMind AI Agent")
st.markdown("--- Say goodbye to information overload. Ask questions, get summaries, and extract key decisions from your meeting transcripts with ease. ---")

# Initialize session state variables if they don't exist
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False

# Gemini API Key input
gemini_api_key = st.secrets["GOOGLE_API_KEY"]

# --- Main Application Logic ---
st.write("### Upload your Meeting Transcript (PDF)")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.session_state.pdf_uploaded = True
    st.success("PDF uploaded successfully!")

    # Save the uploaded file temporarily
    with open("uploaded_meeting.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    doc_filename = "uploaded_meeting.pdf"

    st.write("### Processing Document...")

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
    with st.spinner("Embedding chunks and creating vector store (this may take a moment)..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="my_document"
        )
        st.success("Vector store created successfully!")

    st.write("### Document Ready for Q&A!")

    # Display a text area for questions and a button to ask
    question_input = st.text_area("Ask a question about the meeting transcript:", key="question_area")

    if st.button("Get Answer"):
      if st.session_state.vectorstore is None:
    st.error("Please upload a PDF and wait for it to process.")
else:
            # Setup LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                google_api_key=gemini_api_key,
                temperature=0
            )

            def ask_document(question, show_chunks=False):
                """Ask a question about the document using RAG + Gemini."""

                # Step 1: Retrieve relevant chunks
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                retrieved_chunks = retriever.invoke(question)

                # Step 2: Build the context string from retrieved chunks
                context = "\n\n---\n\n".join([chunk.page_content for chunk in retrieved_chunks])

                # Step 3: Build the prompt
                user_prompt = f"""Answer the question below using ONLY the context provided.
If the answer is not in the context, say "I don't find that information in the document."
Do not use any outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}
"""

                # Step 4: Ask Gemini
                response = llm.invoke(user_prompt)

                # Get Gemini's answer
                answer = response.content # Assuming response.content is directly the string now or handles the list of dicts

                return answer, retrieved_chunks

            with st.spinner("Getting answer from Gemini..."):
                answer, source_chunks = ask_document(question_input)
                st.write("### Answer:")
                st.write(answer)

                st.write("### Source Chunks Used:")
                for i, chunk in enumerate(source_chunks):
                    st.write(f"**Chunk {i+1}:**")
                    st.info(chunk.page_content)

elif not st.session_state.pdf_uploaded:
    st.info("Please upload a PDF meeting transcript to get started!")

