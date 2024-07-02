import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatAnthropic
from langchain.prompts import PromptTemplate
from collections import defaultdict

# Load environment variables
load_dotenv()

# Set the path to your Obsidian vault
VAULT_PATH = r"C:\Users\etherealheim\Documents\Etherealheim Vault"
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), 'db')

# Custom prompt templates
SUMMARY_TEMPLATE = """
You are an AI assistant tasked with summarizing notes from an Obsidian vault. Focus only on the actual content related to {question}, ignoring any metadata, links, or tags that might be present. If there's no relevant information, state that clearly.

Relevant notes:
{summaries}

Please provide a concise summary of the key points related to {question} from these notes. If there's no substantial information, suggest ways the user could add relevant content to their vault.

Summary:
"""

FOLLOWUP_TEMPLATE = """
Based on the summary about {topic}, suggest 3 follow-up questions to explore this topic further:

{summary}

Follow-up questions:
1.
2.
3.
"""

IMPROVE_KNOWLEDGE_TEMPLATE = """
Based on the existing knowledge about {topic}, suggest ways to expand and improve understanding:

{summary}

Suggestions to improve knowledge:
1.
2.
3.
"""

@st.cache_resource
def load_and_process_documents():
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Check if the vector store already exists
        if os.path.exists(PERSIST_DIRECTORY):
            status_text.text("Loading existing vector store...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            status_text.text("Vector store loaded successfully!")
            time.sleep(2)
            status_text.empty()
            progress_bar.empty()
            return db

        status_text.text("Loading documents...")
        loader = DirectoryLoader(VAULT_PATH, glob="**/*.md")
        documents = loader.load()
        progress_bar.progress(25)

        status_text.text("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        progress_bar.progress(50)

        status_text.text("Setting up embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        progress_bar.progress(75)

        status_text.text("Creating vector store...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
        db.persist()
        progress_bar.progress(100)

        status_text.text("Document processing complete!")
        time.sleep(2)
        status_text.empty()
        progress_bar.empty()

        return db
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None

@st.cache_resource
def setup_llm():
    try:
        claude_api_key = os.getenv("CLAUDE_API_KEY")
        if not claude_api_key:
            st.error("Claude API key not found. Please check your .env file.")
            return None
        return ChatAnthropic(anthropic_api_key=claude_api_key)
    except Exception as e:
        st.error(f"Error setting up LLM: {str(e)}")
        return None

# Streamlit UI
st.title("Obsidian Vault Q&A")

# Load documents and set up the retrieval chain
with st.spinner("Initializing... This may take a few moments."):
    db = load_and_process_documents()
    llm = setup_llm()

if db is not None and llm is not None:
    summary_prompt = PromptTemplate(template=SUMMARY_TEMPLATE, input_variables=["summaries", "question"])
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": summary_prompt,
            "document_variable_name": "summaries"
        }
    )

    # User input
    user_question = st.text_input("Ask a question about your Obsidian vault:")

    if user_question:
        try:
            with st.spinner("Processing your question..."):
                # Get the answer
                result = qa_chain({"question": user_question})
            
            # Display the summary
            st.subheader("Summary of Your Notes:")
            st.write(result["answer"])
            
            # Group and display source documents
            st.subheader("Sources:")
            sources = defaultdict(list)
            for doc in result["source_documents"]:
                source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                sources[source_file].append(doc.page_content)
            
            for source_file, contents in sources.items():
                with st.expander(f"Source: {source_file}"):
                    for i, content in enumerate(contents, 1):
                        st.write(f"Excerpt {i}:")
                        st.write(content[:200] + "..." if len(content) > 200 else content)
                        if st.button(f"View full content for Excerpt {i}", key=f"{source_file}_{i}"):
                            st.write(content)
            
            # Follow-up questions
            if st.button("Generate Follow-up Questions"):
                followup_prompt = PromptTemplate(template=FOLLOWUP_TEMPLATE, input_variables=["topic", "summary"])
                followup_result = llm(followup_prompt.format(topic=user_question, summary=result["answer"]))
                st.subheader("Follow-up Questions:")
                st.write(followup_result)
            
            # Improve knowledge
            if st.button("Suggest Ways to Improve Knowledge"):
                improve_prompt = PromptTemplate(template=IMPROVE_KNOWLEDGE_TEMPLATE, input_variables=["topic", "summary"])
                improve_result = llm(improve_prompt.format(topic=user_question, summary=result["answer"]))
                st.subheader("Ways to Improve Knowledge:")
                st.write(improve_result)
            
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
else:
    st.error("Failed to initialize the application. Please check your setup and try again.")

# Add a sidebar with information about the vault
st.sidebar.title("Vault Information")
if db is not None:
    st.sidebar.write(f"Number of documents: {len(db.get()['ids'])}")
    st.sidebar.write(f"Vault path: {VAULT_PATH}")