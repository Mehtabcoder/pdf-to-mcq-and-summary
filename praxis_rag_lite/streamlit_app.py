import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
import re

# --- Environment and Page Configuration ---
load_dotenv()
st.set_page_config(page_title="Praxis-RAG Pro", layout="wide")

# --- Initial Check for API Key ---
if "GOOGLE_API_KEY" not in os.environ:
    st.error("🚨 Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes all required session state variables."""
    keys_to_init = {
        "vector_store": None,
        "chat_history": [],
        "all_pages_content": [],
        "num_pages": 0,
        "processed_file_bytes": None,
        "uploaded_file_name": None,
        "mcqs_generated": [],
        "user_mcq_selections": {}
    }
    for key, value in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Core Processing & LLM Functions ---

def get_llm_response(prompt_template, input_data, temperature=0.7):
    """Generic function to get a response from the LLM."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    response = chain.invoke(input_data)
    return response.content

@st.cache_resource(show_spinner="⚙️ Analyzing Document... Please wait.")
def create_vector_store(file_bytes, file_name):
    """
    Creates and returns a FAISS vector store. This is the only part that should be cached.
    """
    temp_file_path = f"./temp_{file_name}"
    with open(temp_file_path, "wb") as f:
        f.write(file_bytes)
    
    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return vector_store

def load_document_details(file_bytes):
    """
    Extracts page content and count. This is NOT cached to ensure it always runs.
    """
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        st.session_state.num_pages = pdf_document.page_count
        st.session_state.all_pages_content = [page.get_text() for page in pdf_document]
        pdf_document.close()
    except Exception as e:
        st.error(f"Failed to read PDF details: {e}")
        st.session_state.num_pages = 0
        st.session_state.all_pages_content = []

# --- UI Components ---
st.title("🚀 Praxis-RAG ")
st.markdown("Upload a PDF to unlock summaries, Q&A, and interactive quizzes—all powered by generative AI.")

# --- Sidebar and File Handling Logic ---
with st.sidebar:
    st.header("📋 Document Controls")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if st.button("Start Over & Clear Document"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()
        st.rerun()

    if uploaded_file is not None:
        current_file_bytes = uploaded_file.getvalue()
        
        if st.session_state.get("processed_file_bytes") != current_file_bytes:
            st.session_state.uploaded_file_name = uploaded_file.name
            
            # Step 1: Always load page details (not cached)
            load_document_details(current_file_bytes)
            
            # Step 2: Create or get the vector store from cache
            st.session_state.vector_store = create_vector_store(
                current_file_bytes, st.session_state.uploaded_file_name
            )
            
            # Step 3: Mark as processed
            if st.session_state.vector_store is not None:
                 st.session_state.processed_file_bytes = current_file_bytes
            
            # Clear any data from a previous document
            st.session_state.chat_history = []
            st.session_state.mcqs_generated = []
            st.session_state.user_mcq_selections = {}

    if st.session_state.vector_store:
        st.success(f"**{st.session_state.uploaded_file_name}** is ready!")
        st.info(f"Total Pages: **{st.session_state.num_pages}**")
    else:
        st.info("⬆️ Please upload a document to begin.")


# --- Main Application Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Q&A", "📝 Summarizer", "❓ Quiz Generator", "🎯 Section Finder"])

# --- Tab 1: Chat Q&A ---
with tab1:
    st.subheader("Ask Questions About Your Document")
    if st.session_state.vector_store:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("View Sources"):
                        st.info(", ".join(message["sources"]))

        if user_question := st.chat_input("Ask anything about the document..."):
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            with st.spinner("Thinking..."):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer the user's question based *only* on the provided context. Cite the page number in brackets [Page X] for each piece of information. If the answer is not in the context, state that clearly.\n\nContext:\n{context}"),
                    ("human", "Question: {input}")
                ])
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                document_chain = create_stuff_documents_chain(llm, qa_prompt)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                response = retrieval_chain.invoke({"input": user_question})
                answer = response["answer"]
                unique_sources = sorted(list(set([f"Page {doc.metadata.get('page', -1) + 1}" for doc in response.get("context", [])])))

                ai_response = {"role": "ai", "content": answer, "sources": unique_sources}
                st.session_state.chat_history.append(ai_response)

                with st.chat_message("ai"):
                    st.write(answer)
                    if unique_sources:
                        with st.expander("View Sources"):
                            st.info(", ".join(unique_sources))
    else:
        st.info("Upload a PDF in the sidebar to start the chat.")

# --- Reusable Page Selector UI ---
def display_page_selector(tab_key_prefix):
    st.markdown(f"This PDF has **{st.session_state.num_pages}** pages.")
    page_selection_method = st.radio(
        "Select pages to use:",
        ("First 5 pages", "Specify Page Range"),
        key=f"{tab_key_prefix}_page_select",
        horizontal=True
    )
    start_page, end_page = 1, min(5, st.session_state.num_pages)
    if page_selection_method == "Specify Page Range":
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Start Page:", min_value=1, max_value=st.session_state.num_pages, value=1, key=f"{tab_key_prefix}_start")
        with col2:
            end_page = st.number_input("End Page:", min_value=start_page, max_value=st.session_state.num_pages, value=min(start_page + 4, st.session_state.num_pages), key=f"{tab_key_prefix}_end")
    if start_page > end_page:
        st.error("End page must be greater than or equal to start page.")
        return None, ""
    selected_pages_text = "\n".join(st.session_state.all_pages_content[start_page-1:end_page])
    info_message = f"Using content from **pages {start_page}-{end_page}**."
    return selected_pages_text, info_message

# --- Tab 2: Summarizer ---
with tab2:
    st.subheader("Generate a Concise Summary")
    if st.session_state.all_pages_content:
        selected_text, info_msg = display_page_selector("summary")
        st.info(info_msg)
        if st.button("✨ Generate Summary", disabled=not selected_text):
            with st.spinner("Creating summary..."):
                summary_prompt = """
                As an expert summarizer, create a concise, easy-to-read summary of the following text. 
                Focus on the main arguments, key findings, and conclusions. Use bullet points for clarity if appropriate.
                The summary should not exceed 400 words.
                Document Text:
                {document_text}
                Concise Summary:
                """
                summary = get_llm_response(summary_prompt, {"document_text": selected_text})
                st.success("Summary Generated!")
                st.write(summary)
    else:
        st.info("Upload a PDF to use the summarizer.")

# --- Tab 3: Quiz Generator ---
with tab3:
    st.subheader("Generate an Interactive Quiz")
    if st.session_state.all_pages_content:
        selected_text_mcq, info_msg_mcq = display_page_selector("mcq")
        st.info(info_msg_mcq)
        num_mcqs = st.slider("Number of MCQs to generate:", min_value=2, max_value=6, value=3, key="mcq_num")
        if st.button("🧠 Generate Quiz", disabled=not selected_text_mcq):
            st.session_state.mcqs_generated = []
            st.session_state.user_mcq_selections = {}
            with st.spinner("Generating quiz questions..."):
                mcq_prompt = f"""
                Based on the document text provided, generate exactly {num_mcqs} multiple-choice questions (MCQs).
                Each question must test a key concept from the text.
                IMPORTANT: Strictly follow this format for EACH question:
                Q#: [Question text]?
                A) [Option A]
                B) [Option B]
                C) [Option C]
                D) [Option D]
                Correct Answer: [Letter of the correct option]
                ---[SEPARATOR]---
                Document Text:
                {selected_text_mcq}
                """
                raw_mcqs = get_llm_response(mcq_prompt, {"document_text": selected_text_mcq}, temperature=0.5)
                mcq_blocks = raw_mcqs.strip().split("---[SEPARATOR]---")
                mcq_pattern = re.compile(
                    r"Q\d*: (.*?)\n(A\).+?)\n(B\).+?)\n(C\).+?)\n(D\).+?)\nCorrect Answer: ([A-D])",
                    re.DOTALL | re.IGNORECASE
                )
                for block in mcq_blocks:
                    match = mcq_pattern.search(block.strip())
                    if match:
                        question, opt_a, opt_b, opt_c, opt_d, correct_letter = match.groups()
                        st.session_state.mcqs_generated.append({
                            "question": question.strip(),
                            "options": [opt_a.strip(), opt_b.strip(), opt_c.strip(), opt_d.strip()],
                            "correct": correct_letter.strip().upper()
                        })
        if st.session_state.mcqs_generated:
            st.success(f"{len(st.session_state.mcqs_generated)} questions generated!")
            for idx, mcq in enumerate(st.session_state.mcqs_generated):
                st.markdown("---")
                st.markdown(f"**Question {idx + 1}: {mcq['question']}**")
                options_with_labels = [opt[3:] for opt in mcq['options']]
                user_choice_label = st.radio(
                    "Your Answer:", options_with_labels, key=f"mcq_{idx}", label_visibility="collapsed"
                )
                user_choice_letter = chr(ord('A') + options_with_labels.index(user_choice_label))
                st.session_state.user_mcq_selections[idx] = user_choice_letter
                if st.button("Show Answer", key=f"ans_btn_{idx}"):
                    correct_answer_text = next((opt for opt in mcq['options'] if opt.startswith(mcq['correct'])), "N/A")
                    if st.session_state.user_mcq_selections.get(idx) == mcq['correct']:
                        st.success(f"✅ Correct! The answer is **{correct_answer_text}**")
                    else:
                        user_answer_text = next((opt for opt in mcq['options'] if opt.startswith(st.session_state.user_mcq_selections.get(idx, ''))), "N/A")
                        st.error(f"❌ Incorrect. You chose: {user_answer_text}")
                        st.info(f"The correct answer is: **{correct_answer_text}**")
    else:
        st.info("Upload a PDF to generate a quiz.")

# --- Tab 4: Section Finder ---
with tab4:
    st.subheader("Find Specific Sections by Query")
    if st.session_state.vector_store:
        st.info("This feature searches the entire document for relevant text chunks.")
        if section_query := st.text_input("Describe the section you're looking for:", placeholder="e.g., 'the methodology for the climate study'"):
            with st.spinner("Searching..."):
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                relevant_docs = retriever.invoke(section_query)
                if relevant_docs:
                    st.success("Found relevant sections:")
                    for i, doc in enumerate(relevant_docs):
                        page_num = doc.metadata.get('page', -1) + 1
                        with st.expander(f"**Result {i+1} (from Page {page_num})**"):
                            st.markdown(f"**Location:** Page {page_num}")
                            st.write(doc.page_content)
                else:
                    st.warning("No relevant sections found for that query.")
    else:
        st.info("Upload a PDF to find sections.")
