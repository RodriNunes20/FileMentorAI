# ------------ Imports ------------ #
import streamlit as st
import chromadb
from transformers import pipeline
from pathlib import Path
import tempfile
from tempfile import NamedTemporaryFile
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from contextlib import contextmanager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from datetime import datetime
import base64, streamlit as st
import base64, os


# ------------ Functions ------------ #
def set_bg(image_filename: str): 
    """
    Embed a local image as a full-page background.
    Works with .jpg / .png in the same folder as this script.
    """
    file_path = os.path.join(os.path.dirname(__file__), image_filename)
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Set a fixed background image (local jpg/png) using base64.
set_bg("background.jpg")
st.set_page_config(page_title="Doc-QA ChatBot", page_icon="🤖", layout="wide")

#Returns: Markdown string of file contents.
def convert_to_markdown(file_path: str) -> str:
    fpath = Path(file_path)
    ext = fpath.suffix.lower()
    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts, backend=DoclingParseV2DocumentBackend)}
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")
    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")
    if ext == ".txt":
        try:
            return fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return fpath.read_text(encoding="latin-1", errors="replace")
    raise ValueError(f"Unsupported extension: {ext}")

# ignore 'not found' errors when deleting
def reset_collection(client, collection_name: str):
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name)

def add_docs_list_to_chroma(docs, collection):
    """
    docs        List[dict]  – each item = {"filename": str, "content": str}
    collection  chromadb.Collection  – already created / reset

    Calls your existing add_text_to_chromadb() for every doc.
    """
    for d in docs:
        add_text_to_chromadb(d["content"], d["filename"], collection.name)

def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_text(text)
    if not hasattr(add_text_to_chromadb, 'client'):
        add_text_to_chromadb.client = chromadb.Client()
        add_text_to_chromadb.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        add_text_to_chromadb.collections = {}
    if collection_name not in add_text_to_chromadb.collections:
        try:
            collection = add_text_to_chromadb.client.get_collection(name=collection_name)
        except:
            collection = add_text_to_chromadb.client.create_collection(name=collection_name)
        add_text_to_chromadb.collections[collection_name] = collection
    collection = add_text_to_chromadb.collections[collection_name]
    for i, chunk in enumerate(chunks):
        embedding = add_text_to_chromadb.embedding_model.encode(chunk).tolist()
        metadata = {"filename": filename, "chunk_index": i, "chunk_size": len(chunk)}
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata],
            ids=[f"{filename}_chunk_{i}"]
        )
    return collection

# save to temp file so docling can open by path
def convert_uploaded_files(files):
    converted = []
    for f in files:
        with NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
            tmp.write(f.getvalue())
            md = convert_to_markdown(tmp.name)
        converted.append({"filename": f.name, "content": md})
    return converted

def show_document_stats():
    st.info("📊 Stats feature not implemented yet – coming soon.")

def get_answer_with_source(collection, question):
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    distances = results["distances"][0]
    if not docs or min(distances) > 0.8:
        return "I don't have information about that topic in your document.", "No relevant source found"
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    answer = response[0]['generated_text'].strip()
    # Use the first document as the source for simplicity
    source = docs[0] if docs else "No source"
    return answer, source

def show_document_manager():
    st.markdown("#### <span style='color:#2196f3'>Manage Documents</span>", unsafe_allow_html=True)

    if not st.session_state.converted_docs:
        st.info("No documents uploaded yet.")
        return

    docs = st.session_state.converted_docs 
    
    for i, doc in enumerate(st.session_state.converted_docs):
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.write(f"📄 {doc['filename']}")
            st.write(f"   Words: {len(doc['content'].split())}")

        with col2:
            if st.button("Preview", key=f"preview_{i}"):
                st.session_state[f"show_preview_{i}"] = True

        # Two-step confirmation stored via session flag
        with col3:
            key_del  = f"del_{i}"
            key_cfm  = f"cfm_{i}"
            key_flag = f"pending_{i}"       # flag to track deletion confirmation

            # 1st cliq  ➜  "Delete" button
            if st.button("Delete", key=key_del, type="primary"):
                st.session_state[key_flag] = True
                st.rerun()

            # 2nd cliq  ➜  "Click again to confirm" button
            if st.session_state.get(key_flag):
                if st.button("Click again to confirm", key=key_cfm, type="secondary"):
                    docs.pop(i)                                           # remove
                    st.session_state.collection = reset_collection(
                        chromadb.Client(), "user_docs"
                    )
                    add_docs_list_to_chroma(docs, st.session_state.collection)
                    st.session_state.pop(key_flag)                        # clear flag
                    st.rerun()

        # optional preview expander
        if st.session_state.get(f"show_preview_{i}", False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                snippet = doc["content"][:500]
                st.text(snippet + ("…" if len(doc["content"]) > 500 else ""))
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f"show_preview_{i}"] = False
                    st.rerun()

# FEATURE 3 – Search history
def add_to_search_history(question, answer, source):
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    st.session_state.search_history.insert(0, {
        "question": question,
        "answer": answer,
        "source": source,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    st.session_state.search_history = st.session_state.search_history[:10]

def show_search_history():
    st.subheader("🕒 Recent Searches")
    if not st.session_state.get("search_history"):
        st.info("No searches yet.")
        return
    for item in st.session_state.search_history:
        with st.expander(f"Q: {item['question'][:50]}… ({item['timestamp']})"):
            st.write("**Question:**", item['question'])
            st.write("**Answer:**", item['answer'])
            st.write("**Source:**", item['source'])

# FEATURE 5 – Tabbed interface
def create_tabbed_interface():
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📁 Upload", "❓ Ask Questions", "📋 Manage", "📊 Stats"]
    )

    # ------------ Upload tab ------------ #
    with tab1:
        st.markdown("#### <span style='color:#2196f3'>Upload & Convert Documents</span>", unsafe_allow_html=True)
        uploads = st.file_uploader(
             "Choose files", type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True
        )
        if st.button("Convert & Add", type="primary"):
            if uploads:
                with st.spinner("🔄 Crunching words…"):
                    new_docs = convert_uploaded_files(uploads)
                if new_docs:
                    add_docs_list_to_chroma(new_docs, st.session_state.collection)
                    st.session_state.converted_docs.extend(new_docs)
                st.success(f"Added {len(new_docs)} documents!")
                st.info("💡 Tip – switch to **Ask Questions** tab once processing is done.")

# ------------ Ask-questions tab ------------ #
    with tab2:
        st.markdown("#### <span style='color:#2196f3'>Ask Questions</span>", unsafe_allow_html=True)

        if st.session_state.converted_docs:
            q = st.text_input("Your question:")
            st.markdown("<style>input:focus{border:2px solid #ff9800 !important;}</style>", unsafe_allow_html=True)
            if st.button("Get Answer", type="primary"):
                if q:
                    with st.spinner("🧠 Reasoning…"):
                        ans, src = get_answer_with_source(
                st.session_state.collection, q
             )

                    st.write("**Answer:**", ans)
                    st.write(f"**Source:** {src}")
                    add_to_search_history(q, ans, src)
        else:
            st.info("Upload documents first!")
        show_search_history()

    # ------------ Manage tab ------------ #
    with tab3:
        show_document_manager()

    # ------------ Stats tab ------------ #
    with tab4:
        docs = st.session_state.converted_docs
        if not docs:
            st.info("Upload docs to see stats.")
            return
        total_words = sum(len(d['content'].split()) for d in docs)
        st.metric("Total documents", len(docs))
        st.metric("Total words indexed", total_words)

# ------------ MAIN ------------ #
def main():
    # init session lists only once
    if "converted_docs" not in st.session_state:
        st.session_state.converted_docs = []
    if "collection" not in st.session_state:
        client = chromadb.Client()
        st.session_state.collection = reset_collection(client, "user_document")

    st.title("🤖 FileMentorAI – Document Q&A")
    st.markdown("### 🛠️ Welcome! Upload documents, ask questions, manage them & view stats — all in one place.")
    create_tabbed_interface()

if __name__ == "__main__":
    st.markdown("<hr><center>Built for **MADA Course 2025** | Rodrigo Nunes</center>", unsafe_allow_html=True)
    main()