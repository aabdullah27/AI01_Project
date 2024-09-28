import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS  # FAISS instead of Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
import tempfile
from fpdf import FPDF
import hashlib

# Set the page config to add title and favicon
st.set_page_config(page_title="AI MCQ Generation from PDF", page_icon="🤖", layout="wide")


# Function to generate a unique ID based on the PDF content
def generate_pdf_id(pdf_path):
    with open(pdf_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash


# Function to generate MCQs based on difficulty level and content
def generate_mcqs(pdf_content, difficulty_level, num_questions=10, temp_file_path=None):
    # Define the system prompt template
    template = """
    You are an MCQ Generation AI, specialized in creating multiple-choice questions (MCQs) from provided PDF documents.
    Your primary function is to generate relevant, clear, and accurate MCQs based exclusively on the content of the PDF provided by the user.
    Your role is strictly confined to the generation of MCQs, categorized into three levels of difficulty: Easy, Medium, and Hard.
    Any information or questions generated must strictly adhere to the content within the provided document.

    Context: {context}
    Human: Generate {num_questions} MCQs at {difficulty_level} difficulty based on the given PDF content.
    AI:
    """
    # Prepare the prompt template
    prompt = PromptTemplate(input_variables=["context", "difficulty_level", "num_questions"], template=template)

    # Embeddings and FAISS setup
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store from the PDF content
    faiss_index = FAISS.from_documents(documents=pdf_content, embedding=embeddings)

    retriever = faiss_index.as_retriever()
    query = f'Provide diverse MCQs from the entire document.'

    context_from_pdf = retriever.invoke(query)
    input_dict = {"context": context_from_pdf, "difficulty_level": difficulty_level, "num_questions": num_questions}

    formatted_prompt = prompt.format(context=context_from_pdf, difficulty_level=difficulty_level,
                                     num_questions=num_questions)

    # LLM setup
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 3000, "repetition_penalty": 1.1, "return_full_text": False}
    )

    mcq_result = llm.invoke(formatted_prompt)
    return mcq_result


# Function to paginate MCQs
def paginate_mcqs(mcqs, page, questions_per_page=5):
    mcq_list = mcqs.split("\n")
    start = page * questions_per_page
    end = start + questions_per_page
    return mcq_list[start:end], len(mcq_list)


# Function to export MCQs as PDF
def convert_to_pdf(mcqs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in mcqs.split("\n"):
        pdf.multi_cell(0, 10, txt=line)  # Use multi_cell for better formatting
    return pdf.output(dest='S').encode('latin1')


# Streamlit GUI setup
st.title("🤖 AI MCQ Generation from PDF 📚")
huggingface_token = st.text_input("🔑 Enter your HuggingFace API token:", type="password")
if huggingface_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token

uploaded_pdf = st.file_uploader("📥 Upload your PDF", type="pdf")
difficulty_level = st.selectbox("🎯 Select difficulty level:", ["Easy", "Medium", "Hard"])
num_questions = st.slider("⚖️ Select the number of questions:", min_value=5, max_value=30, step=2)

# Define questions_per_page
questions_per_page = 5  # You can adjust this number as needed

# Initialize session state for pagination
if "page" not in st.session_state:
    st.session_state.page = 0

# Progress bar and Generate button
if st.button("✨ Generate MCQs") and uploaded_pdf:
    with st.spinner("Processing and generating MCQs..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_pdf.read())
                temp_file_path = temp_file.name

            # Load the PDF content
            pdf_loader = PyPDFLoader(temp_file_path)
            data = pdf_loader.load()

            # Split the document into chunks for processing
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)

            # Generate MCQs
            mcqs = generate_mcqs(chunks, difficulty_level, num_questions, temp_file_path=temp_file_path)

            # Display the generated MCQs with pagination
            mcq_page, total_questions = paginate_mcqs(mcqs, st.session_state.page, questions_per_page)
            st.subheader(f"📜 Generated {difficulty_level} MCQs:")
            for question in mcq_page:
                st.write(question)

            # Add navigation for pagination
            if st.session_state.page > 0:
                if st.button("⬅️ Previous"):
                    st.session_state.page -= 1

            if (st.session_state.page + 1) * questions_per_page < total_questions:
                if st.button("➡️ Next"):
                    st.session_state.page += 1

            # Export MCQs options
            st.subheader("💾 Export MCQs:")
            if mcqs.strip():  # Ensure mcqs is not empty
                pdf_data = convert_to_pdf(mcqs)
                st.download_button("📄 Download MCQs as PDF", data=pdf_data, file_name="mcqs.pdf",
                                   mime="application/pdf")
            else:
                st.warning("❌ No MCQs generated to download.")

        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
else:
    if not uploaded_pdf:
        st.warning("⚠️ Please upload a PDF file to proceed.")
