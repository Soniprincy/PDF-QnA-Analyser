# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS  
# from langchain_community.vectorstores.chroma import Chroma
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Load API key from .env
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # ✅ Store API key

# # Function to extract text from PDF
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Function to chunk text
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Function to create and save FAISS vector store
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001",
#         google_api_key=GOOGLE_API_KEY  # ✅ Provide API key
#     )
#     # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store = Chroma.from_texts(text_chunks, embedding=embeddings )
#     # vector_store.save_local("faiss_index")
#     Chroma(persist_directory="chroma_index", embedding_function=embeddings)

# # Function to set up conversational chain
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. 
#     If the answer is not in the context, say "answer is not available in the context".

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(
#         model="models/gemini-1.5-flash-latest",  # ✅ Correct model path
#         temperature=0.3,
#         google_api_key=GOOGLE_API_KEY  # ✅ Provide API key
#     )

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# # Function to process user question
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001",
#         google_api_key=GOOGLE_API_KEY  # ✅ Provide API key
#     )

#     # new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     new_db = Chroma.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True
#     )

#     print(response)
#     st.write("Reply:", response["output_text"])

# # Streamlit app main function
# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini 💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()



# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Qdrant
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import fitz

# # Load API key from .env
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Global Qdrant client
# qdrant_client = QdrantClient(url="https://your-qdrant-instance", api_key=GOOGLE_API_KEY)
# # Extract text from PDF
# # def get_pdf_text(pdf_docs):
# #     text = ""
# #     for pdf in pdf_docs:
# #         pdf_reader = PdfReader(pdf)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
# #     return text

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         doc = fitz.open(stream=pdf.read(), filetype="pdf")
#         for page in doc:
#             text += page.get_text()
#     return text

# # Split text into chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# # Create Qdrant vector store
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001",
#         google_api_key=GOOGLE_API_KEY
#     )

#     # Create/overwrite collection
#     if not qdrant_client.collection_exists("pdf_chunks"):
#         qdrant_client.create_collection(
#             collection_name="pdf_chunks",
#             vectors_config=VectorParams(size=768, distance=Distance.COSINE)
#         )

#     Qdrant.from_texts(
#         texts=text_chunks,
#         embedding=embeddings,
#         client=qdrant_client,
#         collection_name="pdf_chunks"
#     )

# # Setup Gemini QA Chain
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. 
#     If the answer is not in the context, say "answer is not available in the context".

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(
#         model="models/gemini-1.5-flash-latest",
#         temperature=0.3,
#         google_api_key=GOOGLE_API_KEY
#     )

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # Handle user question
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001",
#         google_api_key=GOOGLE_API_KEY
#     )

#     vector_store = Qdrant(
#         client=qdrant_client,
#         collection_name="pdf_chunks",
#         embedding_function=embeddings
#     )

#     docs = vector_store.similarity_search(user_question, k=5)

#     chain = get_conversational_chain()
#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True
#     )

#     st.write("Reply:", response["output_text"])


# # Streamlit UI
# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini 💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     # if user_question:
#     #     user_input(user_question)
#     user_question = st.text_input("Ask a question from the PDF files")

#     if user_question:
#         try:
#             user_input(user_question)
#         except Exception as e:
#             st.error(f"Error: {e}. Please upload and process PDFs first.")


#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()


import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Directory to save FAISS index
FAISS_INDEX_DIR = "faiss_index"

# Function to get text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to save FAISS index
def save_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)

# Function to load FAISS index
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# Function to create a QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "The answer is not available in the context", 
    do not provide a fabricated answer.

    Context:\n {context} \n
    Question: \n {question} \n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to answer user queries
def user_input(user_question):
    vectorstore = load_vector_store()
    docs = vectorstore.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    st.write("Reply:", response["output_text"])

# Streamlit UI
def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using Google Gemini 💬")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                save_vector_store(text_chunks)
                st.success("Processing complete!")

if __name__ == "__main__":
    main()
