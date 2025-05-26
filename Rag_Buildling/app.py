import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

st.title("Venky Text Book RAG")

def load_vector_store(persist_path="./chroma_db"):
    embedding_model = "text-embedding-005"
    embeddings = VertexAIEmbeddings(model_name=embedding_model)
    vector_store = Chroma(persist_directory=persist_path, embedding_function=embeddings)
    return vector_store

# if st.sidebar.button("Load Pdf from Directory"):
#     loader = DirectoryLoader("./data/sixth/science", glob="*.pdf", loader_cls=PyPDFLoader)
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_documents(docs)
#     embeddings = VertexAIEmbeddings()
#     vectordb = Chroma.from_documents(splits, embeddings, persist_directory="./streamlit_db")
#     st.session_state["vectordb"] = vectordb

query = st.text_input("Enter your query")
if query:
    vectordb = load_vector_store()
    retriever = vectordb.as_retriever()
    llm = ChatVertexAI(model_name="gemini-2.0-flash-lite-001")
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful teacher. Given the following Content from textbook as text, 
        answer the question in detail.

        Context:
        {context}

        Question:
        {question}

        Ensure if the answer is not found in context, Reply I dont know the answer.

        Answer:"""
    )
    chain = prompt | llm 
    relevant_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    answer = chain.invoke({"context": context, "question": query})
    st.subheader("Generated Answer:")
    st.write(answer.content)