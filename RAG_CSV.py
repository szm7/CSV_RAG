import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from itertools import combinations

load_dotenv()

class FinancialRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama3-8b-8192",
            groq_api_key = st.secrets["groq"]["api_key"]
        )
        self.vector_store = None
        self.qa_chain = None
        
    def load_and_process_data(self, data_path):
        """
        Load financial data and prepare it for RAG
        """
        df = pd.read_csv(data_path)
        st.write(f"Loading data...")
        
        documents = []
        
        for _, row in df.iterrows():
            text_parts = []
            for column in df.columns:
                text_parts.append(f"{column}: {row[column]}")
            text = ", ".join(text_parts)
            documents.append(text.lower())
        
        relevant_columns = ['customer name','order status', 'product line','year', 'quarter','month','country']

        max_dimensions = 2
        combinations_list = []
        for i in range(1, max_dimensions + 1):
            combinations_list.extend(combinations(relevant_columns, i))
        results = {}
        for cols in combinations_list:
            group = df.groupby(list(cols)).agg({
                'sales amount': 'sum',  
                'quantity ordered': 'sum' ,
                'order number': 'count' 
            }).reset_index()

            group.rename(columns={
                'sales amount': f'total sales {" ".join(cols)}',
                'quantity ordered': f'total quantity {" ".join(cols)}',
                'order number': f'total order {" ".join(cols)}'
            }, inplace=True)
            results[cols] = group

        for cols, group_df in results.items():
            formatted_rows = []
            for _, row in group_df.iterrows():
                row_data = [f"{col.lower()}:{row[col]}" for col in group_df.columns]
                formatted_rows.append(", ".join(row_data))
            document_content = "\n".join(formatted_rows).lower()
            documents.append(document_content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=250,
            length_function=len
        )
        texts = text_splitter.create_documents(documents)
        st.write(f"Data processed successfully!")
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        
    def setup_retrieval_qa(self):
        st.write(f"Setting Model...")
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        custom_prompt = PromptTemplate(
            template = (
                "You are an assistant with access to financial data. "
                "Based on the context provided "
                "Consider only the question variables and provide a concise final answer"
                "If you don't know the answer, say you don't know. "
                "Generate a concise answer.\n\n"
                "{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
        ),
            input_variables=["context", "question"],
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt},
            verbose=True
        )
        
    def query(self, question: str) -> str:
        if not self.vector_store or not self.qa_chain:
            raise ValueError("Please load data and setup the QA chain first")
        try:
            response = self.qa_chain({"query": question})
            return response["result"]
        except Exception as e:
            return f"Error processing query: {str(e)}"

def main():
    st.title("RAG System for CSV")
    if 'rag' not in st.session_state:
        st.session_state.rag = FinancialRAG()
        st.session_state.rag.load_and_process_data("Data.csv")
        st.session_state.rag.setup_retrieval_qa()
    
    question = st.text_input("Enter your query:")
    st.write("_Tips : ._")
    st.write("_Try altering the phrasing of your query for improved results._")
    st.write("_Elaborating on details, or specifying keywords for better results._")


    if question:
        response = st.session_state.rag.query(question)
        st.write("### Response:")
        st.write(response)

if __name__ == "__main__":
    main()
