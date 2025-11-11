import os
import glob
import pandas as pd
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class AgentDocument():  
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    def __init__(self, path: str):  
        self.path = path
        self.retriever = self.rag_retriever()


    def rag_retriever(self):
        files = glob.glob(self.path)
        df = pd.concat(
            [pd.read_csv(arq, sep=";", on_bad_lines="skip") for arq in files],
            ignore_index=True
        )
        df = df.head(10)

        columns_from_doc = [
            "NU_IDADE_N","CS_GESTANT","AVE_SUINO","FEBRE","TOSSE","GARGANTA","DISPNEIA",
            "DESC_RESP","DIARREIA","VOMITO","FATOR_RISC","VACINA","ANTIVIRAL","TP_ANTIVIR",
            "UTI","TP_AMOSTRA","EVOLUCAO","VACINA_COV","SURTO_SG","CO_DETEC"
        ]


        docs = []
        for idx, row in df.iterrows():
            text_line = " | ".join([f"{col}: {row[col]}" for col in columns_from_doc if col in df.columns])
            docs.append(Document(page_content=f"Linha {idx}: {text_line}", metadata={"row_index": idx}))

        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(docs, embedding=embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        return retriever

    def create_new_state(self, state: dict) -> dict:
        information = self.retriever.invoke(state["question"]) 
        split_information = "\n".join([doc.page_content for doc in information])  
        state['retrived_docs'] = split_information 
        return state
