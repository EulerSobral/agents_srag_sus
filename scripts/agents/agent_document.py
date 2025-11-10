import os
import glob
import pandas as pd
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings  
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

path = "/home/euler/projeto_srag_agents/datalake/gold/*.csv"
files = glob.glob(path)
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

class AgentDocumentInput(TypedDict):
    question: str
    retrived_docs: List[str]
    answer: str

def retriever_node(state: dict) -> dict:
    relevant_docs = retriever.invoke(state["question"])
    retrieved_texts = [doc.page_content for doc in relevant_docs]
    return {**state, "retrived_docs": retrieved_texts}

llm =  ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

prompt_template = """Você é um agente especialista em saúde pública e síndromes respiratórias agudas graves (SRAG).
Use as informações recuperadas da base de dados para responder à pergunta do usuário de forma completa e precisa.

Informações recuperadas do banco de dados:
{retrived_docs}

Pergunta: {question}
"""

prompt = PromptTemplate(
    input_variables=["question", "retrived_docs"],
    template=prompt_template
)

def answer_node(state: dict) -> dict:
    prompt_filled = prompt.format(
        question=state["question"],
        retrived_docs="\n".join(state["retrived_docs"])
    )
    response = llm.invoke(prompt_filled)
    return {**state,"answer": response.content}

graph = StateGraph(dict)
graph.add_node("retrieve", retriever_node)
graph.add_node("answer", answer_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "answer")
graph.set_finish_point("answer")
compiled_graph = graph.compile()

def run_agent(question: str):
    state: AgentDocumentInput = {"question": question, "retrived_docs": [], "answer": ""}
    result = compiled_graph.invoke(state)
    return result["answer"]

if __name__ == "__main__":
    pergunta = "A febre está relacionada com casos de pacientes em UTI?"
    resposta = run_agent(pergunta)
    print("Resposta do agente:", resposta)
