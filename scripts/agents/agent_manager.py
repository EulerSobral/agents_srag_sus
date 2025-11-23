import os  
import logging
import pandas as pd

from typing import TypedDict, List 
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph
from agent_document import AgentDocument
from agent_internet import AgentInternet 
from langchain_core.prompts import PromptTemplate 

from tools.tool_visualization import visualize_last_30_days, visualize_last_12_months


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

DATALAKE_DIR = os.path.join(PROJECT_ROOT, "projeto_srag_agents","datalake")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "projeto_srag_agents","output")


class ManagerState(TypedDict):
    question: str 
    retrived_docs: str 
    internet_results: str
    answer: str


class Manager: 
    """ 
    Agent role with manager the work of Document Agent and Internet Agent to answer questions about public respiratory diseases and medications.
    """

    def __init__(self, path):
        self.agent_document = AgentDocument(path)
        self.agent_internet = AgentInternet(3) 
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5) 
        self.graph = self.build_graph() 

    def build_graph(self) -> StateGraph:  
        graph = StateGraph(ManagerState) 

        graph.add_node("retrieve", self.agent_document.create_new_state) 
        graph.add_node("internet", self.agent_internet.fetch_information)
        graph.add_node("answer", self.answer_node)  

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "answer") 
        graph.add_edge("internet", "answer")
        graph.set_finish_point("answer") 
        
        logging.info("Manager graph built successfully with nodes: retrieve, internet, answer.")

        return graph.compile() 

    def answer_node(self, state: ManagerState) -> ManagerState:
        prompt_template = """Você é um agente especialista em saúde pública e síndromes respiratórias agudas graves (SRAG).
                Use as informações recuperadas da base de dados (RAG) e da Internet para responder à pergunta do usuário
                de forma completa e precisa.
                
                Você deve mostrar os valores, comentar e analisar as seguintes métricas disponíveis na base de dados: 
                - taxa de aumento de casos, mostre o valor da taxa de aumento de casos
                - taxa de mortalidade, mostre o valor da taxa de mortalidade
                - taxa de ocupação de UTI, , mostre o valor da taxa de UTI
                - taxa de vacinação da população, , mostre o valor da taxa de vacinação da população

                Se a pergunta for relacionada a doenças respiratórias, sempre inclua:
                - explicações clínicas,
                - fatores de risco,
                - medidas preventivas,
                - recomendações de saúde pública.

                Se a pergunta envolver antivirais:
                - inclua dosagem, efeitos colaterais, interações medicamentosas
                - e recomendações de baixo custo.

                Informações recuperadas do banco de dados:
                {retrived_docs}

                Informações recuperadas da Internet:
                {internet_results}

                Pergunta do usuário:
                {question}

                ================================
                FORMATO FINAL OBRIGATÓRIO
                ================================

                1. Primeiro, um arquivo do tipo md bem formatado contendo:
                - título
                - análise
                - tabelas se necessário
                - recomendações clínicas

              Gere um arquivo do tipo md detalhado como resposta final à pergunta do usuário.
        """    
        prompt = PromptTemplate(
            input_variables=["question", "retrived_docs", "internet_results"],
            template=prompt_template
        )

        prompt_filled = prompt.format(
            question=state["question"],
            retrived_docs="\n".join(state["retrived_docs"]),
            internet_results=state.get("internet_results","")
        )

        response = self.llm.invoke(prompt_filled)

        logging.info("Generated answer using LLM based on retrieved documents and internet results and built one prompt with role and information for Manager Agent.")

        return {**state, "answer": response.content}  
    
    def run_agent(self, question: str) -> str:
        initial_state: ManagerState = {
            "question": question,
            "retrived_docs": "",
            "internet_results": "",
            "answer": ""
        }

        final_state = self.graph.invoke(initial_state)

       
        df_path = os.path.join(DATALAKE_DIR, "gold", "srag_2025_final_processed.csv")
        df = pd.read_csv(df_path, sep=";")

    
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        file_30 = os.path.join(OUTPUT_DIR, "visualization_last_30_days.png")
        file_12 = os.path.join(OUTPUT_DIR, "visualization_last_12_months.png")

        visualize_last_30_days(df, file_30) 
        visualize_last_12_months(df, file_12)

        logging.info("Generated visualizations for the last 30 days and last 12 months.")

       
        output_md = os.path.join(OUTPUT_DIR, "Output.md")
        with open(output_md, "w") as f:
            f.write(str(final_state["answer"]))  

        logging.info("Final answer written to Output.md file.")

        return final_state["answer"]