import os 
import pandas as pd

from typing import TypedDict, List 
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph
from agent_document import AgentDocument
from agent_internet import AgentInternet 
from langchain_core.prompts import PromptTemplate 

from tools.tool_visualization import visualize_last_30_days, visualize_last_12_months

class ManagerState(TypedDict):
    question: str 
    retrived_docs: str 
    internet_results: str
    answer: str

class Manager: 

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
        return graph.compile() 

    def answer_node(self, state: ManagerState) -> ManagerState:
        prompt_template = """Você é um agente especialista em saúde pública e síndromes respiratórias agudas graves (SRAG).
                Use as informações recuperadas da base de dados (RAG) e da Internet para responder à pergunta do usuário
                de forma completa e precisa.
                
                Você deve comentar e analisar as seguintes métricas disponíveis na base de dados: 
                - taxa de aumento de casos
                - taxa de mortalidade
                - taxa de ocupação de UTI
                - taxa de vacinação da população

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

                1. Primeiro, um arquivo Markdown bem formatado contendo:
                - título
                - análise
                - tabelas se necessário
                - recomendações clínicas

              Gere um arquivo Markdown detalhado como resposta final à pergunta do usuário.
        """    
        prompt = PromptTemplate(
            input_variables=["question", "retrived_docs"],
            template=prompt_template
        )
        prompt_filled = prompt.format(
            question=state["question"],
            retrived_docs="\n".join(state["retrived_docs"]),
            internet_results=state.get("internet_results","")
        )
        response = self.llm.invoke(prompt_filled)
        return {**state,"answer": response.content}  
    
    def run_agent(self, question: str) -> str:
        initial_state: ManagerState = {
            "question": question,
            "retrived_docs": "",
            "internet_results": "",
            "answer": ""
        }
        final_state = self.graph.invoke(initial_state)     
         
        df = pd.read_csv("/home/euler/projeto_srag_agents/datalake/gold/srag_2025_final_processed.csv", sep=";")
        
        output_directory = "/home/euler/projeto_srag_agents/output/"
        file_name_30_days = "visualization_last_30_days.png"
        file_name_12_months = "visualization_last_12_months.png"

        os.makedirs(output_directory, exist_ok=True)
       
 
        full_path_30_days = os.path.join(output_directory, file_name_30_days)
        full_path_12_months = os.path.join(output_directory, file_name_12_months)

        visualize_last_30_days(df,  full_path_30_days) 
        visualize_last_12_months(df, full_path_12_months)
        print(f"Gráficos salvo com sucesso")
        
        with open("/home/euler/projeto_srag_agents/output/Output.md", "w") as f:
            f.write(str(final_state["answer"]))
        return final_state["answer"]