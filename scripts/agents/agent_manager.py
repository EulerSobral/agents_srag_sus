import os  
import logging
import json
import pandas as pd

from typing import TypedDict, List 
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, END
from agent_document import AgentDocument
from agent_internet import AgentInternet 
from langchain_core.prompts import PromptTemplate 

from tools.tool_visualization import visualize_last_30_days, visualize_last_12_months
from tools.metrics_calculator import MetricsCalculator
from tools.unify_repo_tool import UnifyRepository

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATALAKE_DIR = os.path.join(PROJECT_ROOT, "datalake")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


class ManagerState(TypedDict, total=False):
    question: str 
    retrived_docs: str 
    internet_results: str
    metrics: str
    answer: str   
    is_valid_input: bool 
    is_valid_output: bool
    chart_30_path: str
    chart_12_path: str
    unified_md_path: str


class Manager: 
    """ 
    Função de agente com o gerente do trabalho de Agente de Documentos e Agente de Internet para responder perguntas sobre doenças respiratórias.
    """

    def __init__(self, path):
        self.agent_document = AgentDocument(path)
        self.agent_internet = AgentInternet(3) 
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5) 
        self.graph = self.build_graph()   

    def valuation_input(self, state: ManagerState) -> ManagerState:
        question = state["question"].lower()
        prompt_guardail = f"""Você é o auditor de um sistema especializado em Síndrome Respiratória Aguda Grave (SRAG) e dados epidemiológicos.
        
        Sua tarefa é avaliar se a pergunta do usuário tem alguma chance de estar relacionada ao seu banco de dados ou a análises de métricas em geral.
        
        REGRAS IMPORTANTES:
        1. Se a pergunta for sobre saúde, doenças e vacinas que sejam relevantes para o escopo de Síndrome Respiratória Aguda Grave (SRAG), é VÁLIDA.
        2. Se a pergunta for genérica sobre estatísticas, dados, gráficos, tabelas ou taxas (ex: "mostre os dados", "qual o percentual?"), considere VÁLIDA, pois assumimos que o usuário está se referindo à base de dados médica do sistema.
        3. Só bloqueie se for CLARAMENTE sobre um assunto aleatório (esportes, culinária, entretenimento, etc).
        
        Exemplos de VÁLIDAS (Responda SIM):
        - "Quais os sintomas da SRAG?" (Motivo: Saúde explícita) 
        - "Essa Síndrome Respiratória Aguda Grave (SRAG) é causada por vírus ou bactérias?" (Motivo: Doenças explícita) 
        - "Essa Síndrome Respiratória Aguda Grave (SRAG) está relacionada à gripe?" (Motivo: SRAG explícita)
        - "Qual a taxa de mortalidade?" (Motivo: Pergunta de dados, relacionada à base)
        - "Quero ver os gráficos do último mês." (Motivo: Comando de visualização de dados)
        - "Resuma as informações que você tem." (Motivo: Interação normal com o agente)
        
        Exemplos de INVÁLIDAS (Responda NAO):
        - "Como fazer um bolo de chocolate?" (Motivo: Culinária)
        - "Quem ganhou o campeonato brasileiro?" (Motivo: Esportes)
        - "Escreva um poema sobre flores." (Motivo: Entretenimento aleatório)
        
        Pergunta do usuário: "{question}"
        
        Responda APENAS com a palavra SIM ou NAO.
        """
        llm_guardrail = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        response = llm_guardrail.invoke(prompt_guardail).content.strip().upper()
        
        is_valid = "SIM" in response   
        
        logging.info(f"Guardrail avaliou a pergunta como: {'Válida' if is_valid else 'Inválida'}")
        
        return {**state, "is_valid_input": is_valid}

    def valuation_output(self, state: ManagerState) -> ManagerState: 
        """Avalia se a resposta gerada pelo agente está dentro do escopo de saúde pública e  Síndrome Respiratória Aguda Grave (SRAG)."""
        answer = state.get("answer", "").lower()
        prompt_guardail = f"""Você é o auditor de um sistema especializado em Síndrome Respiratória Aguda Grave (SRAG) e dados epidemiológicos.
        
        Sua tarefa é avaliar se a resposta gerada pelo agente está dentro do escopo de saúde pública e  Síndrome Respiratória Aguda Grave (SRAG).
        
        REGRAS IMPORTANTES:
        1. Se a resposta for sobre saúde, doenças e vacinas que sejam relevantes para o escopo de Síndrome Respiratória Aguda Grave (SRAG), é VÁLIDA.
        2. Se a resposta for genérica sobre estatísticas, dados, gráficos, tabelas ou taxas (ex: "mostre os dados", "qual o percentual?"), considere VÁLIDA, pois assumimos que o usuário está se referindo à base de dados médica do sistema.
        3. Só bloqueie se for CLARAMENTE sobre um assunto aleatório (esportes, culinária, entretenimento, etc).
        4. Verifique também se a resposta contém informações incorretas ou enganosas sobre saúde pública e  Síndrome Respiratória Aguda Grave (SRAG). Se houver informações incorretas, considere a resposta INVÁLIDA.

        Resposta do agente: "{answer}"
        
        Responda APENAS com a palavra SIM ou NAO.
        """
        llm_guardrail = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        response = llm_guardrail.invoke(prompt_guardail).content.strip().upper()
        
        is_valid = "SIM" in response   
        
        logging.info(f"Guardrail avaliou a resposta como: {'Válida' if is_valid else 'Inválida'}")
        
        return {**state, "is_valid_output": is_valid}
   
    def reject_node(self, state: ManagerState) -> ManagerState:
        """Gera uma resposta padrão para perguntas fora do escopo.""" 

        reject_message = "Desculpe, meu escopo de atuação é limitado a responder perguntas sobre Síndromes Respiratórias Agudas Graves (SRAG)"
        logging.info("Pergunta rejeitada pelo guardrail.")
        
        return {**state, "answer": reject_message}

    def metrics_node(self, state: ManagerState) -> ManagerState:
        """Calcula as métricas epidemiológicas a partir do datalake e salva no estado."""
        df_path = os.path.join(DATALAKE_DIR, "gold", "srag_2025_final_processed.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_json = os.path.join(OUTPUT_DIR, "metrics_2025.json")

        MetricsCalculator(df_path, output_json)
        logging.info("Calculated metrics and saved to JSON file.")

        try:
            with open(output_json, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            metrics_str = json.dumps(metrics_data, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error loading metrics: {e}")
            metrics_str = ""

        return {**state, "metrics": metrics_str}

    def charts_node(self, state: ManagerState) -> ManagerState:
        """Gera os gráficos de visualização (30 dias e 12 meses) e salva no estado."""
        df_path = os.path.join(DATALAKE_DIR, "gold", "srag_2025_final_processed.csv")
        df = pd.read_csv(df_path, sep=";")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_30 = os.path.join(OUTPUT_DIR, "visualization_last_30_days.png")
        file_12 = os.path.join(OUTPUT_DIR, "visualization_last_12_months.png")

        visualize_last_30_days(df, file_30)
        visualize_last_12_months(df, file_12)

        logging.info("Generated visualizations for the last 30 days and last 12 months.")

        return {**state, "chart_30_path": file_30, "chart_12_path": file_12}

    def finalize_node(self, state: ManagerState) -> ManagerState:
        """Salva o arquivo Output.md final e unifica os artefatos em unified_repo.md."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_md = os.path.join(OUTPUT_DIR, "Output.md")
        output_json = os.path.join(OUTPUT_DIR, "metrics_2025.json")
        file_30 = state.get("chart_30_path", os.path.join(OUTPUT_DIR, "visualization_last_30_days.png"))
        file_12 = state.get("chart_12_path", os.path.join(OUTPUT_DIR, "visualization_last_12_months.png"))

        with open(output_md, "w", encoding="utf-8") as f:
            f.write(str(state.get("answer", "")))

        logging.info("Final answer written to Output.md file.")

        unified_md = os.path.join(OUTPUT_DIR, "unified_repo.md")
        UnifyRepository(output_md, output_json, file_30, file_12, output_md, output_json, unified_md)

        logging.info("Unified repository files into unified_repo.md.")

        return {**state, "unified_md_path": unified_md}

    def route_after_input_guardrail(self, state: ManagerState) -> str:
        """Roteia para o cálculo de métricas se a pergunta for válida, ou rejeita se inválida."""
        if state.get("is_valid_input", True):
            return "metrics"
        return "reject"  

    def route_after_output_guardrail(self, state: ManagerState) -> str:
        """Roteia para a finalização se a resposta for válida, ou rejeita se for inválida."""
        if state.get("is_valid_output", True):
            return "valid_output"
        return "reject" 
    
    def build_graph(self) -> StateGraph:   
        graph = StateGraph(ManagerState)  

        graph.add_node("guardrail", self.valuation_input)
        graph.add_node("reject", self.reject_node)
        graph.add_node("metrics", self.metrics_node)
        graph.add_node("retrieve", self.agent_document.create_new_state) 
        graph.add_node("internet", self.agent_internet.fetch_information) 
        graph.add_node("charts", self.charts_node)
        graph.add_node("answer", self.answer_node)  
        graph.add_node("valuation_output", self.valuation_output)
        graph.add_node("finalize", self.finalize_node)

        graph.set_entry_point("guardrail")

        # Roteamento do Guardrail de Entrada
        graph.add_conditional_edges(
            "guardrail", 
            self.route_after_input_guardrail, 
            {
                "metrics": "metrics", 
                "reject": "reject"      
            }
        )

        graph.add_edge("metrics", "retrieve")
        graph.add_edge("retrieve", "internet")
        graph.add_edge("internet", "charts")
        graph.add_edge("charts", "answer")
        graph.add_edge("answer", "valuation_output")

        graph.add_conditional_edges(
            "valuation_output", 
            self.route_after_output_guardrail,
            {
                "valid_output": "finalize", 
                "reject": "reject"      
            }
        )

        graph.add_edge("finalize", END)
        graph.add_edge("reject", END)

        logging.info("Manager graph built successfully with fully orchestrated nodes.")
        return graph.compile()
  
    def answer_node(self, state: ManagerState) -> ManagerState:
        prompt_template = """Você é um agente especialista em síndromes respiratórias agudas graves (SRAG).
                Use as informações recuperadas da base de dados (RAG) e da Internet para responder à pergunta do usuário
                de forma completa e precisa. Também inclua as métricas no contexto da resposta, explicando o significado de cada métrica e suas implicações para a Síndrome Respiratória Aguda Grave (SRAG).
                
                Você deve mostrar os valores, comentar e analisar as seguintes métricas disponíveis na base de dados: 
                - taxa de aumento de casos, mostre o valor da taxa de aumento de casos
                - taxa de mortalidade, mostre o valor da taxa de mortalidade
                - taxa de ocupação de UTI, , mostre o valor da taxa de UTI
                - taxa de vacinação da população, , mostre o valor da taxa de vacinação da população 
                - taxa de pessoas em grupos de risco, mostre o valor da taxa de vacinação de grupos de risco
                - taxa de pessoas com  contato com aves e suinos, mostre o valor da taxa de pessoas com contato com aves e suinos 
                - taxa de pessoas com febre, mostre o valor da taxa de pessoas com febre 
                - taxa de evolução do quado da doença, mostre o valor da taxa de evolução do quadro da doença 
                - taxa de pessoas com sintomas respiratórios, mostre o valor da taxa de pessoas com sintomas respiratórios
                - taxa de pessoas com dispneia, mostre o valor da taxa de pessoas com dispneia 
                - taxa de surtos de SG, mostre o valor da taxa de surtos de SG 
                - taxa da utilização de antivirais, mostre o valor da taxa de utilização de antivirais
                
               É necessário que você sempre inclua:
                - explicações clínicas,
                - fatores de risco,
                - recomendações de saúde pública.

                ================================
                Informações recuperadas do banco de dados:
                {retrived_docs}

                Informações das metrícas: 
                {metrics}

                ================================
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
            input_variables=["question", "retrived_docs", "internet_results", "metrics"],
            template=prompt_template
        )

        retrived_docs_input = state.get("retrived_docs", "")
        if isinstance(retrived_docs_input, list):
            retrived_docs_input = "\n".join(retrived_docs_input)

        prompt_filled = prompt.format(
            question=state["question"],
            retrived_docs=retrived_docs_input,
            internet_results=state.get("internet_results",""),
            metrics=state.get("metrics",""), 
        )

        response = self.llm.invoke(prompt_filled)

        logging.info("Generated answer using LLM based on retrieved documents and internet results and built one prompt with role and information for Manager Agent.")

        return {**state, "answer": response.content}  
    
    def run_agent(self, question: str) -> str:
        initial_state: ManagerState = {
            "question": question,
            "retrived_docs": "",
            "internet_results": "",
            "metrics": "",
            "answer": "",
        }
        
        final_state = self.graph.invoke(initial_state)
        return final_state.get("answer", "")