import os  
import logging
import json
import pandas as pd
import sys

from typing import TypedDict, List 
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, END
from agent_document import AgentDocument
from agent_internet import AgentInternet 
from langchain_core.prompts import PromptTemplate 

from tools.tool_visualization import visualize_last_30_days, visualize_last_12_months
from tools.metrics_calculator import MetricsCalculator
from tools.unify_repo_tool import UnifyRepository

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from logger_config import setup_logger

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
        setup_logger()
        logging.info("[GOVERNANÇA] Inicializando Gerente de Agentes SRAG...")
        self.agent_document = AgentDocument(path)
        self.agent_internet = AgentInternet(3) 
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5) 
        self.graph = self.build_graph()   

    def valuation_input(self, state: ManagerState) -> ManagerState:
        question = state["question"].lower()
        logging.info(f"[GOVERNANÇA - ENTRADA] Iniciando validação por Guardrail da pergunta: '{question}'")
        
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
        
        logging.info(f"[GOVERNANÇA - AUDITORIA ENTRADA] Guardrail avaliou a pergunta como: {'APROVADA (Válida)' if is_valid else 'BLOQUEADA (Inválida)'}")
        
        return {**state, "is_valid_input": is_valid}

    def valuation_output(self, state: ManagerState) -> ManagerState: 
        """Avalia se a resposta gerada pelo agente está dentro do escopo de saúde pública e Síndrome Respiratória Aguda Grave (SRAG)."""
        answer = state.get("answer", "").lower()
        logging.info("[GOVERNANÇA - SAÍDA] Iniciando validação por Guardrail da resposta gerada.")
        
        prompt_guardail = f"""Você é o auditor de um sistema especializado em Síndrome Respiratória Aguda Grave (SRAG) e dados epidemiológicos.
        
        Sua tarefa é avaliar se a resposta gerada pelo agente está dentro do escopo de saúde pública e Síndrome Respiratória Aguda Grave (SRAG).
        
        REGRAS IMPORTANTES:
        1. Se a resposta for sobre saúde, doenças e vacinas que sejam relevantes para o escopo de Síndrome Respiratória Aguda Grave (SRAG), é VÁLIDA.
        2. Se a resposta for genérica sobre estatísticas, dados, gráficos, tabelas ou taxas (ex: "mostre os dados", "qual o percentual?"), considere VÁLIDA, pois assumimos que o usuário está se referindo à base de dados médica do sistema.
        3. Só bloqueie se for CLARAMENTE sobre um assunto aleatório (esportes, culinária, entretenimento, etc).
        4. Verifique também se a resposta contém informações incorretas ou enganosas sobre saúde pública e Síndrome Respiratória Aguda Grave (SRAG). Se houver informações incorretas, considere a resposta INVÁLIDA.

        Resposta do agente: "{answer}"
        
        Responda APENAS com a palavra SIM ou NAO.
        """
        llm_guardrail = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        response = llm_guardrail.invoke(prompt_guardail).content.strip().upper()
        
        is_valid = "SIM" in response   
        
        logging.info(f"[GOVERNANÇA - AUDITORIA SAÍDA] Guardrail avaliou a resposta como: {'APROVADA (Válida)' if is_valid else 'BLOQUEADA (Inválida)'}")
        
        return {**state, "is_valid_output": is_valid}
   
    def reject_node(self, state: ManagerState) -> ManagerState:
        """Gera uma resposta padrão para perguntas fora do escopo.""" 

        reject_message = "Desculpe, meu escopo de atuação é limitado a responder perguntas sobre Síndromes Respiratórias Agudas Graves (SRAG)"
        logging.warning("[GOVERNANÇA - BLOQUEIO] Pergunta rejeitada pelo guardrail de integridade de escopo.")
        
        return {**state, "answer": reject_message}

    def metrics_node(self, state: ManagerState) -> ManagerState:
        """Calcula as métricas epidemiológicas a partir do datalake e salva no estado."""
        df_path = os.path.join(DATALAKE_DIR, "gold", "srag_2025_final_processed.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_json = os.path.join(OUTPUT_DIR, "metrics_2025.json")

        logging.info(f"[GOVERNANÇA - MÉTRICAS] Calculando estatísticas epidemiológicas a partir de {df_path}")
        MetricsCalculator(df_path, output_json)

        try:
            with open(output_json, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            metrics_str = json.dumps(metrics_data, indent=2, ensure_ascii=False)
            logging.info(f"[GOVERNANÇA - MÉTRICAS] Métricas epidemiológicas salvas com sucesso em JSON: {output_json}")
        except Exception as e:
            logging.error(f"[GOVERNANÇA - MÉTRICAS] Erro ao carregar arquivo de métricas: {e}")
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

        logging.info(f"[GOVERNANÇA - GRÁFICOS] Gráficos de visualização (30 dias e 12 meses) gerados em {OUTPUT_DIR}")

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

        logging.info(f"[GOVERNANÇA - ARTEFATOS] Resposta do agente gravada em {output_md}")

        unified_md = os.path.join(OUTPUT_DIR, "unified_repo.md")
        UnifyRepository(output_md, output_json, file_30, file_12, output_md, output_json, unified_md)

        logging.info(f"[GOVERNANÇA - ARTEFATOS] Repositório final unificado gerado com sucesso em {unified_md}")

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

        logging.info("[GOVERNANÇA] Grafo LangGraph compilado com sucesso com 9 nós orquestrados.")
        return graph.compile()
  
    def answer_node(self, state: ManagerState) -> ManagerState:
        prompt_template = """Você é um agente especialista em Síndromes Respiratórias Agudas Graves (SRAG).
                Use as informações recuperadas da base de dados (RAG) e da Internet para responder à pergunta do usuário de forma completa, precisa e clinicamente fundamentada.
                
                No bloco 'Informações das métricas', você receberá um JSON contendo EXATAMENTE as seguintes 10 métricas epidemiológicas calculadas a partir da base oficial:
                1. Proporção de casos notificados (Variação percentual do aumento de casos nos últimos 30 dias vs 30 dias anteriores)
                2. uso antiviral (Percentual de utilização de tratamento antiviral)
                3. casos de contato com aves e suinos (Percentual de pacientes com exposição a aves ou suínos)
                4. casos de febre (Percentual de pacientes que apresentaram febre)
                5. casos de dispneia (Percentual de pacientes que apresentaram dispneia / dificuldade respiratória)
                6. Proporção de pessoas em fator de risco (Percentual de pacientes com fatores de risco / comorbidades)
                7. Proporção de pessoas vacinadas (Percentual de pacientes com registro de vacinação)
                8. Proporção de pacientes na UTI (Percentual de internações em UTI)
                9. evolução da doença (Percentual de mortalidade / óbitos entre os casos concluídos)
                10. casos de surto sg (Percentual de casos vinculados a surtos de Síndrome Gripal)
                
                REGRAS DE COERÊNCIA OBRIGATÓRIAS:
                - Você DEVE utilizar os valores EXATOS numéricos fornecidos no JSON de métricas para comentar cada uma dessas 10 métricas.
                - NÃO invente, extrapole ou cite métricas que não estejam presentes no JSON fornecido.
                - Explique o significado clínico, riscos epidemiológicos e recomendações de saúde pública para cada valor apresentado.
                - Forneça fontes e links de referência confiáveis sempre que possível.

                ================================
                Informações recuperadas do banco de dados (RAG):
                {retrived_docs}

                Informações das métricas (JSON Oficial): 
                {metrics}

                ================================
                Informações recuperadas da Internet:
                {internet_results}

                Pergunta do usuário:
                {question}

                ================================
                FORMATO FINAL OBRIGATÓRIO DE RESPOSTA (MARKDOWN)
                ================================

                Gere um relatório em Markdown bem estruturado contendo:
                - Título descritivo
                - Análise Epidemiológica Detalhada (discutindo as 10 métricas com seus valores numéricos exatos)
                - Tabela Resumo das Métricas (colunas: Métrica | Valor Exato | Comentário Clínico)
                - Recomendações de Saúde Pública e Manejo Clínico
                - Fontes e Referências Confiáveis
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

        logging.info("[GOVERNANÇA - SÍNTESE] Invocando LLM (gpt-4o-mini) para sintetizar resposta final com dados do RAG, Métricas e Internet.")
        response = self.llm.invoke(prompt_filled)

        logging.info(f"[GOVERNANÇA - SÍNTESE] Resposta gerada com sucesso ({len(response.content)} caracteres).")

        return {**state, "answer": response.content}  
    
    def run_agent(self, question: str) -> str:
        logging.info("================================================================================")
        logging.info(f"[GOVERNANÇA - EXECUÇÃO] Nova pergunta enviada ao sistema: '{question}'")
        logging.info("================================================================================")
        
        initial_state: ManagerState = {
            "question": question,
            "retrived_docs": "",
            "internet_results": "",
            "metrics": "",
            "answer": "",
        }
        
        final_state = self.graph.invoke(initial_state)
        logging.info("[GOVERNANÇA - FIM] Execução do agente finalizada.")
        return final_state.get("answer", "")