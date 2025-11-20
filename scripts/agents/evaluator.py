import logging
import os
from dotenv import load_dotenv

from langsmith import Client, traceable
from langsmith.evaluation import RunEvaluator
from langchain_core.prompts import PromptTemplate

from agent_manager import Manager 
from qa_evaluator import QAEvalUniversal

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "default")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

client = Client()


@traceable(name="srag_covid19_brazil_2024_evaluation_agent")
def run_eval_agent(question: str) -> str:
    """Executa o agente principal e retorna a resposta final."""
    manager = Manager(path="/home/euler/projeto_srag_agents/datalake/gold/*.csv")
    answer = manager.run_agent(question)
    logging.info("Manager agent executed successfully.")
    return answer

qa_prompt = """
Você é um avaliador especializado em saúde pública e SRAG.

Compare a resposta do modelo com a resposta correta.

Avalie:
- precisão factual
- clareza
- coerência clínica
- uso correto das métricas (taxas, valores)
- completude da explicação

Retorne "CORRECT" ou "INCORRECT" e explique o motivo.

Pergunta: {input}
Resposta do modelo: {prediction}
Resposta de referência: {reference}
"""

prompt_template = PromptTemplate(
    input_variables=["input", "reference", "prediction"],
    template=qa_prompt,
)

evaluator = QAEvalUniversal(prompt_template)

dataset_name = "srag brazil"

result = client.evaluate(
    run_eval_agent,
    data=dataset_name,
    evaluators=[evaluator],
    experiment_prefix="srag_covid19_brazil_2024_evaluation",
    description="Universal evaluation pipeline for SRAG COVID-19 2024",
)

logging.info("Evaluation completed successfully.")
print("Avaliação concluída!")