import os
from agent_manager import Manager

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))   
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  


GOLD_PATH = os.path.join(PROJECT_ROOT, "datalake", "gold", "srag_2025_final_processed.csv")

question = input("Digite a pergunta: ")

manager = Manager(path=GOLD_PATH)
resposta = manager.run_agent(question)
print(resposta)
