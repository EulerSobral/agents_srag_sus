import os
from agent_manager import Manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

question = input("Digite a pergunta: ")

manager = Manager(path="srag_2025_final_processed.csv")
resposta = manager.run_agent(question)
print(resposta)
