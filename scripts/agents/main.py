import os
import sys

# Adiciona o diretório atual ao sys.path para importações relativas
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from logger_config import setup_logger
from agent_manager import Manager

# Inicializa a camada de governança e logging estruturado persistente
setup_logger()

if __name__ == "__main__":
    question = input("Digite a pergunta: ")
    manager = Manager(path="srag_2025_final_processed.csv")
    resposta = manager.run_agent(question)
    print("\n================== RESPOSTA FINAL ==================\n")
    print(resposta)
