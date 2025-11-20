from agent_manager import Manager

question = """Existem casos de internação em UTI por causa de vírus sincicial respiratório (VSR) ?"""

manager = Manager(path="/home/euler/projeto_srag_agents/datalake/gold/*.csv")
resposta = manager.run_agent(question)
print(resposta)