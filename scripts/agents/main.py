from agent_manager import Manager

question = """Ainda existe casos graves de COVID-19 no Brasil em 2024 ?
        ?"""

manager = Manager(path="/home/euler/projeto_srag_agents/datalake/gold/*.csv")
resposta = manager.run_agent(question)
print(resposta)