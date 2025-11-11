from agent_manager import Manager

question = """Posso usar o Oseltamivir em pacientes com síndrome  
            respiratória aguda grave (SRAG) e que apresentam fatores de risco, como hipertensão  
        ?"""

manager = Manager(path="/home/euler/projeto_srag_agents/datalake/gold/*.csv")
resposta = manager.run_agent(question)
print(resposta)