import os
import pandas as pd
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2', 'true')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT', None)
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
print(os.getenv("LANGCHAIN_API_KEY"))


client = Client()

dataset_name = "Validando as respostas do agente sobre SRAG"

if client.has_dataset(dataset_name=dataset_name):
    logging.info(f"Dataset '{dataset_name}' already exists. Retriever...")
    dataset = client.read_dataset(dataset_name=dataset_name)
else:
    logging.info(f"Create new dataset '{dataset_name}'...")
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Perguntas e respostas para validar o desempenho do agente sobre SRAG no Brasil"
    )

try:
    client.create_examples(
        inputs=[
            {"input": "Ainda existe casos graves de COVID-19 no Brasil em 2024 ?"},
            {"input": "Me fale agora sobre problemas envolvendo gripe aviária no Brasil"},
            {"input": "Existem casos de internação em UTI por causa de vírus sincicial respiratório (VSR) ?"},
        ], 
        outputs=[
            {"output": """Sim, ainda existem casos graves de COVID-19 no Brasil em 2024, embora a maioria dos casos seja leve devido à vacinação e às medidas de saúde pública. No entanto, grupos vulneráveis, como idosos e pessoas com comorbidades, continuam a enfrentar riscos significativos. É importante manter as precauções recomendadas pelas autoridades de saúde."""},
            
            {"output": """1) Situação atual e riscos no Brasil
A. Primeiro surto em granja comercial
Em maio de 2025, o Ministério da Agricultura confirmou o primeiro foco de Influenza Aviária de Alta Patogenicidade (HPAI) em uma granja comercial no Brasil, no município de Montenegro (Rio Grande do Sul)...
(restante do texto da gripe aviária mantido aqui, use aspas triplas para ele funcionar com quebras de linha)
"""},
            
            {"output": """1) Dados e evidências de SRAG / hospitalizações por VSR
Segundo a Agência Gov / Boletim InfoGripe, há aumento de internações por SRAG atribuídas ao VSR. Agência Gov...
(restante do texto do VSR mantido aqui)
"""}
        ],
        dataset_id=dataset.id,
    )
    logging.info("Examples created successfully.")
except Exception as e: 
    logging.error(f"Error creating examples: {e}")