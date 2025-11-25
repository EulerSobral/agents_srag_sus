# Como rodar o Agente de SRAG 

Antes de executar o sistema, obtenha as seguintes variáveis de ambiente para continuar com a execução do sistema: 

OPENAI_API_KEY= 

TAVILY_API_KEY= 

LANGCHAIN_TRACING_V2="true"

LANGCHAIN_PROJECT="default"

LANGCHAIN_API_KEY=""  

LANGHAIN_ENDPOINT="" 

Quando você obter essas variáveis, inclua elas o valor delas no arquivo .env  

## Requisitos 

Python 3.9 ou versão mais atualizada  

Git instalado em sua máquina =

## Clonando o repositório 

```  
git clone https://github.com/EulerSobral/agents_srag_sus.git
```

## Obtendo os dados 

Para obter os dados vá até  até a pasta de etl:

```
cd scripts/etl

python extract.py

python treated_data.py

python load_data.py 
```

## Rodando os agentes 

Com os dados já carregados ao datalake, você finalmente pode executar o sistema: 

```
cd scripts/agents

python main.py
```
