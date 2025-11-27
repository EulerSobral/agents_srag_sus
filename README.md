# Contexto   

Aplicação de uma prova de conceito para avaliar a viabilidade de um sistema que auxlia profissionais da área de saúde a ter um entendimento em tepo real sobre a severidade de surto de doenças.

# Tutorial para executar o projeto 

(Clique aqui)[https://www.youtube.com/watch?v=dRFvq--Vbew] para acessar o tutorial de execução do projeto

# Como rodar o Agente de SRAG 

Antes de executar o sistema, obtenha o valor as seguintes variáveis de ambiente para continuar com a execução do sistema: 

OPENAI_API_KEY= 

TAVILY_API_KEY= 

LANGCHAIN_TRACING_V2="true"

LANGCHAIN_PROJECT="default"

LANGCHAIN_API_KEY=""  

LANGHAIN_ENDPOINT="" 

Quando você obter essas variáveis, inclua elas o valor delas no arquivo example.env  

## Requisitos 

Python 3.9 ou versão mais atualizada  

Git instalado em sua máquina

## Clonando o repositório 

```  
git clone https://github.com/EulerSobral/agents_srag_sus.git
```
## Instalando dependências 

```  
pip install -r requirements/requirements.txt
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
