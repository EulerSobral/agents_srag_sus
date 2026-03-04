# SRAG Agents - Sistema de Análise de Surtos em Tempo Real

## Visão Geral do Projeto

**SRAG Agents** é uma prova de conceito que implementa um sistema multi-agente com IA generativa para auxiliar profissionais da área de saúde a compreender em tempo real a severidade de surtos de **Síndrome Respiratória Aguda Grave (SRAG)**.

### Objetivo Principal
Disponibilizar análises inteligentes e baseadas em dados sobre a evolução de casos de SRAG, permitindo que profissionais de saúde tomem decisões informadas e rápidas durante crises respiratórias.

## Tutorial para executar o projeto 

[Clique aqui](https://www.youtube.com/watch?v=dRFvq--Vbew) para acessar o tutorial de execução do projeto

## Como rodar o Agente de SRAG 

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

Para obter os dados vá  na pasta de etl e execute os arquivos seguindo a sequência indicada:

```
cd scripts/etl

python extract.py

python treated_data.py

python load_data.py 
```

## Rodando os agentes 

Com os dados já carregados no datalake, você finalmente pode executar o sistema: 

```
cd scripts/agents

python main.py
```
