# Sistema de Análise de Surtos de Síndrome Respiratória Aguda Grave em Tempo Real

## Visão Geral do Projeto

Esta prova de conceito que implementa um sistema multi-agente que utiliza LLM (gpt-4o-mini) e visualização de dados  para auxiliar profissionais da área de saúde a compreender em tempo real a severidade de surtos de **Síndrome Respiratória Aguda Grave (SRAG)** em pacientes.

### Objetivo Principal
Disponibilizar análises inteligentes e baseadas em dados sobre a evolução de casos de SRAG, permitindo que profissionais de saúde tomem decisões informadas e rápidas durante crises respiratórias.

## Tutorial para executar o projeto 

[Clique aqui](https://www.youtube.com/watch?v=dRFvq--Vbew) para acessar o tutorial de execução do projeto

## Processamento de Dados do Open DATASUS

### 1. Critérios para Escolha das Colunas
O dataset original do Open DATASUS contém mais de 150 colunas. Para o escopo deste projeto, foram selecionadas **11 colunas prioritárias** que possuem relação direta com a severidade dos surtos de SRAG, sintomas clínicos, fatores de risco, profilaxia e desfecho dos pacientes:

| Coluna | Descrição Epidemiológica | Finalidade no Projeto |
| :--- | :--- | :--- |
| **`DT_NOTIFIC`** | Data da notificação do caso | Base para séries temporais (30 dias, 12 meses) e variação percentual interanual. |
| **`AVE_SUINO`** | Histórico de contato com aves ou suínos | Avaliação de risco zoonótico e potencial de novos surtos. |
| **`FEBRE`** | Presença de febre | Monitoramento de sintoma primário de SRAG. |
| **`DISPNEIA`** | Presença de falta de ar (dispneia) | Monitoramento de sintoma de gravidade respiratória. |
| **`DESC_RESP`** | Desconforto respiratório | Indicador de agravamento clínico. |
| **`FATOR_RISC`** | Presença de fatores de risco/comorbidades | Mapeamento de grupos de vulnerabilidade. |
| **`VACINA`** | Vacinação contra a gripe | Avaliação de cobertura vacinal entre os notificados. |
| **`ANTIVIRAL`** | Uso de medicamento antiviral | Indicador de tratamento precoce/hospitalar. |
| **`UTI`** | Internação em Unidade de Terapia Intensiva | Indicador crítico de severidade e ocupação de leitos. |
| **`EVOLUCAO`** | Desfecho do caso (Cura vs. Óbito) | Cálculo da taxa de letalidade e recuperação. |
| **`SURTO_SG`** | Notificação proveniente de surto de Síndrome Gripal | Identificação de aglomerados de infecção em tempo real. |

---

### 2. Critérios de Limpeza e Tratamento dos Dados (Camadas Silver & Gold)

Conforme documentado no diretório `notebooks/`, o fluxo de tratamento dos dados foi estruturado em camadas de Datalake:

1. **Remoção de Duplicados e Trata Nulos (Camada Silver - `treated_data.py`):**
   * Filtragem restrita às 11 colunas de interesse.
   * Remoção de registros totalmente duplicados (`drop_duplicates()`).
   * Substituição inicial de valores nulos/ausentes por `"0"` (indica campo não preenchido na ficha de notificação).

2. **Padronização de Datas (Camada Gold - `load_data.py`):**
   * Conversão e formatação do campo `DT_NOTIFIC` para o padrão ISO `YYYY-MM-DD`, permitindo agrupamentos por janela temporal (últimos 30 dias e últimos 12 meses).

3. **Mapeamento de Códigos Numéricos para Categorias Textuais (Camada Gold - `load_data.py`):**
   * **Motivação:** No banco de dados bruto do SUS, as respostas são representadas por códigos numéricos (ex: `1` para Sim, `2` para Não, `9` para Ignorado).
   * **Justificativa para IA/LLM:** A conversão desses códigos numéricos para representações textuais completas (ex: `"Sim"`, `"Não"`, `"Ignorado"`, `"Cura"`, `"Óbito"`, `"Sim, aves e/ou suínos"`) foi aplicada para **reduzir a alucinação dos modelos generativos (LLMs)** e garantir que a IA compreenda exatamente o significado biológico/epidemiológico dos dados ao realizar as consultas via RAG.

#### Mapeamentos Aplicados:
* **Campos Binários e Sintomas (`FEBRE`, `DISPNEIA`, `DESC_RESP`, `FATOR_RISC`, `VACINA`, `ANTIVIRAL`, `UTI`, `SURTO_SG`):**
  `{0: "Campo vazio", 1: "Sim", 2: "Não", 9: "Ignorado"}`
* **Contato Zoonótico (`AVE_SUINO`):**
  `{0: "Campo vazio", 1: "Sim, aves e/ou suínos", 2: "Não, nenhum", 3: "Sim, outros", 9: "Ignorado"}`
* **Desfecho do Caso (`EVOLUCAO`):**
  `{0: "Campo vazio", 1: "Cura", 2: "Óbito", 3: "Óbito por outras causas", 9: "Ignorado"}`

---

### 3. Tratamento de Dados Sensíveis e Privacidade (LGPD)

Conforme documentado no notebook de análise exploratória (`exploratory_analysis.ipynb`), o banco de dados original do Open DATASUS contém campos com dados pessoais e identificadores individuais ou geográficos.

Para garantir a privacidade dos pacientes e a conformidade com as diretrizes da LGPD:
* **Eliminação de Identificadores Diretos e Indiretos:** Todas as variáveis com potencial de identificação pessoal foram descartadas no filtro inicial do ETL (transição da camada Bronze para a camada Silver), tais como:
  * `TEM_CPF`: Registro de CPF do paciente.
  * `NM_UN_INTE`: Nome e código da unidade hospitalar de internação.
  * `CO_MUN_RES` / `ID_MN_RESI` / `ID_RG_RESI`: Dados de geolocalização e município de residência.
  * `NU_NOTIFIC`: Código individualizador do número de notificação do paciente.
* **Garantia de Anonimização para LLMs:** As camadas Silver e Gold e a base de conhecimento RAG lidam exclusivamente com dados agregados e desidentificados (sintomas gerais, fatores de risco, evolução clínica e faixa temporal), garantindo que **nenhuma informação pessoal sensível (PII)** seja exposta ou processada pelos modelos de inteligência artificial.

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
