from typing import TypedDict, List 
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph
from agent_document import AgentDocument
from agent_internet import AgentInternet 
from langchain_core.prompts import PromptTemplate

class ManagerState(TypedDict):
    question: str 
    retrived_docs: str 
    internet_results: str
    answer: str

class Manager: 

    def __init__(self, path):
        self.agent_document = AgentDocument(path)
        self.agent_internet = AgentInternet(3) 
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5) 
        self.graph = self.build_graph() 

    def build_graph(self) -> StateGraph:  
        graph = StateGraph(ManagerState) 

        graph.add_node("retrieve", self.agent_document.create_new_state) 
        graph.add_node("internet", self.agent_internet.fetch_information)
        graph.add_node("answer", self.answer_node)  

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "answer") 
        graph.add_edge("internet", "answer")
        graph.set_finish_point("answer")
        return graph.compile() 

    def answer_node(self, state: ManagerState) -> ManagerState:
        prompt_template = """Você é um agente especialista em saúde pública e síndromes respiratórias agudas graves (SRAG).
                Use as informações recuperadas da base de dados (RAG) e da Internet para responder à pergunta do usuário
                de forma completa e precisa.

                Se a pergunta for relacionada a doenças respiratórias, sempre inclua:
                - explicações clínicas,
                - fatores de risco,
                - medidas preventivas,
                - recomendações de saúde pública.

                Se a pergunta envolver antivirais:
                - inclua dosagem, efeitos colaterais, interações medicamentosas
                - e recomendações de baixo custo.

                ==============================
                GERAÇÃO DE GRÁFICOS (SEM PYTHON)
                ==============================

                Sempre que a resposta puder ser melhorada com visualização de dados,
                gere um gráfico como IMAGEM seguindo estas regras:

                1. Crie uma descrição extremamente detalhada do gráfico:
                - tipo (barras, linhas, pizza, etc)
                - eixos e legendas
                - cores
                - valores aproximados extraídos dos dados RAG + Internet
                - estilo (clean, profissional, minimalista)

                2. NÃO gere Python. NÃO gere JSON. NÃO gere código.

                3. A descrição da imagem deve aparecer EM UM BLOCO SEPARADO:

                <graph_image_prompt>
                (descrição completa do gráfico)
                </graph_image_prompt>

                4. A resposta principal deve estar EM MARKDOWN.
                O bloco <graph_image_prompt> deve ficar fora da seção Markdown.

                ================================
                INFORMAÇÕES DISPONÍVEIS
                ================================

                Informações recuperadas do banco de dados:
                {retrived_docs}

                Informações recuperadas da Internet:
                {internet_results}

                Pergunta do usuário:
                {question}

                ================================
                FORMATO FINAL OBRIGATÓRIO
                ================================

                1. Primeiro, um arquivo Markdown bem formatado contendo:
                - título
                - análise
                - tabelas se necessário
                - recomendações clínicas
                - descrição textual do gráfico (opcional)

                2. Depois, em uma linha separada, fora do Markdown:
                O bloco <graph_image_prompt> contendo a descrição da imagem.

                Não misture o bloco <graph_image_prompt> com o Markdown.
        """    
        prompt = PromptTemplate(
            input_variables=["question", "retrived_docs"],
            template=prompt_template
        )
        prompt_filled = prompt.format(
            question=state["question"],
            retrived_docs="\n".join(state["retrived_docs"]),
            internet_results=state.get("internet_results","")
        )
        response = self.llm.invoke(prompt_filled)
        return {**state,"answer": response.content}  
    
    def run_agent(self, question: str) -> str:
        initial_state: ManagerState = {
            "question": question,
            "retrived_docs": "",
            "internet_results": "",
            "answer": ""
        }
        final_state = self.graph.invoke(initial_state)     
        
        with open("Output.md", "w") as f:
            f.write(str(final_state["answer"]))
        return final_state["answer"]

