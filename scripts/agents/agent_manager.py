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
        Use as informações recuperadas da base de dados e da Internet para responder à pergunta do usuário de forma completa e precisa.
        Além disso caso a pergunta for relacionada a alguma doença do tipo síndrome respiratória aguda, forneça dicas para o especialista de saúde como ele pode mitigar os riscos associados à síndrome. 
        Por exemplo, se a pergunta for sobre COVID-19, inclua informações sobre medidas preventivas, tratamentos disponíveis e recomendações de saúde pública. 
        Caso a pergunta for sobre algum um antíviral, forneça informações sobre dosagem, efeitos colaterais, interações medicamentosas e opções de baixo custo para o paciente.

        Informações recuperadas do banco de dados:
        {retrived_docs}

        Informações recuperadas da Internet: 
        {internet_results} 

        Pergunta: {question}
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
        return final_state["answer"]

