import logging

from langchain_community.tools.tavily_search import TavilySearchResults  
from typing import TypedDict, List 
from dotenv import load_dotenv 
import os

class AgentInternet(): 
    
    """ 
    Agente responsável por recuperar informações da Internet usando a API Tavily.
    """
    
    def __init__(self, max_results: int = 3):  
        load_dotenv()
        os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
        self.tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), max_results=self.max_results) 

    def fetch_information(self, state: dict) -> dict: 
        results = self.tool.invoke({"query": state["question"]})   
        if isinstance(results, dict): 
            formatted = "\n".join([r.get("content", "") for r in results.get("results", [])]) 
            logging.info(f"Fetched many results from internet and formatted them: {formatted}.")
        else: 
            formatted = str(results)
            logging.info(f"Fetched one single string result from internet: {formatted}")
        return formatted
