from langchain_community.tools.tavily_search import TavilySearchResults  
from typing import TypedDict, List 
from dotenv import load_dotenv 
import os

class AgentInternet(): 
    def __init__(self, max_results: int = 3):  
        load_dotenv()
        os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
        self.tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), max_results=self.max_results) 

    def fetch_information(self, state: dict) -> dict: 
        results = self.tool.invoke({"query": state["question"]})   
        if isinstance(results, dict):
            formatted = "\n".join([r.get("content", "") for r in results.get("results", [])])
        else:
            formatted = str(results)
        return formatted
