import logging
import os
from dotenv import load_dotenv

from langsmith import Client, traceable
from langsmith.evaluation import RunEvaluator
from langchain_core.prompts import PromptTemplate

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "default")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

class QAEvalUniversal(RunEvaluator):
    """ Evaluator manual QA, building for test the agent. """

    def __init__(self, prompt):
        self.prompt = prompt

    def evaluate(self, run, example):

        input_text = example.inputs["input"]
        reference = example.outputs["output"]
        prediction = run.outputs["output"]

        formatted_prompt = self.prompt.format(
            input=input_text,
            reference=reference,
            prediction=prediction,
        ) 
        
        from openai import OpenAI
        llm = OpenAI()

        judge_output = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": formatted_prompt}]
        ).choices[0].message.content

        return {"score": judge_output}
