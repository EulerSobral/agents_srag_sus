import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import Client, traceable
from langsmith.evaluation import RunEvaluator, EvaluationResult
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
        try:
            input_text = example.inputs["input"]
            reference = example.outputs["output"]
            prediction = run.outputs["output"]

            formatted_prompt = self.prompt.format(
                input=input_text,
                reference=reference,
                prediction=prediction,
            ) 
            
            llm = OpenAI()
            response = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": formatted_prompt}]
            )
            judge_output = response.choices[0].message.content

            if judge_output:
                parts = judge_output.strip().split(" ", 1)
                label = parts[0].upper()
                comment = parts[1] if len(parts) > 1 else ""
                score = 1 if label == "CORRECT" else 0
            else:
                score = 0
                comment = "No response from evaluator"

            return [EvaluationResult(key="qa_eval", score=score, comment=comment)]
        except Exception as e:
            logging.error(f"Error in evaluate: {e}")
            return [EvaluationResult(key="qa_eval", score=0, comment=f"Error: {str(e)}")]
