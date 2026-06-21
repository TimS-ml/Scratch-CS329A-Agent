import json
import re
from pydantic import BaseModel
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from cs329a_hw3.api_manager import APIManager
from cs329a_hw3.utils import generate_together

class SubQuery(BaseModel):
    api: str
    params: Dict
    order: int

class DecompositionResponse(BaseModel):
    sub_queries: List[SubQuery]

class MultiLMAgent:
    """A class to manage multiple language models for generation, iterative refinement, and fusion"""
    
    def __init__(
        self,
        api_manager: APIManager,
        decomposition_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        iterative_refinement_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        fusion_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        generation_temp: float = 0.7,
    ):
        """Initializes the MultiLMAgent."""
        ################ CODE STARTS HERE ###############
        self.api_manager = api_manager
        self.decomposition_model = decomposition_model
        self.iterative_refinement_model = iterative_refinement_model
        self.fusion_model = fusion_model
        self.generation_temp = generation_temp
        ################ CODE ENDS HERE ###############

    def generate(
        self, 
        query: str, 
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature: float = 0.7
    ) -> str:
        """Generates a response to the query using the specified model."""
        ################ CODE STARTS HERE ###############
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question directly and concisely."},
            {"role": "user", "content": query},
        ]
        response_message = generate_together(model=model, messages=messages, temperature=temperature)
        return response_message.content if response_message else ""
        ################ CODE ENDS HERE ###############
    
    def single_LM_with_single_API_call(self, query: str, model: str) -> str:
        """Generates a response by querying the API manager for the necessary data and then using an LM to generate a final response based on the API output."""
        ################ CODE STARTS HERE ###############
        routed = self.api_manager.route_query(query, model=model)
        api_used = routed.get("api_used")
        context = json.dumps(routed.get("results"), default=str)[:3000]
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant. Use the provided API result to answer the user's "
                "question directly and concisely. Perform any final reasoning or arithmetic "
                "yourself. If the API result is missing or irrelevant, answer as best you can."
            )},
            {"role": "user", "content": f"Question: {query}\n\nTool used: {api_used}\nTool result:\n{context}\n\nAnswer:"},
        ]
        response_message = generate_together(model=model, messages=messages, temperature=0.2)
        return response_message.content if response_message else ""
        ################ CODE ENDS HERE ###############

    def _get_query_decomposition_prompt(self, query: str) -> str:
        """Helper function to generate a prompt for the LLM to decompose the query into sub-queries."""
        ################ CODE STARTS HERE ###############
        return (
            "You are a planning module for an agent with access to these tools:\n"
            "  - google_search: web facts, recent events, general knowledge\n"
            "  - get_stock_data: stock open/high/low/close/volume for a ticker on a date\n"
            "  - compute: mathematical computation via Wolfram Alpha\n"
            "  - get_weather: historical weather for a location/date/hour\n\n"
            "Break the user's question into the smallest set of INDEPENDENT sub-questions, "
            "each answerable by exactly ONE tool call. If the question already needs only one "
            "tool, output a single sub-query. Each sub-question must be natural language and "
            "self-contained (include any ticker, location, date, etc.).\n"
            "Wrap each sub-question in <sub-query>...</sub-query> tags and output nothing else.\n\n"
            f"Question: {query}"
        )
        ################ CODE ENDS HERE ###############

    def _get_sub_queries(self, query: str, max_sub_queries: int = 4) -> List[str]:
        """Helper function to break down a complex user query into smaller subqueries."""
        ################ CODE STARTS HERE ###############
        prompt = self._get_query_decomposition_prompt(query)
        response_message = generate_together(
            model=self.decomposition_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response_message.content if response_message else ""
        subs = [s.strip() for s in re.findall(r"<sub-query>(.*?)</sub-query>", text, flags=re.DOTALL)]
        subs = [s for s in subs if s]
        if not subs:
            subs = [query]
        return subs[:max_sub_queries]
        ################ CODE ENDS HERE ###############
    
    def decompose_query(self, query: str, max_sub_queries: int = 4) -> List[Dict[str, Any]]:
        """Decomposes a query into independent sub-queries executed in parallel via the API manager."""
        ################ CODE STARTS HERE ################
        sub_queries = self._get_sub_queries(query, max_sub_queries)

        def _run(sq: str) -> Dict[str, Any]:
            routed = self.api_manager.route_query(sq)
            return {
                "sub_query": sq,
                "api": routed.get("api_used"),
                "params": routed.get("params"),
                "results": routed.get("results"),
                "error": routed.get("error"),
            }

        with ThreadPoolExecutor(max_workers=max(1, min(4, len(sub_queries)))) as ex:
            return list(ex.map(_run, sub_queries))
        ################ CODE ENDS HERE ###############

    def _get_synthesis_prompt(self, query: str, decomposed_queries: List[Dict[str, Any]] = None) -> str:
        """Constructs a prompt for an LLM to synthesize a final response using the sub-query API results."""
        ################ CODE STARTS HERE ###############
        blocks = []
        for i, dq in enumerate(decomposed_queries or [], 1):
            blocks.append(
                f"Sub-question {i}: {dq.get('sub_query')}\n"
                f"Tool: {dq.get('api')}\n"
                f"Result: {json.dumps(dq.get('results'), default=str)[:1500]}"
            )
        context = "\n\n".join(blocks) if blocks else "(no sub-query results available)"
        return (
            "You are a helpful assistant. Using ONLY the tool results below, answer the user's "
            "question directly and concisely. Perform any final reasoning or arithmetic yourself "
            "(e.g. percent changes, comparisons). If the results are insufficient, give your best "
            "answer.\n\n"
            f"User question: {query}\n\n"
            f"Tool results:\n{context}\n\n"
            "Final answer:"
        )
        ################ CODE ENDS HERE ###############

    def decompose_and_fuse(self, query: str) -> str:
        """Decomposes a query, runs the tools, samples several models on the synthesis prompt, and fuses them."""
        ################ CODE STARTS HERE ###############
        decomposed = self.decompose_query(query)
        synthesis_prompt = self._get_synthesis_prompt(query, decomposed)

        # Sample the synthesis prompt across three diverse models (mapped to
        # gpt-4.1-nano / claude-haiku / gemini-2.5-flash in utils.MODEL_MAP).
        fusion_models = [
            "google/gemma-3n-E4B-it",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "OpenAI/gpt-oss-20B",
        ]

        def _sample(m: str) -> str:
            r = generate_together(
                model=m,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=self.generation_temp,
            )
            return r.content if r else ""

        with ThreadPoolExecutor(max_workers=3) as ex:
            candidates = list(ex.map(_sample, fusion_models))

        fuse_prompt = (
            "You are given a question and several candidate answers from different models. "
            "Produce a single, correct, concise final answer, resolving any disagreement in "
            "favour of the best-justified response.\n\n"
            f"Question: {query}\n\n"
            + "\n\n".join(f"Candidate {i + 1}:\n{c}" for i, c in enumerate(candidates))
            + "\n\nFinal fused answer:"
        )
        fused = generate_together(
            model=self.fusion_model,
            messages=[{"role": "user", "content": fuse_prompt}],
            temperature=0.2,
        )
        if fused and fused.content:
            return fused.content
        return next((c for c in candidates if c), "")
        ################ CODE ENDS HERE ###############
    
    def _get_iterative_refinement_prompt(self, original_query: str, history: List[Dict[str, Any]]) -> str:
        """Helper function to generate the prompt for the iterative refinement model."""
        ################ CODE STARTS HERE ###############
        if history:
            hist_lines = []
            for i, step in enumerate(history, 1):
                hist_lines.append(
                    f"Step {i} sub-query: {step.get('sub_query')}\n"
                    f"  Tool: {step.get('api')}\n"
                    f"  Result: {json.dumps(step.get('results'), default=str)[:1200]}"
                )
            hist_text = "\n".join(hist_lines)
        else:
            hist_text = "(no steps taken yet)"
        return (
            "You are an agent answering a question using these tools: google_search, "
            "get_stock_data, compute, get_weather.\n"
            "At each step, EITHER issue exactly one more natural-language sub-query to gather "
            "missing information, OR give the final answer if you already have enough.\n"
            "Respond with EXACTLY ONE of the following and nothing else:\n"
            "  <sub-query>a single self-contained, single-tool question</sub-query>\n"
            "  <final_answer>your concise final answer</final_answer>\n\n"
            f"Original question: {original_query}\n\n"
            f"History so far:\n{hist_text}\n\n"
            "Your next action:"
        )
        ################ CODE ENDS HERE ###############
        

    def iterative_refine(self, query: str, max_iterations: int = 3) -> str:
        """Sequentially issues/executes sub-queries until the model returns a final answer."""
        ################ CODE STARTS HERE ###############
        history: List[Dict[str, Any]] = []
        for _ in range(max_iterations):
            prompt = self._get_iterative_refinement_prompt(query, history)
            response_message = generate_together(
                model=self.iterative_refinement_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = response_message.content if response_message else ""

            final = re.search(r"<final_answer>(.*?)</final_answer>", text, flags=re.DOTALL)
            if final:
                return final.group(1).strip()

            sub = re.search(r"<sub-query>(.*?)</sub-query>", text, flags=re.DOTALL)
            if not sub or not sub.group(1).strip():
                # No structured action -> treat the response as the final answer.
                return text.strip()

            sub_query = sub.group(1).strip()
            routed = self.api_manager.route_query(sub_query)
            history.append({
                "sub_query": sub_query,
                "api": routed.get("api_used"),
                "params": routed.get("params"),
                "results": routed.get("results"),
                "error": routed.get("error"),
            })

        # Max iterations reached: synthesise a final answer from gathered evidence.
        synthesis_prompt = self._get_synthesis_prompt(query, history)
        response_message = generate_together(
            model=self.iterative_refinement_model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0,
        )
        return response_message.content if response_message else ""
        ################ CODE ENDS HERE ###############
        

    def run_pipeline(self, query: str) -> str:
        """Runs the full agentic pipeline for a given query."""
        ################ CODE STARTS HERE ###############
        # Iterative refinement is the strongest single strategy: it routes each
        # step through the API manager and lets the (sonnet) refinement model do
        # the final reasoning/arithmetic.
        return self.iterative_refine(query, max_iterations=4)
        ################ CODE ENDS HERE ###############
