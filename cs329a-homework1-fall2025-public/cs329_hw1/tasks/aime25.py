import os
from datasets import load_dataset
import random


class AIME25:
    """
    A class for the AIME25 dataset from Hugging Face.
    The dataset contains 30 challenging mathematical problems from the American Invitational Mathematics Examination.
    """

    def __init__(
        self, split: str = "test", hf_token: str = os.getenv("HUGGINGFACE_HUB_TOKEN")
    ):
        """
        Initialize the AIME25 dataset.

        Args:
            split (str): Dataset split to use. Defaults to "test".
        """
        self.dataset = load_dataset("math-ai/aime25", split=split, token=hf_token)
        self.rng = random.Random(42)

    def get_problems(self, debug_mode: bool = False) -> list[dict]:
        """
        Returns a list of all problems in the dataset.
        Each problem is a dictionary containing 'problem', 'answer', and 'id' keys.
        Problems are randomly shuffled with a fixed seed for reproducibility.

        Args:
            debug_mode (bool): If True, returns only first 5 problems for debugging.
                              If False, returns all 30 problems.

        Returns:
            list[dict]: List of problem dictionaries
        """
        problems = []
        for item in self.dataset:
            problems.append(
                {"problem": item["problem"], "answer": item["answer"], "id": item["id"]}
            )

        self.rng.shuffle(problems)

        if debug_mode:
            return problems[:5]  # Return first 5 for debugging
        else:
            return problems  # Return all 30 problems

    def get_system_prompt(self) -> str:
        """
        Returns a system prompt suitable for AIME-level mathematical problems.
        AIME problems are more challenging and require more sophisticated reasoning.
        """
        return """You are an expert mathematician specializing in competition mathematics. 
Solve the given AIME (American Invitational Mathematics Examination) problem step by step. 
These problems require advanced mathematical reasoning and creative problem-solving techniques.

Guidelines:
1. Think carefully about the problem structure and what is being asked
2. Use appropriate mathematical techniques (algebra, geometry, number theory, combinatorics, etc.)
3. Show all your work clearly
4. Provide a complete solution with proper reasoning
5. Express your final answer as requested in the problem

Format your response as:
Problem: [restate the problem]
Solution: [step-by-step solution]
Final Answer: [your answer]"""
