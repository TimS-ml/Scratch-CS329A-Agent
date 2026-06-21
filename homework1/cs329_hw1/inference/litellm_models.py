import os
import threading
import time
from typing import List, Dict, Union
from litellm import completion
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ._llm_cache import cached

# Provider keys litellm understands; any one of them is enough to make calls.
_PROVIDER_KEYS = (
    "ANTHROPIC_API_KEY",
    "TOGETHER_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
)


class LiteLLMModel:
    """
    A class to send multiple requests to a specified model concurrently using threading.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_workers: int = 256,
    ):
        """
        Initializes the LiteLLMModel with the specified model.

        Args:
            model (str): The name of the model to send requests to.
            system_prompt (str): The system prompt to use for the model.
            temperature (float): The temperature to use for the model.
            max_tokens (int): The maximum number of tokens to use for the model.
            max_workers (int): The maximum number of concurrent requests to the model.

        Raises:
            ValueError: If no supported provider API key is set.
        """
        if not any(os.getenv(k) for k in _PROVIDER_KEYS):
            raise ValueError(
                "No LLM provider API key found. Set one of: "
                + ", ".join(_PROVIDER_KEYS)
            )

        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        self.max_workers = max_workers

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def _raw_completion(self, messages: List[Dict[str, str]]) -> str:
        """Make a raw completion request with retry logic (no caching)."""
        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response["choices"][0]["message"]["content"]

    def _make_completion_request(
        self, messages: List[Dict[str, str]], occ: int = 0
    ) -> str:
        """
        Makes a completion request, served from the disk cache when possible.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            occ (int): Occurrence index that distinguishes identical prompts in a
                sampling batch so temperature>0 samples stay diverse + reproducible.

        Returns:
            str: The response from the model.
        """
        payload = {
            "provider": "litellm",
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "occ": occ,
        }
        return cached(payload, lambda: self._raw_completion(messages), tag="hw1")

    def send_request(self, prompt: str, occ: int = 0) -> str:
        """
        Sends a single request to the model and returns the response.

        Args:
            prompt (str): The prompt to send to the model.
            occ (int): Occurrence index (see ``_make_completion_request``).

        Returns:
            str: The response from the model or an error message.
        """
        messages = [{"content": prompt, "role": "user"}]
        if self.system_prompt:
            messages.insert(0, {"content": self.system_prompt, "role": "system"})
        try:
            return self._make_completion_request(messages, occ=occ)
        except Exception as e:
            import traceback

            print(f"Error in send_request: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return f"Error: {str(e)}"

    def send_requests(self, prompts: List[str]) -> List[str]:
        """
        Sends multiple requests to the model concurrently and returns the list of responses.
        Uses a thread pool to limit concurrent requests.
        """
        responses = [None] * len(prompts)
        # Occurrence index among identical prompts -> diversity-safe caching for
        # temperature>0 sampling (each of the N identical samples is cached apart).
        seen: Dict[str, int] = {}
        occs: List[int] = []
        for p in prompts:
            c = seen.get(p, 0)
            occs.append(c)
            seen[p] = c + 1
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, prompt in enumerate(prompts):
                # Add small delay between request submissions to avoid rate limits
                if i > 0:
                    time.sleep(0.12)  # ~8.3 requests per second max
                future = executor.submit(self.send_request, prompt, occs[i])
                futures.append((i, future))

            with tqdm(total=len(prompts), desc="Processing requests") as progress_bar:
                for i, future in futures:
                    responses[i] = future.result()
                    progress_bar.update(1)

        return responses

    def __call__(self, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Allows the instance to be called as a function to send prompts.

        Args:
            prompts (str or List[str]): A prompt or a list of prompts to send to the model.

        Returns:
            str or List[str]: The response(s) from the model.
        """
        if isinstance(prompts, str):
            return self.send_request(prompts)
        elif isinstance(prompts, list):
            if not all(isinstance(p, str) for p in prompts):
                raise ValueError("All prompts must be strings.")
            return self.send_requests(prompts)
        else:
            raise TypeError("prompts must be a string or a list of strings.")
