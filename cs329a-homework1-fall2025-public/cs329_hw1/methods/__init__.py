from .simple_samplers import GreedyMethod, SampleMultiple
from .aime25_verifier import AIME25Verifier

__all__ = ["GreedyMethod", "SampleMultiple", "AIME25Verifier"]


def get_sampler(
    method: str,
    model: str,
    system_prompt: str = None,
    n_samples: int = 3,
    temperature: float = 0.0,
):
    if method == "sample_multiple":
        assert temperature > 0.0, "Sample multiple requires temperature > 0"
        return SampleMultiple(model, system_prompt, n_samples, temperature)
    elif method == "greedy":
        assert (
            temperature == 0.0
        ), "Greedy decoding does not use temperature, it will be 0 by default"
        return GreedyMethod(model, system_prompt)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_verifier(dataset_name: str):
    """
    Get the appropriate verifier for the dataset.

    Args:
        dataset_name (str): Name of the dataset ("aime25")

    Returns:
        Verifier instance
    """
    if dataset_name.lower() == "aime25":
        return AIME25Verifier()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
