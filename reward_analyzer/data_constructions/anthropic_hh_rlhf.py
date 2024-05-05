import torch

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers import pipeline


def setup_llama_reward_model(test_texts=None):
    rm_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")

    rm_pipe = pipeline(
        "sentiment-analysis",
        model="weqweasdas/hh_rlhf_rm_open_llama_3b",
        device="cuda",
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1
    }

    test_texts = test_texts or [
        "###Human: My daughter wants to know how to convert fractions to decimals, but I'm not sure how to explain it. Can you help? ###Assistant: Sure. So one way of converting fractions to decimals is to ask “how many halves are there?” and then write this as a decimal number. But that's a little tricky. Here's a simpler way:  if a fraction is expressed as a/b, then it's decimal equivalent is just a/b * 1.0  So, for example, the decimal equivalent of 1/2 is 1/2 * 1.0 = 0.5.",
        "###Human: I have fresh whole chicken in my fridge. What dish can I prepare using it that will take me less than an hour to cook? ###Assistant: Are you interested in a quick and easy recipe you can prepare with chicken you have on hand, or something more involved?  In terms of both effort and time, what are you looking for?",
        "###Human: My daughter wants to know how to convert fractions to decimals, but I'm not sure how to explain it. Can you help? ###Assistant: Yes, of course. Here you go."
    ]

    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    return rewards

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(split: str = 'train', sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 300)))

    def split_prompt_and_responses(sample) -> dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)