"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

from email import header
import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD) — update if pricing changes
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-4o": 0.010,
    "gpt-4o-mini": 0.0006,
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Task 1 — Call GPT-4o
# ---------------------------------------------------------------------------
from openai import OpenAI
OPENAI_MODEL = "gpt-4o"

def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API and return the response text + latency.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    """
    # TODO: import OpenAI, create client, call chat.completions.create,
    #       measure start/end time, return (response_text, latency)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [{"role": "user", "content":prompt}]
    
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    end_time = time.perf_counter()

    response_text = response.choices[0].message.content
    latency = end_time - start_time

    return response_text, latency


# ---------------------------------------------------------------------------
# Task 2 — Call GPT-4o-mini
# ---------------------------------------------------------------------------

OPENAI_MINI_MODEL = "gpt-4o-mini"

def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API using gpt-4o-mini and return the
    response text + latency.

    Args:
        prompt:      The user message to send.
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        Reuse call_openai() by passing model=OPENAI_MINI_MODEL.
    """
    # TODO: call call_openai with model=OPENAI_MINI_MODEL
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )


# ---------------------------------------------------------------------------
# Task 3 — Compare GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------

COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-4o": 0.010,
}

def compare_models(prompt: str) -> dict:
    """
    Call both gpt-4o and gpt-4o-mini with the same prompt and return a
    comparison dictionary.

    Args:
        prompt: The user message to send to both models.

    Returns:
        A dict with keys:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (estimated USD for the response)

    Hint:
        Cost estimate = (len(response.split()) / 0.75) / 1000 * COST_PER_1K_OUTPUT_TOKENS["gpt-4o"]
        (0.75 words ≈ 1 token is a rough approximation)
    """
    # TODO: call call_openai and call_openai_mini, assemble and return the dict
    gpt4o_response, gpt4o_latency = call_openai(prompt)
    
    mini_response, mini_latency = call_openai_mini(prompt)
    
    word_count = len(gpt4o_response.split())
    gpt4o_cost_estimate = (word_count / 0.75) / 1000 * COST_PER_1K_OUTPUT_TOKENS["gpt-4o"]

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate
    }


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal.

    Behaviour:
        - Streams tokens from OpenAI as they arrive (print each chunk).
        - Maintains the last 3 conversation turns in history.
        - Typing 'quit' or 'exit' ends the loop.

    Hints:
        - Keep a list `history` of {"role": ..., "content": ...} dicts.
        - Use stream=True in client.chat.completions.create() and iterate:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - After each turn, append the assistant reply to history.
        - Trim history to the last 3 turns: history = history[-3:]
    """
    # TODO: enter while-loop, read user input, stream response, maintain history
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history = []
    
    while True:
        try:
            user_input = input("\nYou:")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chatbot.")
            break
        
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chatbot.")
            break
        
        if not user_input.strip():
            continue
        
        history.append({"role": "user", "content": user_input})
        print("Assistant:", end=" ", flush=True)
        
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            stream=True
        )
        
        assistant_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            assistant_reply += delta
        print()
        
        history.append({"role": "assistant", "content": assistant_reply})
        history = history[-3:]  # Keep last 3 turns (user + assistant)

# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    # TODO: implement retry loop with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, add "prompt" key
    comparision_results = []
    
    for prompt in prompts:
        result_dict = compare_models(prompt)
        result_dict["prompt"] = prompt
        comparision_results.append(result_dict)
        
    return comparision_results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of compare_models results as a readable text table.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A formatted string table with columns:
        Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency

    Hint:
        Truncate long text to 40 characters for readability.
    """
    # TODO: build and return a formatted table string
    def clean_text(text: Any, max_len: int = 40) -> str:
        safe_str = str(text).replace("\n", " ").strip()
        if len(safe_str) > max_len:
            return safe_str[:max_len - 3] + "..."
        return safe_str
    
    widths = {
        "prompt": 40,
        "gpt4o": 40,
        "mini": 40,
        "lat": 15
    }
        
    header = (
        f"{'Prompt':<{widths['prompt']}} | "
        f"{'GPT-4o Response':<{widths['gpt4o']}} | "
        f"{'Mini Response':<{widths['mini']}} | "
        f"{'GPT-4o Latency':<{widths['lat']}} | "
        f"{'Mini Latency':<{widths['lat']}}"
    )
    
    separator = (
        f"{'-' * widths['prompt']} | "
        f"{'-' * widths['gpt4o']} | "
        f"{'-' * widths['mini']} | "
        f"{'-' * widths['lat']} | "
        f"{'-' * widths['lat']}"
    )
    
    table_lines = [header, separator]

    # 3. Build Data Rows
    for row in results:
        prompt_txt = clean_text(row.get("prompt", ""), max_len=widths["prompt"])
        gpt4o_txt = clean_text(row.get("gpt4o_response", ""), max_len=widths["gpt4o"])
        mini_txt = clean_text(row.get("mini_response", ""), max_len=widths["mini"])
        
        # Safely convert and format latencies to 3 decimal positions
        gpt4o_lat = f"{row.get('gpt4o_latency', 0.0):.3f}s"
        mini_lat = f"{row.get('mini_latency', 0.0):.3f}s"
        
        row_str = (
            f"{prompt_txt:<{widths['prompt']}} | "
            f"{gpt4o_txt:<{widths['gpt4o']}} | "
            f"{mini_txt:<{widths['mini']}} | "
            f"{gpt4o_lat:<{widths['lat']}} | "
            f"{mini_lat:<{widths['lat']}}"
        )
        table_lines.append(row_str)

    # Return the aggregated table block as a single cohesive string
    return "\n".join(table_lines)

# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()
