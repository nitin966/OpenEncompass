"""
Run GSM8K benchmark with local Ollama LLM.

This is a REAL benchmark with actual LLM calls, not an oracle.

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: `ollama pull llama2`
    3. pip install aiohttp

Usage:
    python benchmarks/run_gsm8k_ollama.py --model llama2 --num-problems 10
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.gsm8k_full import create_math_agent, load_gsm8k_dataset
from encompass.llm.ollama import OllamaModel
from runtime.engine import ExecutionEngine
from search.strategies import BeamSearch
from storage.filesystem import FileSystemStore


async def create_llm_sampler(llm):
    """
    Create a sampler that uses LLM to generate solution steps.

    This is the KEY difference from oracle - the LLM actually solves the problem.
    """

    async def sampler(node, metadata=None):
        problem = metadata.get("problem", "")
        context = metadata.get("context", "")

        # Build prompt
        prompt = f"""You are solving a math word problem step by step.

Problem: {problem}

{context}

What's the next step? Choose ONE:
- If you need to do a calculation, respond with "Calculate: <expression>" where <expression> is valid Python.
  - IMPORTANT: Do NOT include any text, units, or explanations inside the expression. Use ONLY numbers and operators.
  - Correct: "Calculate: 12 * 52"
  - Incorrect: "Calculate: 12 * 52 (weeks)"
- If you're ready to give the final answer, respond with "Final Answer"
- If you're stuck, respond with "Give up"

Examples:
- "Calculate: 48 / 2"
- "Calculate: 100 - 95"
- "Calculate: 100 - (15 + 30)"  # Explicit subtraction example
- "Final Answer"

Your response:"""

        try:
            response = await llm.generate(prompt, max_tokens=100, temperature=0.3)

            # Parse response
            response = response.strip()

            # Robust parsing for chatty models (e.g. Llama 3)
            import re

            # Look for "Calculate: <expression>"
            calc_match = re.search(r"Calculate:\s*(.+)", response, re.IGNORECASE)
            if calc_match:
                # Extract just the calculation part, stopping at newline if present
                calc_expr = calc_match.group(1).split("\n")[0].strip()
                response = f"Calculate: {calc_expr}"
            elif "Final Answer" in response:
                response = "Final Answer"
            elif "Give up" in response:
                response = "Give up"

            # Provide options for the agent
            options = [response]

            # Add some variations to explore
            if "Calculate:" in response:
                options.append("Final Answer")
            else:
                options.append("Give up")

            return options[:3]  # Limit to 3 options

        except Exception as e:
            print(f"LLM sampler error: {e}")
            return ["Final Answer", "Give up"]

    return sampler


async def run_benchmark(model_name="llama2", num_problems=5, strategy="beam", width=3):
    """
    Run GSM8K benchmark with Ollama LLM.

    Args:
        model_name: Ollama model name (e.g., "llama2", "mistral")
        num_problems: Number of problems to evaluate
        strategy: Search strategy ("beam")
        width: Beam width
    """
    print(f"\n{'=' * 60}")
    print(f"GSM8K Benchmark with Ollama ({model_name})")
    print(f"{'=' * 60}\n")

    # Load dataset
    print(f"Loading {num_problems} problems...")
    problems = load_gsm8k_dataset(split="test", num_samples=num_problems)

    # Initialize LLM
    print(f"Connecting to Ollama ({model_name})...")
    llm = OllamaModel(model=model_name, temperature=0.3)

    # Test LLM connection
    try:
        test_response = await llm.generate("Test: respond with 'OK'", max_tokens=10)
        print(f"✓ LLM connected: {test_response}\n")
    except Exception as e:
        print(f"✗ LLM connection failed: {e}")
        print("Make sure Ollama is running: `ollama serve`")
        return

    # Setup engine and strategy
    engine = ExecutionEngine()
    store = FileSystemStore("./data/gsm8k_ollama")
    sampler = await create_llm_sampler(llm)

    if strategy == "beam":
        search_strategy = BeamSearch(store=store, engine=engine, sampler=sampler, width=width)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Run evaluation
    results = []
    solved = 0
    total_time = 0
    total_nodes = 0

    for i, problem in enumerate(problems):
        print(f"\n{'─' * 60}")
        print(f"Problem {i + 1}/{len(problems)}")
        print(f"{'─' * 60}")
        print(f"Q: {problem['question'][:100]}...")
        print(f"Expected: {problem['answer']}")

        agent = create_math_agent(problem["question"], problem["answer"])

        start_time = time.time()
        try:
            nodes = await search_strategy.search(agent)
            elapsed = time.time() - start_time

            # Find best terminal node
            terminal_nodes = [n for n in nodes if n.is_terminal]
            if terminal_nodes:
                best = max(terminal_nodes, key=lambda n: n.score)
                is_solved = best.score > 0

                if is_solved:
                    solved += 1
                    print(f"✓ SOLVED in {elapsed:.1f}s ({len(nodes)} nodes)")
                else:
                    print(f"✗ FAILED in {elapsed:.1f}s ({len(nodes)} nodes)")

                results.append(
                    {
                        "problem_id": i,
                        "solved": is_solved,
                        "score": best.score,
                        "nodes": len(nodes),
                        "time": elapsed,
                    }
                )
            else:
                print(f"✗ NO SOLUTION in {elapsed:.1f}s ({len(nodes)} nodes)")
                results.append(
                    {"problem_id": i, "solved": False, "nodes": len(nodes), "time": elapsed}
                )

            total_time += elapsed
            total_nodes += len(nodes)

        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append({"problem_id": i, "error": str(e)})

    # Summary
    accuracy = solved / len(problems) if problems else 0
    avg_time = total_time / len(problems) if problems else 0
    avg_nodes = total_nodes / len(problems) if problems else 0

    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Strategy: {strategy} (width={width})")
    print(f"Problems: {len(problems)}")
    print(f"Solved: {solved}/{len(problems)} ({accuracy:.1%})")
    print(f"Avg time: {avg_time:.1f}s")
    print(f"Avg nodes: {avg_nodes:.1f}")
    print(f"Total time: {total_time:.1f}s")

    # Save results
    output = {
        "model": model_name,
        "strategy": strategy,
        "width": width,
        "num_problems": len(problems),
        "solved": solved,
        "accuracy": accuracy,
        "avg_time": avg_time,
        "avg_nodes": avg_nodes,
        "total_time": total_time,
        "results": results,
    }

    output_file = Path("data/gsm8k_ollama") / f"results_{model_name}_{len(problems)}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GSM8K with Ollama")
    parser.add_argument("--model", default="llama2", help="Ollama model name")
    parser.add_argument("--num-problems", type=int, default=5, help="Number of problems")
    parser.add_argument("--strategy", default="beam", choices=["beam"], help="Search strategy")
    parser.add_argument("--width", type=int, default=3, help="Beam width")

    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            model_name=args.model,
            num_problems=args.num_problems,
            strategy=args.strategy,
            width=args.width,
        )
    )
