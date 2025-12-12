import argparse
import asyncio
import json
import time
import random
from runtime.engine import ExecutionEngine
from storage.filesystem import FileSystemStore
from search.strategies import BeamSearch, MCTS, BestOfNSearch, BestFirstSearch, BFS, DFS

# Import benchmarks
from benchmarks.gsm8k import create_math_agent, solver_sampler, DATASET as GSM8K_DATASET
from benchmarks.gsm8k_full import load_gsm8k_dataset
from benchmarks.arc_hypothesis import arc_agent, arc_sampler, create_arc_llm_sampler
from benchmarks.reflexion import reflexion_agent, reflexion_sampler, create_reflexion_llm_sampler
from benchmarks.run_gsm8k_ollama import create_llm_sampler
from encompass.llm.ollama import OllamaModel

def get_strategy(name, store, engine, sampler, **kwargs):
    if name == "beam":
        return BeamSearch(store, engine, sampler, width=kwargs.get("width", 3), diversity_penalty=kwargs.get("diversity", 0.0))
    elif name == "mcts":
        return MCTS(store, engine, sampler, iterations=kwargs.get("iterations", 100))
    elif name == "best_of_n":
        return BestOfNSearch(store, engine, sampler, n=kwargs.get("n", 10))
    elif name == "bfs":
        return BFS(store, engine, sampler)
    elif name == "dfs":
        return DFS(store, engine, sampler)
    elif name == "befs":
        return BestFirstSearch(store, engine, sampler)
    else:
        raise ValueError(f"Unknown strategy: {name}")

async def run(args):
    # Setup
    random.seed(args.seed)
    
    # Determine Dataset and Agent Factory
    tasks = []
    if args.benchmark == "gsm8k":
        # Initialize LLM if requested
        llm_sampler = None
        if args.real_llm:
            print(f"Initializing real LLM: {args.model} (T={args.temperature})")
            llm = OllamaModel(model=args.model, temperature=args.temperature)
            llm_sampler = await create_llm_sampler(llm)

        for i, item in enumerate(GSM8K_DATASET):
            agent_factory = create_math_agent(item["question"], item["answer"])
            # Use real sampler if available, otherwise mock
            sampler_to_use = llm_sampler if args.real_llm else solver_sampler
            tasks.append((f"gsm8k_{i}", agent_factory, sampler_to_use))
            
    elif args.benchmark == "gsm8k_full":
        # Initialize LLM if requested
        llm_sampler = None
        if args.real_llm:
            print(f"Initializing real LLM: {args.model} (T={args.temperature})")
            llm = OllamaModel(model=args.model, temperature=args.temperature)
            llm_sampler = await create_llm_sampler(llm)
        else:
            print("WARNING: Running gsm8k_full without --real-llm will likely fail as mock sampler only knows mini-eval problems.")

        print(f"Loading GSM8K full dataset (limit={args.limit})...")
        problems = load_gsm8k_dataset(split="test", num_samples=args.limit)
        print(f"Loaded {len(problems)} problems.")

        for i, item in enumerate(problems):
            agent_factory = create_math_agent(item["question"], item["answer"])
            sampler_to_use = llm_sampler if args.real_llm else solver_sampler
            tasks.append((f"gsm8k_full_{i}", agent_factory, sampler_to_use))
            
    elif args.benchmark == "arc":
        if args.real_llm:
            print(f"Initializing real LLM: {args.model} (T={args.temperature})")
            llm = OllamaModel(model=args.model, temperature=args.temperature)
            sampler = await create_arc_llm_sampler(llm)
        else:
            sampler = arc_sampler
        tasks.append(("arc_task", arc_agent, sampler))
    elif args.benchmark == "reflexion":
        if args.real_llm:
            print(f"Initializing real LLM: {args.model} (T={args.temperature})")
            llm = OllamaModel(model=args.model, temperature=args.temperature)
            sampler = await create_reflexion_llm_sampler(llm)
        else:
            sampler = reflexion_sampler
        tasks.append(("reflexion_task", reflexion_agent, sampler))
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
        
    print(f"--- Running {args.benchmark} ({len(tasks)} tasks) with {args.strategy} (Seed={args.seed}) ---")
    
    total_solved = 0
    total_nodes = 0
    start_time = time.time()
    
    for task_id, agent_factory, sampler in tasks:
        store = FileSystemStore(f"traces/{args.benchmark}_{args.strategy}_{args.seed}/{task_id}")
        engine = ExecutionEngine()
        engine.set_scope(f"{task_id}_{args.seed}")
        
        strategy = get_strategy(args.strategy, store, engine, sampler, 
                                width=args.width, 
                                n=args.n, 
                                iterations=args.iterations,
                                diversity=args.diversity)
        
        results = await strategy.search(agent_factory)
        
        best_node = max(results, key=lambda n: n.score) if results else None
        # Check metadata for "Solved" OR high score (>= 100)
        solved = False
        if best_node:
            solved = "Solved" in str(best_node.metadata) or best_node.score >= 100.0
        
        if solved:
            total_solved += 1
        total_nodes += len(results)
        
        print(f"Task {task_id}: Solved={solved}, Best Score={best_node.score if best_node else 0}")

    duration = time.time() - start_time
    accuracy = total_solved / len(tasks) if tasks else 0.0
    
    metrics = {
        "benchmark": args.benchmark,
        "strategy": args.strategy,
        "seed": args.seed,
        "tasks": len(tasks),
        "accuracy": accuracy,
        "total_nodes": total_nodes,
        "duration": duration
    }
    
    print(json.dumps(metrics, indent=2))
    
    # Save metrics
    if args.output:
        with open(args.output, "a") as f:
            f.write(json.dumps(metrics) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnCompass Benchmark Runner")
    parser.add_argument("--benchmark", type=str, required=True, choices=["gsm8k", "gsm8k_full", "arc", "reflexion"])
    parser.add_argument("--strategy", type=str, required=True, choices=["beam", "mcts", "best_of_n", "bfs", "dfs", "befs"])
    parser.add_argument("--width", type=int, default=3, help="Beam width")
    parser.add_argument("--n", type=int, default=10, help="N for Best of N")
    parser.add_argument("--iterations", type=int, default=100, help="MCTS iterations")
    parser.add_argument("--diversity", type=float, default=0.0, help="Diversity penalty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, help="Output file for metrics")
    parser.add_argument("--real-llm", action="store_true", help="Use real LLM (Ollama) instead of mock sampler")
    parser.add_argument("--model", type=str, default="mistral", help="Ollama model name (if --real-llm is set)")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature (default: 0.7)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems (for gsm8k_full)")
    
    args = parser.parse_args()
    asyncio.run(run(args))
