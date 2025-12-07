import asyncio
import json
import re
import os
from typing import List, Dict, Any
from encompass import compile, branchpoint, record_score, effect
from encompass.std import calculator, early_stop
from encompass.llm import OpenAIModel, CachingLM
from runtime.engine import ExecutionEngine
from search.strategies import BeamSearch

# --- 1. Dataset Loader ---
def load_gsm8k_test(path: str = "gsm8k_test.jsonl") -> List[Dict[str, str]]:
    """Loads GSM8K test set. If file missing, returns dummy set for demo."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using dummy data.")
        return [
            {"question": "Janet has 5 apples. She eats 2. How many left?", "answer": "3"},
            {"question": "Bob runs 10 miles. Ann runs half. Total?", "answer": "15"}
        ]
    
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# --- 2. Agent Definition ---
def create_math_agent(problem: str, llm):
    @compile
    def math_solver():
        # Context for the LLM
        context = f"Solve this math problem step-by-step. End with 'Final Answer: X'.\nProblem: {problem}\n"
        history = ""
        
        for _ in range(8): # Max 8 steps
            # Generate next step
            prompt = context + history + "Next step:"
            step = yield effect(llm.generate, prompt, stop=["\n"])
            
            history += f"Step: {step}\n"
            
            # Check for calculator usage
            if "Calculate:" in step:
                expr = step.split("Calculate:", 1)[1].strip()
                val = yield calculator(expr)
                history += f"Result: {val}\n"
                
            # Check for final answer
            if "Final Answer:" in step:
                answer = step.split("Final Answer:", 1)[1].strip()
                yield record_score(1.0) # Tentative score
                return answer
        
        return "Failed"
    return math_solver

# --- 3. Grader ---
def grade_answer(prediction: str, ground_truth: str) -> bool:
    """Extracts numbers and compares."""
    def extract_last_num(text):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        return float(nums[-1]) if nums else None
    
    try:
        pred_num = extract_last_num(prediction)
        gt_num = extract_last_num(ground_truth)
        return pred_num is not None and gt_num is not None and abs(pred_num - gt_num) < 1e-6
    except Exception:
        return False

# --- 4. Runner ---
async def run_benchmark(limit: int = 5):
    print(f"--- Running GSM8K Full Benchmark (Limit: {limit}) ---")
    
    # Setup
    base_lm = OpenAIModel(model="gpt-3.5-turbo") # Requires API Key
    llm = CachingLM(base_lm, "gsm8k_cache.json")
    engine = ExecutionEngine()
    
    dataset = load_gsm8k_test()[:limit]
    correct = 0
    total = 0
    
    for i, task in enumerate(dataset):
        print(f"Task {i+1}/{limit}: {task['question'][:50]}...")
        
        agent = create_math_agent(task['question'], llm)
        
        # Use Beam Search
        # We need a sampler. For now, use a dummy sampler that just returns [0] 
        # because our agent is linear (no BranchPoints, just Effects).
        # Wait, if it's linear, Beam Search is overkill.
        # But we want to support branching later.
        # For this parity demo, let's assume the agent makes *choices*?
        # The current agent uses `llm.generate` which is an Effect.
        # So it's a single path.
        
        # To make it a search, we should use `BranchPoint` for "Generate 3 candidates".
        # But let's stick to the linear agent for now to verify the harness.
        
        # Just run it directly with engine.step loop?
        # Or use BeamSearch with width=1.
        
        async def dummy_sampler(node, metadata=None):
            return [0]

        from storage.filesystem import FileSystemStore
        store = FileSystemStore()
        beam = BeamSearch(store, engine, dummy_sampler, width=1)
        results = await beam.search(agent)
        
        if results:
            best_node = results[0]
            prediction = best_node.metadata.get('result', "")
            is_correct = grade_answer(str(prediction), str(task['answer']))
            
            print(f"  Pred: {prediction}, GT: {task['answer']}, Correct: {is_correct}")
            if is_correct:
                correct += 1
        else:
            print("  Failed to solve.")
            
        total += 1
    
    print(f"\nResults: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
