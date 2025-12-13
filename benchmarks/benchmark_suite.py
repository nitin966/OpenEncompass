"""
Comprehensive benchmark suite for evaluating LLM agents.

This module provides robust, respectable benchmarks with:
- Statistical significance (50+ problems)
- Multiple domains (math, code)
- Proper metrics (accuracy, pass@k, CI)
- Comparison framework
- Detailed reporting

Usage:
    python benchmarks/benchmark_suite.py --model mistral --suite all
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for LLM agent evaluation.

    Provides:
    - GSM8K: Math word problems (50-100 problems)
    - HumanEval: Code generation (164 problems)
    - Proper statistical analysis
    - Comparison across models/strategies
    - Detailed reporting
    """

    def __init__(self, model_name: str = "mistral", output_dir: str = "data/benchmarks"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    async def run_gsm8k(self, num_problems: int = 50, strategy: str = "beam", width: int = 3):
        """
        Run GSM8K benchmark with statistical rigor.

        Args:
            num_problems: Number of problems (50-100 recommended)
            strategy: Search strategy
            width: Beam width

        Returns:
            Dict with results and statistics
        """
        print(f"\n{'=' * 70}")
        print(f"GSM8K Benchmark: {num_problems} problems")
        print(f"{'=' * 70}\n")

        from benchmarks.run_gsm8k_ollama import run_benchmark

        # Run evaluation
        start_time = time.time()
        await run_benchmark(
            model_name=self.model_name, num_problems=num_problems, strategy=strategy, width=width
        )

        # Find the most recent results file for this model
        results_dir = Path("data/gsm8k_ollama")
        results_files = list(results_dir.glob(f"results_{self.model_name}_*.json"))
        if not results_files:
            raise FileNotFoundError(f"No results found in {results_dir}")

        # Get most recent
        results_file = max(results_files, key=lambda p: p.stat().st_mtime)

        with open(results_file) as f:
            data = json.load(f)

        # Calculate statistics
        solved = data["solved"]
        total = data["num_problems"]
        accuracy = data["accuracy"]

        # Wilson score confidence interval (95%)
        ci_lower, ci_upper = self._wilson_ci(solved, total)

        results = {
            "benchmark": "GSM8K",
            "model": self.model_name,
            "strategy": strategy,
            "num_problems": total,  # Use actual, not requested
            "solved": solved,
            "accuracy": accuracy,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "avg_time": data["avg_time"],
            "avg_nodes": data["avg_nodes"],
            "total_time": time.time() - start_time,
        }

        self.results["gsm8k"] = results
        return results

    async def run_humaneval(self, num_problems: int = 50, strategy: str = "beam", width: int = 3):
        """
        Run HumanEval code generation benchmark.

        Args:
            num_problems: Number of problems (up to 164)
            strategy: Search strategy
            width: Beam width

        Returns:
            Dict with results and statistics
        """
        print(f"\n{'=' * 70}")
        print(f"HumanEval Benchmark: {num_problems} problems")
        print(f"{'=' * 70}\n")

        # TODO: Implement HumanEval
        # For now, placeholder
        print("HumanEval implementation coming soon...")
        print(f"Would evaluate {num_problems} code generation problems")

        return {
            "benchmark": "HumanEval",
            "model": self.model_name,
            "status": "not_implemented",
            "num_problems": num_problems,
        }

    async def run_comparison(
        self, benchmarks: list[str], models: list[str], num_problems: int = 20
    ):
        """
        Compare multiple models on benchmarks.

        Args:
            benchmarks: List of benchmark names
            models: List of model names
            num_problems: Problems per benchmark

        Returns:
            Comparison results
        """
        print(f"\n{'=' * 70}")
        print(f"Comparison: {len(models)} models x {len(benchmarks)} benchmarks")
        print(f"{'=' * 70}\n")

        comparison = {}

        for model in models:
            self.model_name = model
            comparison[model] = {}

            for benchmark in benchmarks:
                if benchmark == "gsm8k":
                    result = await self.run_gsm8k(num_problems=num_problems)
                    comparison[model][benchmark] = result
                elif benchmark == "humaneval":
                    result = await self.run_humaneval(num_problems=num_problems)
                    comparison[model][benchmark] = result

        self.results["comparison"] = comparison
        return comparison

    def _wilson_ci(self, successes: int, trials: int, confidence: float = 0.95) -> tuple:
        """
        Calculate Wilson score confidence interval.

        More accurate than normal approximation for small samples.
        """
        if trials == 0:
            return (0.0, 0.0)

        from math import sqrt

        # Z-score for confidence level
        z = 1.96  # 95% confidence

        p = successes / trials
        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        spread = z * sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator

        return (max(0, centre - spread), min(1, centre + spread))

    def generate_report(self) -> str:
        """
        Generate markdown report of benchmark results.

        Returns:
            Markdown formatted report
        """
        report = "# Benchmark Results\n\n"
        report += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "---\n\n"

        # GSM8K Results
        if "gsm8k" in self.results:
            gsm8k = self.results["gsm8k"]
            report += "## GSM8K (Math Word Problems)\n\n"
            report += f"**Model**: {gsm8k['model']}\n"
            report += f"**Strategy**: {gsm8k['strategy']}\n\n"
            report += "| Metric | Value |\n"
            report += "|--------|-------|\n"
            report += f"| Problems | {gsm8k['num_problems']} |\n"
            report += f"| Solved | {gsm8k['solved']} |\n"
            report += f"| **Accuracy** | **{gsm8k['accuracy']:.1%}** |\n"
            report += f"| 95% CI | [{gsm8k['ci_lower']:.1%}, {gsm8k['ci_upper']:.1%}] |\n"
            report += f"| Avg Time | {gsm8k['avg_time']:.2f}s |\n"
            report += f"| Avg Nodes | {gsm8k['avg_nodes']:.1f} |\n\n"

        # Comparison Results
        if "comparison" in self.results:
            report += "## Model Comparison\n\n"
            comparison = self.results["comparison"]

            # GSM8K comparison table
            report += "### GSM8K Results\n\n"
            report += "| Model | Accuracy | 95% CI | Avg Time | Avg Nodes |\n"
            report += "|-------|----------|--------|----------|----------|\n"

            for model, benchmarks in comparison.items():
                if "gsm8k" in benchmarks:
                    r = benchmarks["gsm8k"]
                    report += f"| {model} | {r['accuracy']:.1%} | "
                    report += f"[{r['ci_lower']:.1%}, {r['ci_upper']:.1%}] | "
                    report += f"{r['avg_time']:.2f}s | {r['avg_nodes']:.1f} |\n"

            report += "\n"

        return report

    def save_results(self):
        """Save detailed results and report."""
        # Save raw results as JSON
        results_file = (
            self.output_dir
            / f"results_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save markdown report
        report = self.generate_report()
        report_file = (
            self.output_dir
            / f"report_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(report_file, "w") as f:
            f.write(report)

        print(f"\n{'=' * 70}")
        print("Results saved:")
        print(f"  JSON: {results_file}")
        print(f"  Report: {report_file}")
        print(f"{'=' * 70}\n")

        # Print report preview
        print(report)


async def main():
    parser = argparse.ArgumentParser(description="Run comprehensive benchmark suite")
    parser.add_argument("--model", default="mistral", help="Model name")
    parser.add_argument(
        "--suite", default="gsm8k", choices=["gsm8k", "humaneval", "all", "comparison"]
    )
    parser.add_argument("--num-problems", type=int, default=50, help="Number of problems")
    parser.add_argument("--strategy", default="beam", help="Search strategy")
    parser.add_argument("--width", type=int, default=3, help="Beam width")
    parser.add_argument("--compare-models", nargs="+", help="Models to compare")

    args = parser.parse_args()

    suite = BenchmarkSuite(model_name=args.model)

    if args.suite == "gsm8k":
        await suite.run_gsm8k(
            num_problems=args.num_problems, strategy=args.strategy, width=args.width
        )
    elif args.suite == "humaneval":
        await suite.run_humaneval(
            num_problems=args.num_problems, strategy=args.strategy, width=args.width
        )
    elif args.suite == "all":
        await suite.run_gsm8k(num_problems=args.num_problems)
        await suite.run_humaneval(num_problems=args.num_problems)
    elif args.suite == "comparison":
        models = args.compare_models or ["llama2", "mistral"]
        await suite.run_comparison(
            benchmarks=["gsm8k"], models=models, num_problems=args.num_problems
        )

    suite.save_results()


if __name__ == "__main__":
    asyncio.run(main())
