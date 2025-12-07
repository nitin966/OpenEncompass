import asyncio
from runtime.engine import ExecutionEngine
from core.compiler import compile_agent
from encompass import branchpoint
from encompass.std import local_search
from search.strategies import BeamSearch

async def main():
    def sub_agent():
        x = yield branchpoint("sub_choice")
        return x * 2
        
    SubAgentClass = compile_agent(sub_agent)
    
    async def dummy_sampler(node, metadata=None):
        return [5, 10]
        
    engine = ExecutionEngine()
    
    # Run BeamSearch directly
    strategy = BeamSearch(None, engine, dummy_sampler, width=2)
    results = await strategy.search(SubAgentClass)
    
    print(f"Results count: {len(results)}")
    for i, node in enumerate(results):
        print(f"Node {i}: score={node.score}, result={node.metadata.get('result')}")

if __name__ == "__main__":
    asyncio.run(main())
