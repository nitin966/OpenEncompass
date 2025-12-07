import logging
import asyncio
from typing import Any, Tuple, Union, Generator, Callable, List
from runtime.node import SearchNode
from core.signals import BranchPoint, ScoreSignal, Effect, Protect, KillBranch, EarlyStop, RecordCosts, ControlSignal, LocalSearch

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Runtime engine for executing AgentMachine instances.
    
    Features:
    - O(1) Replay: Uses machine.save()/load() to resume execution efficiently.
    - Signal Handling: Processes BranchPoint, Effect, LocalSearch, etc.
    - Caching: Caches results of deterministic steps (optional).
    - Safety: Ensures idempotent execution of effects.
    """
    @staticmethod
    def create_root() -> SearchNode:
        """Creates the initial root node with an empty history."""
        # Root has empty history
        return SearchNode(trace_history=[], depth=0, action_taken="<init>")

    def __init__(self):
        # Cache: history_hash -> (score, last_signal, is_terminal, final_result)
        self._cache = {}
        self._cache = {} # history_hash -> (score, last_signal, is_terminal, final_result)
        self._effect_cache = {} # scope -> effect_key -> result (Scoped Memoization)
        self._current_scope = "global"

    def set_scope(self, scope: str):
        """Sets the current caching scope."""
        self._current_scope = scope
        if scope not in self._effect_cache:
            self._effect_cache[scope] = {}

    def clear_scope(self, scope: str):
        """Clears the cache for a specific scope."""
        if scope in self._effect_cache:
            del self._effect_cache[scope]

    def _compute_history_hash(self, history: List[Any]) -> str:
        """
        Computes a robust hash for the history.
        Tries to use JSON for stability, falls back to string representation.
        """
        import json
        import hashlib
        
        try:
            # Try JSON serialization with sorted keys for determinism
            serialized = json.dumps(history, sort_keys=True)
        except (TypeError, ValueError):
            # Fallback: String representation of the tuple
            # This is less safe for objects with identical __str__ but better than crashing
            serialized = str(tuple(history))
            
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()

    async def step(self, agent_factory: Callable[[], Generator], node: SearchNode, input_value: Any = None) -> Tuple[SearchNode, Union[ControlSignal, None]]:
        """
        Executes the agent.
        1. Replays history from 'node'.
        2. Injects 'input_value' (if provided) as the decision for the current BranchPoint.
        3. Continues execution, handling any 'Effect' signals automatically.
        4. Stops at the next 'BranchPoint' or termination.
        """
        from core.signals import Effect, BranchPoint, ScoreSignal
        
        # 1. Construct the history to replay
        # The node.trace_history contains [choice_0, effect_res_1, choice_2, ...]
        replay_history = list(node.trace_history)
        if input_value is not None:
            replay_history.append(input_value)
            
        # We will build the *new* history for the child node as we go
        # It starts as a copy of replay_history, but might grow if we encounter NEW effects
        current_history = list(replay_history)
        
        # Check Cache (Optimization)
        history_key = self._compute_history_hash(current_history)
        # Note: The cache might return a state that is "in the middle" of effects if we cached poorly.
        # But our protocol says we only stop at BranchPoints. So cached states are always at BranchPoints.
        if history_key in self._cache:
             return self._reconstruct_from_cache(history_key, current_history, node)

        # 2. Start Execution
        # Check if agent_factory returns an AgentMachine class or instance
        # We need to inspect what agent_factory() returns.
        # But agent_factory is usually a function that returns a generator.
        # If we use the new compiler, agent_factory might be the Machine Class itself?
        
        # Let's assume agent_factory returns an AgentMachine INSTANCE or a Generator.
        agent_instance = agent_factory()
        
        from core.compiler import AgentMachine
        if isinstance(agent_instance, AgentMachine):
            # --- O(1) REPLAY PATH ---
            machine = agent_instance
            
            # Load state if available
            if node.machine_state:
                machine.load(node.machine_state)
            
            # If input provided, we are making a choice.
            # If input is None, we are starting or continuing auto-execution?
            # The machine.run() takes input.
            
            # DEBUG
            print(f"DEBUG: Engine step input: {input_value}")
            
            current_score = node.score
            is_done = False
            final_result = None
            last_signal = None
            
            try:
                # Run the machine
                signal = machine.run(input_value)
                
                # Check for completion
                if signal is None and machine._done:
                    print(f"DEBUG: Machine done. Result: {machine._result}")
                    raise StopIteration(machine._result)
                
                # Auto-execute Effects loop
                current_score = node.score
                is_done = False
                final_result = None
                last_signal = None
                
                while True:
                    if signal is None and machine._done:
                        print(f"DEBUG: Machine done in loop. Result: {machine._result}")
                        raise StopIteration(machine._result)
                        
                    if isinstance(signal, ScoreSignal):
                        current_score += signal.value
                        signal = machine.run(None)
                        
                    elif isinstance(signal, Effect):
                        # ... (Effect handling logic) ...
                        # Same logic as before
                        effect_key = signal.key
                        if effect_key is None:
                            try:
                                import json
                                import hashlib
                                payload = str(signal.args) + str(signal.kwargs)
                                arg_hash = hashlib.md5(payload.encode('utf-8')).hexdigest()
                                effect_key = f"{signal.func.__module__}.{signal.func.__name__}:{arg_hash}"
                            except Exception:
                                effect_key = None
                        
                        if self._current_scope not in self._effect_cache:
                            self._effect_cache[self._current_scope] = {}
                        scope_cache = self._effect_cache[self._current_scope]
                        
                        if effect_key and effect_key in scope_cache:
                            result = scope_cache[effect_key]
                        else:
                            if asyncio.iscoroutinefunction(signal.func):
                                 result = await signal.func(*signal.args, **signal.kwargs)
                            else:
                                 result = signal.func(*signal.args, **signal.kwargs)
                            if effect_key:
                                scope_cache[effect_key] = result
                        
                        # Feed result back to machine
                        signal = machine.run(result)
                        
                    elif isinstance(signal, Protect):
                         # ... (Protect logic) ...
                         attempts = signal.attempts
                         last_exc = None
                         result = None
                         success = False
                         for _ in range(attempts):
                             try:
                                 if asyncio.iscoroutinefunction(signal.func):
                                     result = await signal.func(*signal.args, **signal.kwargs)
                                 else:
                                     result = signal.func(*signal.args, **signal.kwargs)
                                 success = True
                                 break
                             except signal.exceptions as e:
                                 last_exc = e
                                 continue
                         if not success:
                             raise last_exc
                         signal = machine.run(result)

                    elif isinstance(signal, KillBranch):
                        current_score = -1e9
                        is_done = True
                        break
                        
                    elif isinstance(signal, EarlyStop):
                        current_score = 1e9
                        is_done = True
                        break
                        
                    elif isinstance(signal, RecordCosts):
                        signal = machine.run(None)
                        
                    elif isinstance(signal, BranchPoint):
                        last_signal = signal
                        break
                    
                    elif isinstance(signal, LocalSearch):
                        # Execute the local search strategy
                        # We need to instantiate the strategy
                        strategy_cls = signal.strategy_factory
                        sub_agent = signal.agent_factory
                        kwargs = signal.kwargs
                        
                        # We need a sampler. If not provided, use dummy?
                        # Or maybe the strategy doesn't need one if it's DFS?
                        # Let's assume kwargs has everything needed except engine/store.
                        
                        # We use a temporary MemoryStore or None
                        # Ideally we should share the store, but engine doesn't have reference to it.
                        # For now, pass None.
                        strategy = strategy_cls(None, self, **kwargs)
                        
                        # Run search
                        # This is async
                        results = await strategy.search(sub_agent)
                        
                        # Return best result
                        # results is list of SearchNode
                        if results:
                            # Prefer terminal nodes
                            terminals = [n for n in results if n.is_terminal]
                            if terminals:
                                best_result = terminals[0].metadata.get('result')
                            else:
                                best_result = results[0].metadata.get('result')
                            
                            signal = machine.run(best_result)
                        else:
                            # Search failed
                            signal = machine.run(None)

                    else:
                        raise TypeError(f"Unexpected signal type: {type(signal)}")
                        
            except StopIteration as e:
                is_done = True
                final_result = e.value
                
            # Create Child Node with SAVED STATE
            child = SearchNode(
                trace_history=node.trace_history + [input_value] if input_value is not None else [],
                score=current_score,
                depth=node.depth + 1 if input_value is not None else node.depth,
                parent_id=node.node_id,
                is_terminal=is_done,
                action_taken=str(input_value) if input_value is not None else "<auto>",
                metadata={'result': final_result} if is_done else {},
                machine_state=machine.save() # O(1) Checkpoint
            )
            return child, last_signal

        # --- LEGACY GENERATOR PATH (O(N) Replay) ---
        # We maintain a stack of generators to support nested agents
        gen_stack = [agent_instance]
        current_gen = gen_stack[-1]
        
        current_score = 0.0
        last_signal = None
        is_done = False
        final_result = None
        
        # Helper to advance the current generator stack
        def advance_generator(val=None):
            nonlocal current_gen
            try:
                sig = current_gen.send(val)
                
                # Handle nested generator yield
                while isinstance(sig, Generator):
                    # Push new generator
                    gen_stack.append(sig)
                    current_gen = sig
                    # Start it
                    sig = current_gen.send(None)
                    
                return sig
            except StopIteration as e:
                # Current generator finished
                if len(gen_stack) > 1:
                    # Pop and return result to parent
                    gen_stack.pop()
                    current_gen = gen_stack[-1]
                    # Pass the result of the sub-agent back to the parent
                    return advance_generator(e.value)
                else:
                    # Root agent finished
                    raise e

        try:
            # Start the root generator
            # We use send(None) initially
            signal = advance_generator(None)
            
            # --- REPLAY PHASE ---
            for stored_input in replay_history:
                # 1. Handle Scores
                while isinstance(signal, ScoreSignal):
                    current_score += signal.value
                    signal = advance_generator(None)
                
                # 2. Inject stored input
                if isinstance(signal, (BranchPoint, Effect)):
                    signal = advance_generator(stored_input)
                else:
                    raise TypeError(f"During replay, expected BranchPoint or Effect, got {type(signal)}")

            # --- FRONTIER PHASE ---
            while True:
                # Consume scores
                while isinstance(signal, ScoreSignal):
                    current_score += signal.value
                    signal = advance_generator(None)
                
                if isinstance(signal, Effect):
                    # ... (Effect handling logic) ...
                    effect_key = signal.key
                    if effect_key is None:
                        try:
                            import json
                            import hashlib
                            payload = str(signal.args) + str(signal.kwargs)
                            arg_hash = hashlib.md5(payload.encode('utf-8')).hexdigest()
                            effect_key = f"{signal.func.__module__}.{signal.func.__name__}:{arg_hash}"
                        except Exception:
                            effect_key = None
                    
                    # Check Scoped Cache
                    if self._current_scope not in self._effect_cache:
                        self._effect_cache[self._current_scope] = {}
                    scope_cache = self._effect_cache[self._current_scope]
                    
                    if effect_key and effect_key in scope_cache:
                        result = scope_cache[effect_key]
                    else:
                        if asyncio.iscoroutinefunction(signal.func):
                             result = await signal.func(*signal.args, **signal.kwargs)
                        else:
                             result = signal.func(*signal.args, **signal.kwargs)
                        if effect_key:
                            scope_cache[effect_key] = result
                    
                    current_history.append(result)
                    signal = advance_generator(result)
                
                elif isinstance(signal, Protect):
                    # Handle Protect
                    attempts = signal.attempts
                    last_exc = None
                    result = None
                    success = False
                    for _ in range(attempts):
                        try:
                            if asyncio.iscoroutinefunction(signal.func):
                                result = await signal.func(*signal.args, **signal.kwargs)
                            else:
                                result = signal.func(*signal.args, **signal.kwargs)
                            success = True
                            break
                        except signal.exceptions as e:
                            last_exc = e
                            continue
                    if not success:
                        raise last_exc
                    current_history.append(result)
                    signal = advance_generator(result)

                elif isinstance(signal, KillBranch):
                    current_score = -1e9
                    is_done = True
                    break
                    
                elif isinstance(signal, EarlyStop):
                    current_score = 1e9
                    is_done = True
                    break
                    
                elif isinstance(signal, RecordCosts):
                    current_history.append(None)
                    signal = advance_generator(None)
                    
                elif isinstance(signal, BranchPoint):
                    last_signal = signal
                    break
                    
                else:
                    raise TypeError(f"Unexpected signal type: {type(signal)}")

        except StopIteration as e:
            is_done = True
            final_result = e.value
            
        # Update Cache
        # We cache the state at this specific history point (which corresponds to a BranchPoint or End)
        history_key = self._compute_history_hash(current_history)
        self._cache[history_key] = (current_score, last_signal, is_done, final_result)
        
        # Create Child Node
        child = SearchNode(
            trace_history=current_history,
            score=current_score,
            depth=node.depth + 1 if input_value is not None else node.depth, # Depth increments on *choices*? Or steps? Let's say choices.
            parent_id=node.node_id,
            is_terminal=is_done,
            action_taken=str(input_value) if input_value is not None else "<auto>",
            metadata={'result': final_result} if is_done else {}
        )
        
        return child, last_signal

    def _reconstruct_from_cache(self, key: str, history: List[Any], parent: SearchNode) -> Tuple[SearchNode, Any]:
        """Helper to reconstruct node from cache."""
        score, signal, is_done, result = self._cache[key]
        child = SearchNode(
            trace_history=history,
            score=score,
            depth=parent.depth + 1, # Approx
            parent_id=parent.node_id,
            is_terminal=is_done,
            action_taken="<cached>",
            metadata={'result': result} if is_done else {}
        )
        return child, signal
