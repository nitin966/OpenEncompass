import dill

class AgentMachine:
    """
    Base class for compiled state machines.
    """
    def __init__(self):
        self._state = 0
        self._ctx = {} # Local variables
        self._done = False
        self._result = None
        self._stack = [] # Call stack for nested agents

    def run(self, _input=None):
        """
        Executes one step of the machine.
        Returns: (signal, done)
        """
        raise NotImplementedError

    def save(self):
        """Returns a pickleable state object (bytes)."""
        return dill.dumps(self)

    def load(self, state):
        """Restores state from bytes."""
        if isinstance(state, bytes):
            loaded = dill.loads(state)
            self.__dict__.update(loaded.__dict__)
        else:
            # Legacy dict support (if needed)
            self._state = state["_state"]
            self._ctx = state["_ctx"]
            self._done = state["_done"]
            self._result = state["_result"]
            self._stack = state.get("_stack", [])

import ast
import inspect
import textwrap
from typing import Any, Callable, Dict, Type

class CPSCompiler(ast.NodeTransformer):
    """
    Compiler that transforms generator functions into Continuation-Passing Style (CPS)
    state machines.
    
    This allows generator functions to be pickled and resumed from any yield point
    in O(1) time, enabling true checkpointing and branching.
    
    The compiler:
    1. Hoists local variables to `self._ctx`.
    2. Splits code into blocks at `yield` points.
    3. Generates a `run()` method with a state dispatcher.
    """
    def __init__(self, varnames):
        self.states = {} # state_id -> list of statements
        self.current_state = 0
        self.current_stmts = []
        self.varnames = set(varnames)
        self.loop_stack = [] # List of dicts: {'head': int, 'after': int (placeholder)}
        self.placeholder_counter = -100

    def _flush_state(self, next_state=None):
        # If next_state is provided, we are transitioning.
        # The transition assignment MUST happen before the return (which is already in current_stmts?)
        # No, visit_Assign appended Return.
        # We need to insert state assignment BEFORE the Return.
        
        if next_state is not None:
            # Find the return statement
            if self.current_stmts and isinstance(self.current_stmts[-1], ast.Return):
                ret = self.current_stmts.pop()
                # Add transition
                self.current_stmts.append(ast.parse(f"self._state = {next_state}").body[0])
                # Add return back
                self.current_stmts.append(ret)
            else:
                # Just append
                self.current_stmts.append(ast.parse(f"self._state = {next_state}").body[0])
        
        self.states[self.current_state] = self.current_stmts
        self.current_state = next_state if next_state is not None else self.current_state + 1
        self.current_stmts = []

    def visit_Assign(self, node):
        # Rewrite assignments to use self._ctx
        # x = ... -> self._ctx['x'] = ...
        targets = []
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in self.varnames:
                targets.append(ast.Subscript(
                    value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_ctx', ctx=ast.Load()),
                    slice=ast.Constant(value=t.id),
                    ctx=ast.Store()
                ))
            else:
                targets.append(t)
        
        # Check yield
        if isinstance(node.value, ast.Yield):
            # x = yield foo
            # 1. return foo
            # 2. next state: self._ctx['x'] = _input
            
            # Emit return yield_val
            # We must visit the yield value to rewrite variables!
            yield_val = self.visit(node.value.value) if node.value.value else None
            self.current_stmts.append(ast.Return(value=yield_val))
            
            next_state = self.current_state + 1
            self._flush_state(next_state)
            
            # In next state, assign input
            # self._ctx['x'] = _input
            assign = ast.Assign(
                targets=targets,
                value=ast.Name(id='_input', ctx=ast.Load())
            )
            self.current_stmts.append(assign)
        else:
            # Normal assignment
            # self._ctx['x'] = value
            # We need to recursively visit value to rewrite variable reads too!
            value = self.visit(node.value)
            self.current_stmts.append(ast.Assign(targets=targets, value=value))

    def visit_Global(self, node):
        # Preserve global declarations
        self.current_stmts.append(node)
    
    def visit_Nonlocal(self, node):
        # Preserve nonlocal declarations
        self.current_stmts.append(node)
    
    def visit_AugAssign(self, node):
        # Handle augmented assignments like x += 1
        # If x is a local variable (in varnames), rewrite to self._ctx['x'] += ...
        # Otherwise, keep as is (for globals)
        if isinstance(node.target, ast.Name) and node.target.id in self.varnames:
            # Rewrite to self._ctx['x'] += value
            target = ast.Subscript(
                value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_ctx', ctx=ast.Load()),
                slice=ast.Constant(value=node.target.id),
                ctx=ast.Store()
            )
            value = self.visit(node.value)
            self.current_stmts.append(ast.AugAssign(target=target, op=node.op, value=value))
        else:
            # Keep as is (for globals or attributes)
            value = self.visit(node.value)
            self.current_stmts.append(ast.AugAssign(target=node.target, op=node.op, value=value))

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Yield):
            yield_val = self.visit(node.value.value) if node.value.value else None
            self.current_stmts.append(ast.Return(value=yield_val))
            
            next_state = self.current_state + 1
            self._flush_state(next_state)
        else:
            self.current_stmts.append(self.generic_visit(node))

    def visit_If(self, node):
        # 1. Visit test
        test = self.visit(node.test)
        
        # 2. Emit conditional jump with placeholders
        entry_state = self.current_state
        
        # Placeholder values
        THEN_PH = -1
        ELSE_PH = -2
        JOIN_PH = -3
        
        # Construct the If statement for the state machine
        if_stmt = ast.If(
            test=test,
            body=[
                ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
                    value=ast.Constant(value=THEN_PH)
                ),
                ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[]))
            ],
            orelse=[
                ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
                    value=ast.Constant(value=ELSE_PH)
                ),
                ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[]))
            ]
        )
        self.current_stmts.append(if_stmt)
        self._flush_state(None) # Flush entry block
        
        # 3. Visit THEN block
        then_start = self.current_state
        for stmt in node.body:
            self.visit(stmt)
        
        # Check if we need to jump to join
        if not (self.current_stmts and isinstance(self.current_stmts[-1], ast.Return)):
             self.current_stmts.append(ast.Assign(
                 targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
                 value=ast.Constant(value=JOIN_PH)
             ))
             self.current_stmts.append(ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[])))
        
        then_end = self.current_state
        self._flush_state(None)
        
        # 4. Visit ELSE block
        else_start = self.current_state
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
            
            if not (self.current_stmts and isinstance(self.current_stmts[-1], ast.Return)):
                 self.current_stmts.append(ast.Assign(
                     targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
                     value=ast.Constant(value=JOIN_PH)
                 ))
                 self.current_stmts.append(ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[])))
        else:
            # Empty else means jump directly to join
            # We can handle this by setting else_start = join_start later?
            # Or we can emit a jump block?
            # Better: Set else_start to JOIN_PH (resolved later)
            pass
            
        else_end = self.current_state
        self._flush_state(None)
        
        # 5. Join State
        join_start = self.current_state
        
        # 6. Backpatching
        # Fix Entry
        entry_stmts = self.states[entry_state]
        # The last stmt is the If
        if_node = entry_stmts[-1]
        # Fix THEN
        if_node.body[0].value.value = then_start
        # Fix ELSE
        if node.orelse:
            if_node.orelse[0].value.value = else_start
        else:
            if_node.orelse[0].value.value = join_start
            
        # Fix THEN End (if it has placeholder)
        def patch_jump(state_id, target):
            stmts = self.states[state_id]
            if not stmts: return
            # Look for assignment to _state with placeholder
            for stmt in stmts:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    t = stmt.targets[0]
                    if isinstance(t, ast.Attribute) and t.attr == '_state':
                        if isinstance(stmt.value, ast.Constant) and stmt.value.value == JOIN_PH:
                            stmt.value.value = target
                            
        patch_jump(then_end, join_start)
        patch_jump(else_end, join_start)

    def visit_While(self, node):
        # 1. Emit jump to HEAD
        head_start = self.current_state + 1
        self.current_stmts.append(ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[])))
        self._flush_state(head_start)
        
        # 2. HEAD State: Test
        test = self.visit(node.test)
        
        BODY_PH = self.placeholder_counter
        self.placeholder_counter -= 1
        AFTER_PH = self.placeholder_counter
        self.placeholder_counter -= 1
        
        self.loop_stack.append({'head': head_start, 'after': AFTER_PH})
        
        # if test: goto BODY; else: goto AFTER
        if_stmt = ast.If(
            test=test,
            body=[
                ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
                    value=ast.Constant(value=BODY_PH)
                ),
                ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[]))
            ],
            orelse=[
                ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
                    value=ast.Constant(value=AFTER_PH)
                ),
                ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[]))
            ]
        )
        self.current_stmts.append(if_stmt)
        self._flush_state(None) # Flush HEAD
        
        # 3. BODY State
        body_start = self.current_state
        for stmt in node.body:
            self.visit(stmt)
            
        # Jump back to HEAD
        # Always add this jump, even if there was a yield (which creates a Return in an earlier state)
        # The jump needs to be in the CURRENT state (after all yields have been processed)
        self.current_stmts.append(ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
            value=ast.Constant(value=head_start)
        ))
        self.current_stmts.append(ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[])))
        
        body_end = self.current_state
        self._flush_state(None)
        
        self.loop_stack.pop()
        
        # 4. AFTER State
        after_start = self.current_state
        
        # 5. Backpatching
        # Fix HEAD
        head_stmts = self.states[head_start]
        if_node = head_stmts[-1]
        if_node.body[0].value.value = body_start
        if_node.orelse[0].value.value = after_start
        
        # Global backpatch for breaks/after
        def patch_placeholders(ph, target):
            for sid, stmts in self.states.items():
                if not stmts: continue
                for stmt in stmts:
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                        t = stmt.targets[0]
                        if isinstance(t, ast.Attribute) and t.attr == '_state':
                            if isinstance(stmt.value, ast.Constant) and stmt.value.value == ph:
                                stmt.value.value = target
        
        patch_placeholders(AFTER_PH, after_start)

    def visit_For(self, node):
        # 1. Init Iterator
        iter_name = f"_iter_{self.current_state}"
        iter_expr = self.visit(node.iter)
        
        # self._ctx[iter_name] = iter(expr)
        self.current_stmts.append(ast.Assign(
            targets=[ast.Subscript(
                value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_ctx', ctx=ast.Load()),
                slice=ast.Constant(value=iter_name),
                ctx=ast.Store()
            )],
            value=ast.Call(func=ast.Name(id='iter', ctx=ast.Load()), args=[iter_expr], keywords=[])
        ))
        
        head_start = self.current_state + 1
        self.current_stmts.append(ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[])))
        self._flush_state(head_start)
        
        # 2. HEAD State: Next
        BODY_PH = self.placeholder_counter
        self.placeholder_counter -= 1
        AFTER_PH = self.placeholder_counter
        self.placeholder_counter -= 1
        
        self.loop_stack.append({'head': head_start, 'after': AFTER_PH})
        
        val_name = f"_val_{head_start}"
        
        try_body = [
            ast.Assign(
                targets=[ast.Name(id=val_name, ctx=ast.Store())],
                value=ast.Call(func=ast.Name(id='next', ctx=ast.Load()), args=[
                    ast.Subscript(
                        value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_ctx', ctx=ast.Load()),
                        slice=ast.Constant(value=iter_name),
                        ctx=ast.Load()
                    )
                ], keywords=[])
            )
        ]
        
        # Assign to target
        saved_stmts = self.current_stmts
        self.current_stmts = []
        
        # Create Assign node: target = val
        assign_node = ast.Assign(targets=[node.target], value=ast.Name(id=val_name, ctx=ast.Load()))
        self.visit_Assign(assign_node)
        
        assign_stmts = self.current_stmts
        self.current_stmts = saved_stmts
        
        try_body.extend(assign_stmts)
        
        # Jump to BODY
        try_body.append(ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
            value=ast.Constant(value=BODY_PH)
        ))
        try_body.append(ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[])))
        
        # Handler
        handler = ast.ExceptHandler(
            type=ast.Name(id='StopIteration', ctx=ast.Load()),
            name=None,
            body=[
                ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
                    value=ast.Constant(value=AFTER_PH)
                ),
                ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[]))
            ]
        )
        
        self.current_stmts.append(ast.Try(
            body=try_body,
            handlers=[handler],
            orelse=[],
            finalbody=[]
        ))
        
        self._flush_state(None) # Flush HEAD
        
        # 3. BODY State
        body_start = self.current_state
        for stmt in node.body:
            self.visit(stmt)
            
        # Jump back to HEAD
        if not (self.current_stmts and isinstance(self.current_stmts[-1], ast.Return)):
             self.current_stmts.append(ast.Assign(
                 targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
                 value=ast.Constant(value=head_start)
             ))
             self.current_stmts.append(ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[])))
             
        self._flush_state(None)
        
        self.loop_stack.pop()
        
        # 4. AFTER State
        after_start = self.current_state
        
        # 5. Backpatching
        # Fix HEAD (BODY_PH)
        head_stmts = self.states[head_start]
        try_node = head_stmts[-1]
        if isinstance(try_node, ast.Try):
            for stmt in try_node.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and stmt.value.value == BODY_PH:
                    stmt.value.value = body_start
                    
        # Fix AFTER_PH (in handler)
        if isinstance(try_node, ast.Try):
            for handler in try_node.handlers:
                for stmt in handler.body:
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and stmt.value.value == AFTER_PH:
                        stmt.value.value = after_start
                        
        # Global backpatch
        def patch_placeholders(ph, target):
            for sid, stmts in self.states.items():
                if not stmts: continue
                for stmt in stmts:
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                        t = stmt.targets[0]
                        if isinstance(t, ast.Attribute) and t.attr == '_state':
                            if isinstance(stmt.value, ast.Constant) and stmt.value.value == ph:
                                stmt.value.value = target
        
        patch_placeholders(AFTER_PH, after_start)

    def visit_Break(self, node):
        if not self.loop_stack:
            return
        after_ph = self.loop_stack[-1]['after']
        self.current_stmts.append(ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
            value=ast.Constant(value=after_ph)
        ))
        self.current_stmts.append(ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[])))

    def visit_Continue(self, node):
        if not self.loop_stack:
            return
        head = self.loop_stack[-1]['head']
        self.current_stmts.append(ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
            value=ast.Constant(value=head)
        ))
        self.current_stmts.append(ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='run', ctx=ast.Load()), args=[ast.Constant(value=None)], keywords=[])))

    def visit_Name(self, node):
        # Rewrite variable reads: x -> self._ctx['x']
        if isinstance(node.ctx, ast.Load) and node.id in self.varnames:
             return ast.Subscript(
                value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_ctx', ctx=ast.Load()),
                slice=ast.Constant(value=node.id),
                ctx=ast.Load()
            )
        return node

    def visit_FunctionDef(self, node):
        # We do NOT visit the body of the inner function, as it's a separate scope.
        # But we must ensure the function is stored in _ctx.
        self.current_stmts.append(node)
        
        if node.name in self.varnames:
            assign = ast.Assign(
                targets=[ast.Subscript(
                    value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_ctx', ctx=ast.Load()),
                    slice=ast.Constant(value=node.name),
                    ctx=ast.Store()
                )],
                value=ast.Name(id=node.name, ctx=ast.Load())
            )
            self.current_stmts.append(assign)

    def visit_AsyncFunctionDef(self, node):
        # Same as FunctionDef
        self.current_stmts.append(node)
        
        if node.name in self.varnames:
            assign = ast.Assign(
                targets=[ast.Subscript(
                    value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_ctx', ctx=ast.Load()),
                    slice=ast.Constant(value=node.name),
                    ctx=ast.Store()
                )],
                value=ast.Name(id=node.name, ctx=ast.Load())
            )
            self.current_stmts.append(assign)

    def visit_Return(self, node):
        # self._result = value
        # self._done = True
        # return
        if node.value:
            self.current_stmts.append(ast.Assign(
                targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_result', ctx=ast.Store())],
                value=self.visit(node.value)
            ))
        self.current_stmts.append(ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_done', ctx=ast.Store())],
            value=ast.Constant(value=True)
        ))
        self.current_stmts.append(ast.Return(value=None))

def compile_agent(func: Callable) -> Type[AgentMachine]:
    """
    Compiles a generator function into an AgentMachine class.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    
    compiler = CPSCompiler(func.__code__.co_varnames)
    
    # Process body
    for stmt in func_def.body:
        compiler.visit(stmt)
    compiler._flush_state(None) # Flush last block
    
    # Generate run method
    # def run(self, _input=None):
    #     if self._state == 0: ...
    #     elif self._state == 1: ...
    
    cases = []
    for state_id, stmts in compiler.states.items():
        if not stmts: continue
        cases.append(ast.If(
            test=ast.Compare(
                left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Load()),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=state_id)]
            ),
            body=stmts,
            orelse=[]
        ))
    
    # Chain ifs (elif)
    run_body = []
    if cases:
        current = cases[0]
        run_body.append(current)
        for case in cases[1:]:
            current.orelse = [case]
            current = case
            
    # Create class
    class_name = f"{func.__name__}_Machine"
    
    # Generate __init__
    # def __init__(self, arg1, arg2, ...):
    #     self._state = 0
    #     self._ctx = {'arg1': arg1, 'arg2': arg2, ...}
    #     self._done = False
    #     self._result = None
    
    # Extract args
    sig = inspect.signature(func)
    init_args = [ast.arg(arg='self')]
    ctx_keys = []
    
    for param in sig.parameters.values():
        init_args.append(ast.arg(arg=param.name))
        ctx_keys.append(param.name)
        
    # Build _ctx dict
    ctx_dict = ast.Dict(
        keys=[ast.Constant(value=k) for k in ctx_keys],
        values=[ast.Name(id=k, ctx=ast.Load()) for k in ctx_keys]
    )
    
    init_body = [
        ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_state', ctx=ast.Store())],
            value=ast.Constant(value=0)
        ),
        ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_ctx', ctx=ast.Store())],
            value=ctx_dict
        ),
        ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_done', ctx=ast.Store())],
            value=ast.Constant(value=False)
        ),
        ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_result', ctx=ast.Store())],
            value=ast.Constant(value=None)
        ),
        ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_stack', ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load())
        )
    ]
    
    # We need to update run() to handle stack
    # But run() is generated from the function body.
    # We can wrap the generated run body?
    # Or we can add a preamble to run().
    
    # Actually, if we use a stack, the `run` method needs to delegate to the top of the stack.
    # But `run` IS the code for this machine.
    # So we need a `dispatch` method?
    # Or `run` checks stack first.
    
    # If stack is not empty:
    #   sub = stack[-1]
    #   sig = sub.run(_input)
    #   if sub._done:
    #       stack.pop()
    #       _input = sub._result
    #       # Fall through to execute self's logic with result
    #   else:
    #       return sig
    
    # We need to inject this logic at the start of `run`.
    
    stack_logic = [
        ast.If(
            test=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_stack', ctx=ast.Load()),
            body=[
                ast.Assign(
                    targets=[ast.Name(id='sub', ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_stack', ctx=ast.Load()),
                        slice=ast.Constant(value=-1),
                        ctx=ast.Load()
                    )
                ),
                ast.Assign(
                    targets=[ast.Name(id='sig', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id='sub', ctx=ast.Load()), attr='run', ctx=ast.Load()),
                        args=[ast.Name(id='_input', ctx=ast.Load())],
                        keywords=[]
                    )
                ),
                ast.If(
                    test=ast.Attribute(value=ast.Name(id='sub', ctx=ast.Load()), attr='_done', ctx=ast.Load()),
                    body=[
                        ast.Expr(value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_stack', ctx=ast.Load()),
                                attr='pop',
                                ctx=ast.Load()
                            ),
                            args=[],
                            keywords=[]
                        )),
                        ast.Assign(
                            targets=[ast.Name(id='_input', ctx=ast.Store())],
                            value=ast.Attribute(value=ast.Name(id='sub', ctx=ast.Load()), attr='_result', ctx=ast.Load())
                        ),
                        ast.Expr(value=ast.Call(
                            func=ast.Name(id='print', ctx=ast.Load()),
                            args=[ast.Constant(value="DEBUG: Stack pop. Result:"), ast.Name(id='_input', ctx=ast.Load())],
                            keywords=[]
                        ))
                    ],
                    orelse=[
                        ast.Return(value=ast.Name(id='sig', ctx=ast.Load()))
                    ]
                )
            ],
            orelse=[]
        )
    ]
    
    run_body = stack_logic + run_body
    
    class_def = ast.ClassDef(
        name=class_name,
        bases=[ast.Name(id='AgentMachine', ctx=ast.Load())],
        keywords=[],
        body=[
            ast.FunctionDef(
                name='__init__',
                args=ast.arguments(
                    posonlyargs=[],
                    args=init_args,
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[] # We should handle defaults too, but for now strict
                ),
                body=init_body,
                decorator_list=[]
            ),
            ast.FunctionDef(
                name='run',
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg='self'), ast.arg(arg='_input', default=ast.Constant(value=None))],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[ast.Constant(value=None)]
                ),
                body=run_body,
                decorator_list=[]
            )
        ],
        decorator_list=[]
    )
    
    # Fix locations
    ast.fix_missing_locations(class_def)
    
    # Compile and execute definition
    module_ast = ast.Module(body=[class_def], type_ignores=[])
    
    # Execute in the function's globals to support shared state
    # We use a separate locals dict to capture the generated class without polluting globals
    exec_locals = {'AgentMachine': AgentMachine}
    
    exec(compile(module_ast, filename="<string>", mode="exec"), func.__globals__, exec_locals)
    return exec_locals[class_name]
