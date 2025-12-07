class AgentMachine:
    """
    Base class for compiled state machines.
    """
    def __init__(self):
        self._state = 0
        self._ctx = {} # Local variables
        self._done = False
        self._result = None

    def run(self, _input=None):
        """
        Executes one step of the machine.
        Returns: (signal, done)
        """
        raise NotImplementedError

    def save(self):
        """Returns a pickleable state dict."""
        return {
            "_state": self._state,
            "_ctx": self._ctx,
            "_done": self._done,
            "_result": self._result
        }
    def load(self, state):
        """Restores state."""
        self._state = state["_state"]
        self._ctx = state["_ctx"]
        self._done = state["_done"]
        self._result = state["_result"]

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

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Yield):
            yield_val = self.visit(node.value.value) if node.value.value else None
            self.current_stmts.append(ast.Return(value=yield_val))
            
            next_state = self.current_state + 1
            self._flush_state(next_state)
        else:
            self.current_stmts.append(self.generic_visit(node))

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
    class_def = ast.ClassDef(
        name=class_name,
        bases=[ast.Name(id='AgentMachine', ctx=ast.Load())],
        keywords=[],
        body=[
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
    exec_globals = globals().copy()
    exec_globals['AgentMachine'] = AgentMachine
    # We need to inject globals from the function's module
    exec_globals.update(func.__globals__)
    
    exec(compile(module_ast, filename="<string>", mode="exec"), exec_globals)
    return exec_globals[class_name]
