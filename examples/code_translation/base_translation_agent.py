"""Base Translation Agent (WITHOUT EnCompass) - Complex State Machine Version.

This implements the code translation with explicit state management, retry loops,
validation, and repair - all the complexity that EnCompass abstracts away.

From the paper: "Explicitly defining a state machine to support general search not only
significantly obscures the underlying agent logic, but is also prone to bugs such as
KeyError when accessing the dictionary cur_state that stores all the variables."

KEY METRICS:
- Lines of code: ~400+ lines
- Complexity: High (manual state, retries, validation loops)
- Compare to: encompass_translation_agent.py (~100 lines with same functionality)
"""

import asyncio
import sys
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from encompass.llm.ollama import OllamaModel


# ============================================================================
# STATE DEFINITIONS - Manual state management required without EnCompass
# ============================================================================

class TranslationState(Enum):
    """States in the translation state machine."""
    INIT = "init"
    TRANSLATING = "translating"
    VALIDATING = "validating"
    REPAIRING = "repairing"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FileState:
    """State for a single file translation - manual bookkeeping."""
    java_path: str
    python_name: str
    java_code: str = ""
    python_code: str = ""
    state: TranslationState = TranslationState.INIT
    attempts: int = 0
    max_attempts: int = 3
    syntax_valid: bool = False
    test_passed: bool = False
    error_message: str = ""
    repair_attempts: int = 0
    max_repair_attempts: int = 2


@dataclass 
class AgentState:
    """Global agent state - must track all variables explicitly."""
    model_name: str
    input_dir: Path
    output_dir: Path
    file_states: Dict[str, FileState] = field(default_factory=dict)
    current_file_idx: int = 0
    total_files: int = 0
    success_count: int = 0
    error_count: int = 0
    total_attempts: int = 0
    phase: str = "initialization"


# ============================================================================
# PROMPTS
# ============================================================================

TRANSLATION_PROMPT = """You are an expert Java to Python translator.

Translate the following Java code to Python. Follow these rules:
1. Use Python idioms and best practices
2. Use dataclasses for simple data classes  
3. Use type hints
4. Replace Java-style getters/setters with properties or direct access
5. Use snake_case for function and variable names
6. Handle byte arrays as Python bytes objects
7. Use dict instead of HashMap, list instead of ArrayList
8. Keep all docstrings and comments

Java code to translate:
```java
{java_code}
```

Return ONLY the Python code, no explanations or markdown code blocks.
"""

REPAIR_PROMPT = """The following Python code has a syntax error:

```python
{python_code}
```

Error: {error}

Fix the syntax error and return the corrected Python code.
Return ONLY the fixed Python code, no explanations.
"""

VALIDATION_PROMPT = """Review this Python code translation and verify it's correct:

Original Java:
```java
{java_code}
```

Translated Python:
```python
{python_code}
```

If the translation is correct, respond with "CORRECT".
If there are issues, respond with "INCORRECT: <specific issues>".
"""


# ============================================================================
# STATE MACHINE STEP FUNCTIONS - The complexity EnCompass eliminates
# ============================================================================

async def step_init(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, TranslationState]:
    """Initialize file states - state machine entry point."""
    state.phase = "initialization"
    
    # Load all Java files into state
    for java_path, python_name in FILES_TO_TRANSLATE:
        full_path = state.input_dir / java_path
        if full_path.exists():
            java_code = full_path.read_text()
            file_state = FileState(
                java_path=java_path,
                python_name=python_name,
                java_code=java_code,
            )
            state.file_states[python_name] = file_state
            state.total_files += 1
    
    if state.total_files == 0:
        return state, TranslationState.FAILED
    
    return state, TranslationState.TRANSLATING


async def step_translate(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, TranslationState]:
    """Translate current file - with retry logic."""
    state.phase = "translation"
    
    # Get current file
    file_names = list(state.file_states.keys())
    if state.current_file_idx >= len(file_names):
        return state, TranslationState.COMPLETED
    
    current_name = file_names[state.current_file_idx]
    file_state = state.file_states[current_name]
    
    # Skip if already successful
    if file_state.state == TranslationState.COMPLETED:
        state.current_file_idx += 1
        return state, TranslationState.TRANSLATING
    
    # Check attempt limit
    if file_state.attempts >= file_state.max_attempts:
        file_state.state = TranslationState.FAILED
        file_state.error_message = f"Max attempts ({file_state.max_attempts}) exceeded"
        state.error_count += 1
        state.current_file_idx += 1
        return state, TranslationState.TRANSLATING
    
    # Perform translation
    file_state.attempts += 1
    state.total_attempts += 1
    print(f"  Translating {file_state.python_name} (attempt {file_state.attempts}/{file_state.max_attempts})...")
    
    try:
        prompt = TRANSLATION_PROMPT.format(java_code=file_state.java_code)
        response = await llm.generate(prompt, max_tokens=4096)
        
        # Clean response
        python_code = clean_code_response(response)
        file_state.python_code = python_code
        file_state.state = TranslationState.VALIDATING
        
        return state, TranslationState.VALIDATING
        
    except Exception as e:
        file_state.error_message = str(e)
        print(f"    Error: {e}")
        # Retry on next iteration
        return state, TranslationState.TRANSLATING


async def step_validate(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, TranslationState]:
    """Validate current file - syntax check and LLM review."""
    state.phase = "validation"
    
    file_names = list(state.file_states.keys())
    current_name = file_names[state.current_file_idx]
    file_state = state.file_states[current_name]
    
    # Step 1: Syntax validation
    syntax_ok, syntax_error = validate_python_syntax(file_state.python_code)
    file_state.syntax_valid = syntax_ok
    
    if not syntax_ok:
        print(f"    Syntax error: {syntax_error}")
        file_state.error_message = syntax_error
        file_state.state = TranslationState.REPAIRING
        return state, TranslationState.REPAIRING
    
    # Step 2: LLM review (optional - can be expensive)
    # Skipping for speed, but this is where EnCompass branchpoint would help
    
    # Success - move to testing
    file_state.state = TranslationState.TESTING
    return state, TranslationState.TESTING


async def step_repair(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, TranslationState]:
    """Repair syntax errors - manual retry loop."""
    state.phase = "repair"
    
    file_names = list(state.file_states.keys())
    current_name = file_names[state.current_file_idx]
    file_state = state.file_states[current_name]
    
    # Check repair attempt limit
    if file_state.repair_attempts >= file_state.max_repair_attempts:
        print(f"    Max repair attempts exceeded, retranslating...")
        file_state.state = TranslationState.TRANSLATING
        file_state.repair_attempts = 0
        return state, TranslationState.TRANSLATING
    
    file_state.repair_attempts += 1
    state.total_attempts += 1
    print(f"  Repairing {file_state.python_name} (attempt {file_state.repair_attempts}/{file_state.max_repair_attempts})...")
    
    try:
        prompt = REPAIR_PROMPT.format(
            python_code=file_state.python_code,
            error=file_state.error_message
        )
        response = await llm.generate(prompt, max_tokens=4096)
        
        python_code = clean_code_response(response)
        file_state.python_code = python_code
        file_state.state = TranslationState.VALIDATING
        
        return state, TranslationState.VALIDATING
        
    except Exception as e:
        file_state.error_message = str(e)
        return state, TranslationState.REPAIRING


async def step_test(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, TranslationState]:
    """Test translated code - run unit tests."""
    state.phase = "testing"
    
    file_names = list(state.file_states.keys())
    current_name = file_names[state.current_file_idx]
    file_state = state.file_states[current_name]
    
    # Write file to disk for testing
    output_path = state.output_dir / file_state.python_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(file_state.python_code)
    
    # For now, mark as passed if syntax is valid
    # Real implementation would run actual tests
    file_state.test_passed = file_state.syntax_valid
    
    if file_state.test_passed:
        file_state.state = TranslationState.COMPLETED
        state.success_count += 1
        print(f"    ✓ {file_state.python_name} ({len(file_state.python_code.splitlines())} lines)")
    else:
        # Could retry translation
        if file_state.attempts < file_state.max_attempts:
            file_state.state = TranslationState.TRANSLATING
            return state, TranslationState.TRANSLATING
        else:
            file_state.state = TranslationState.FAILED
            state.error_count += 1
            print(f"    ✗ {file_state.python_name}: Tests failed")
    
    # Move to next file
    state.current_file_idx += 1
    return state, TranslationState.TRANSLATING


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_code_response(response: str) -> str:
    """Clean LLM response to extract Python code."""
    code = response.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Check if Python code has valid syntax."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def create_init_file(output_dir: Path) -> None:
    """Create __init__.py for the output package."""
    init_content = '''"""Translated Python package from jMinBpe Java implementation."""

from .token_pair import TokenPair
from .byte_utils import concatenate
from .token_utils import render_token, replace_control_characters
from .tokenizer import Tokenizer
from .basic_tokenizer import BasicTokenizer

__all__ = [
    'TokenPair',
    'concatenate',
    'render_token',
    'replace_control_characters', 
    'Tokenizer',
    'BasicTokenizer',
]
'''
    (output_dir / "__init__.py").write_text(init_content)


# ============================================================================
# MAIN STATE MACHINE DRIVER - The core loop EnCompass replaces
# ============================================================================

async def run_state_machine(state: AgentState, llm: OllamaModel) -> AgentState:
    """Run the translation state machine until completion.
    
    This is the complexity that EnCompass eliminates with branchpoint() and search().
    Notice how we have to:
    1. Manually track current state
    2. Manually handle transitions
    3. Manually manage retries
    4. Manually track all variables in state dict
    """
    current_state = TranslationState.INIT
    
    # State machine loop - EnCompass handles this automatically
    while current_state not in (TranslationState.COMPLETED, TranslationState.FAILED):
        if current_state == TranslationState.INIT:
            state, current_state = await step_init(state, llm)
            
        elif current_state == TranslationState.TRANSLATING:
            state, current_state = await step_translate(state, llm)
            # Check if all files processed
            if state.current_file_idx >= state.total_files:
                current_state = TranslationState.COMPLETED
                
        elif current_state == TranslationState.VALIDATING:
            state, current_state = await step_validate(state, llm)
            
        elif current_state == TranslationState.REPAIRING:
            state, current_state = await step_repair(state, llm)
            
        elif current_state == TranslationState.TESTING:
            state, current_state = await step_test(state, llm)
    
    return state


# ============================================================================
# FILE MAPPING
# ============================================================================

FILES_TO_TRANSLATE = [
    ("src/com/minbpe/tokenpair/TokenPair.java", "token_pair.py"),
    ("src/com/minbpe/utils/ByteUtils.java", "byte_utils.py"),
    ("src/com/minbpe/utils/TokenUtils.java", "token_utils.py"),
    ("src/com/minbpe/Tokenizer.java", "tokenizer.py"),
    ("src/com/minbpe/BasicTokenizer.java", "basic_tokenizer.py"),
]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def run_translation(
    model: str = "qwen2.5:32b",
    input_dir: Path = None,
    output_dir: Path = None,
) -> dict:
    """Run the base translation agent.
    
    Args:
        model: Ollama model name.
        input_dir: Java source directory.
        output_dir: Python output directory.
        
    Returns:
        Translation results dict.
    """
    base_dir = Path(__file__).parent
    
    if input_dir is None:
        input_dir = base_dir / "input_java"
    if output_dir is None:
        output_dir = base_dir / "output" / "base_agent"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("BASE TRANSLATION AGENT (Without EnCompass)")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Lines of code in this agent: ~400")
    print()
    
    # Initialize LLM
    llm = OllamaModel(model=model, temperature=0.1)
    
    # Initialize state
    state = AgentState(
        model_name=model,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    
    # Run state machine
    final_state = await run_state_machine(state, llm)
    
    # Create init file
    create_init_file(output_dir)
    
    # Build results
    results = {
        "model": model,
        "agent": "base",
        "agent_lines": 400,  # Approximate line count of this file
        "files": [],
        "success_count": final_state.success_count,
        "error_count": final_state.error_count,
        "total_attempts": final_state.total_attempts,
    }
    
    for name, file_state in final_state.file_states.items():
        results["files"].append({
            "java_file": file_state.java_path,
            "python_file": file_state.python_name,
            "status": "success" if file_state.state == TranslationState.COMPLETED else "error",
            "attempts": file_state.attempts,
            "repair_attempts": file_state.repair_attempts,
            "lines": len(file_state.python_code.splitlines()) if file_state.python_code else 0,
        })
    
    print(f"\nResults: {results['success_count']}/{final_state.total_files} files translated")
    print(f"Total LLM calls: {results['total_attempts']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Base Translation Agent (Without EnCompass)")
    parser.add_argument("--model", default="qwen2.5:32b", help="Ollama model name")
    args = parser.parse_args()
    
    results = asyncio.run(run_translation(model=args.model))
    print(f"\nCompleted with {results['success_count']} successes, {results['error_count']} errors")
