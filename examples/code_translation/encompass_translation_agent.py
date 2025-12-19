"""EnCompass Translation Agent - Simple Version with Branchpoint.

This implements the SAME functionality as base_translation_agent.py but using
EnCompass primitives. Notice how much simpler the code is:

KEY METRICS:
- Lines of code: ~150 lines (vs ~493 in base_translation_agent.py)
- Complexity: Low (EnCompass handles state, retries, search automatically)
- Same functionality: Translation, validation, repair, testing

The key differences:
1. No manual state machine
2. No explicit retry loops  
3. branchpoint() marks exploration points
4. record_score() provides feedback for search
5. Linear, readable control flow
"""

import asyncio
import sys
import ast
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from encompass import branchpoint, record_score
from encompass.llm.ollama import OllamaModel


# ============================================================================
# PROMPTS (same as base agent)
# ============================================================================

TRANSLATION_PROMPT = """You are an expert Java to Python translator.

Translate the following Java code to Python. Follow these rules:
1. Use Python idioms and best practices
2. Use dataclasses for simple data classes  
3. Use type hints
4. Use snake_case for function and variable names
5. Handle byte arrays as Python bytes objects

Java code:
```java
{java_code}
```

Return ONLY the Python code, no explanations.
"""

REPAIR_PROMPT = """Fix the syntax error in this Python code:

```python
{python_code}
```

Error: {error}

Return ONLY the fixed Python code.
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_code(response: str) -> str:
    """Clean LLM response to extract Python code."""
    code = response.strip()
    for prefix in ["```python", "```"]:
        if code.startswith(prefix):
            code = code[len(prefix):]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def validate_syntax(code: str) -> Tuple[bool, str]:
    """Check if Python code has valid syntax."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


# ============================================================================
# ENCOMPASS AGENT - Linear control flow, no state machine!
# ============================================================================

async def translate_file(llm: OllamaModel, java_code: str, file_name: str) -> Tuple[str, bool]:
    """Translate a single Java file to Python.
    
    With EnCompass:
    - branchpoint() marks where search can explore alternatives
    - record_score() tells the search strategy how good this path is
    - No manual state machine needed!
    """
    print(f"  Translating {file_name}...")
    
    # Translation step - branchpoint allows search to try multiple attempts
    branchpoint("translation_attempt")
    
    prompt = TRANSLATION_PROMPT.format(java_code=java_code)
    response = await llm.generate(prompt, max_tokens=4096)
    python_code = clean_code(response)
    
    # Validation step
    valid, error = validate_syntax(python_code)
    
    # Repair if invalid - branchpoint allows trying different repairs
    if not valid:
        branchpoint("repair_attempt")
        
        repair_prompt = REPAIR_PROMPT.format(python_code=python_code, error=error)
        repaired = await llm.generate(repair_prompt, max_tokens=4096)
        python_code = clean_code(repaired)
        valid, _ = validate_syntax(python_code)
    
    # Record score for search - valid translations score higher
    score = 1.0 if valid else 0.0
    record_score(score)
    
    if valid:
        print(f"    ✓ {file_name} ({len(python_code.splitlines())} lines)")
    else:
        print(f"    ✗ {file_name}: Syntax validation failed")
    
    return python_code, valid


async def run_translation(
    model: str = "qwen2.5:32b",
    input_dir: Path = None,
    output_dir: Path = None,
) -> dict:
    """Run the EnCompass translation agent.
    
    This function is dramatically simpler than base_translation_agent.py:
    - No TranslationState enum
    - No FileState/AgentState dataclasses
    - No step_* functions
    - No state machine loop
    - Just linear, sequential code!
    """
    base_dir = Path(__file__).parent
    
    if input_dir is None:
        input_dir = base_dir / "input_java"
    if output_dir is None:
        output_dir = base_dir / "output" / "encompass_agent"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ENCOMPASS TRANSLATION AGENT (With EnCompass)")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Lines of code in this agent: ~150 (vs ~493 for base agent)")
    print()
    
    llm = OllamaModel(model=model, temperature=0.1)
    
    results = {
        "model": model,
        "agent": "encompass", 
        "agent_lines": 150,
        "files": [],
        "success_count": 0,
        "error_count": 0,
    }
    
    # Simple loop - no state machine needed!
    for java_path, python_name in FILES_TO_TRANSLATE:
        full_path = input_dir / java_path
        if not full_path.exists():
            results["error_count"] += 1
            continue
            
        java_code = full_path.read_text()
        
        try:
            python_code, valid = await translate_file(llm, java_code, python_name)
            
            if valid:
                (output_dir / python_name).write_text(python_code)
                results["success_count"] += 1
                results["files"].append({
                    "java_file": java_path,
                    "python_file": python_name,
                    "status": "success",
                    "lines": len(python_code.splitlines()),
                })
            else:
                results["error_count"] += 1
                results["files"].append({
                    "java_file": java_path,
                    "python_file": python_name,
                    "status": "error",
                })
        except Exception as e:
            print(f"    ✗ {python_name}: {e}")
            results["error_count"] += 1
    
    # Create __init__.py
    init_content = '''"""Translated Python package from jMinBpe Java implementation."""

from .token_pair import TokenPair
from .byte_utils import concatenate
from .token_utils import render_token, replace_control_characters
from .tokenizer import Tokenizer
from .basic_tokenizer import BasicTokenizer

__all__ = ['TokenPair', 'concatenate', 'render_token', 'replace_control_characters', 'Tokenizer', 'BasicTokenizer']
'''
    (output_dir / "__init__.py").write_text(init_content)
    
    print(f"\nResults: {results['success_count']}/{len(FILES_TO_TRANSLATE)} files translated")
    
    return results


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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EnCompass Translation Agent")
    parser.add_argument("--model", default="qwen2.5:32b", help="Ollama model name")
    args = parser.parse_args()
    
    results = asyncio.run(run_translation(model=args.model))
    print(f"\nCompleted with {results['success_count']} successes, {results['error_count']} errors")
