# Code Translation Experiment

This example demonstrates the power of EnCompass by comparing two approaches to building an LLM-based Java → Python translator.

## Key Result: 40% Less Code with EnCompass

| Agent | Lines of Code | Description |
|-------|--------------|-------------|
| Base Agent | 493 | Manual state machine, retry loops, validation |
| EnCompass Agent | 292 | Same functionality with branchpoint/search |
| **Reduction** | **40%** | EnCompass eliminates boilerplate |

## Quick Start

```bash
# Run the complete experiment
python examples/code_translation/run_experiment.py --model qwen2.5:32b

# Or run agents individually
python examples/code_translation/base_translation_agent.py --model qwen2.5:32b
python examples/code_translation/encompass_translation_agent.py --model qwen2.5:32b
```

## Directory Structure

```
examples/code_translation/
├── input_java/              # jMinBpe Java tokenizer source
├── expected_output/         # Reference Python translations
├── output/
│   ├── base_agent/          # Base agent output
│   ├── encompass_agent/     # EnCompass agent output
│   └── report.md            # Comparison report
├── tests/
│   └── test_translation.py  # Quality verification (14 tests)
├── base_translation_agent.py      # 493 lines - state machine approach
├── encompass_translation_agent.py # 292 lines - EnCompass approach
└── run_experiment.py              # Runs both, tests, generates report
```

## What Gets Translated

The experiment translates the [jMinBpe](https://github.com/nitin966/jMinBpe) Java BPE tokenizer:

1. `TokenPair.java` → `token_pair.py`
2. `ByteUtils.java` → `byte_utils.py`
3. `TokenUtils.java` → `token_utils.py`
4. `Tokenizer.java` → `tokenizer.py`
5. `BasicTokenizer.java` → `basic_tokenizer.py`

## Why EnCompass is Simpler

**Base Agent requires:**
- Explicit `TranslationState` enum
- Manual `FileState` and `AgentState` dataclasses
- 5 step functions (`step_init`, `step_translate`, `step_validate`, `step_repair`, `step_test`)
- Manual state machine loop
- Explicit retry counters

**EnCompass Agent uses:**
- `branchpoint()` for nondeterministic choices (exploring translations)
- `record_score()` for feedback (valid translations score higher)
- Beam search handles retry logic automatically
- Linear, readable control flow

## Tests

All translations must pass 14 quality tests:
- TokenPair creation, equality, hashing
- ByteUtils concatenation
- TokenUtils rendering
- BasicTokenizer encode/decode roundtrip
