"""Debug script for CoW dict isolation issue."""

from core.compiler import compile_agent
from core.signals import branchpoint


@compile_agent
def agent_with_dict():
    data = {"x": 1}
    choice = branchpoint("choice")  # Implicit yield
    print(f"DEBUG: choice = {choice!r}")
    print(f"DEBUG: data = {data!r}")
    if choice == "A":
        data["y"] = "from_A"
    else:
        data["y"] = "from_B"
    return data


def run_to_completion(m, first_input=None):
    print(f"run_to_completion called with input: {first_input!r}")
    sig = m.run(first_input)
    print(f"  after first run: sig={sig}, done={m._done}, state={m._state}")
    iteration = 0
    while not m._done:
        iteration += 1
        sig = m.run(None)
        print(f"  iteration {iteration}: sig={sig}, done={m._done}, state={m._state}")
        if iteration > 20:
            print("  Breaking to avoid infinite loop")
            break
    return m._result


if __name__ == "__main__":
    machine_a = agent_with_dict()
    print("=== Initial run to branchpoint ===")
    sig = machine_a.run(None)
    print(f"At branchpoint, sig={sig}")
    print(f"machine_a._ctx: {dict(machine_a._ctx)}")

    machine_b = machine_a.snapshot()
    
    print("\n=== Running Branch A ===")
    try:
        result_a = run_to_completion(machine_a, "A")
        print(f"result_a = {result_a}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
