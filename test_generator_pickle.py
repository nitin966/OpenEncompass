import dill
import types

def my_generator():
    x = 1
    yield x
    x += 1
    yield x
    x += 1
    yield x

def test_pickle_generator():
    gen = my_generator()
    val1 = next(gen)
    print(f"Step 1: {val1}") # Should be 1
    
    # Pickle the suspended generator
    try:
        saved_state = dill.dumps(gen)
        print("Pickling successful!")
    except Exception as e:
        print(f"Pickling failed: {e}")
        return

    # Resume original
    val2 = next(gen)
    print(f"Original Step 2: {val2}") # Should be 2
    
    # Unpickle and resume copy
    gen_copy = dill.loads(saved_state)
    val2_copy = next(gen_copy)
    print(f"Copy Step 2: {val2_copy}") # Should be 2
    
    if val2 == val2_copy:
        print("SUCCESS: Generator state restored correctly.")
    else:
        print("FAILURE: State mismatch.")

if __name__ == "__main__":
    test_pickle_generator()
