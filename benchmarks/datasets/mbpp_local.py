# MBPP-Style Local Dataset - Python Code Generation
# Curated coding problems for benchmarking


PROBLEMS = [
    {
        "id": 1,
        "task": "Write a function to find the maximum of three numbers.",
        "prompt": "def max_of_three(a, b, c):",
        "canonical_solution": "    return max(a, b, c)",
        "test_cases": [
            "assert max_of_three(10, 20, 30) == 30",
            "assert max_of_three(5, 5, 5) == 5",
            "assert max_of_three(-1, -2, -3) == -1",
        ],
        "difficulty": "easy",
    },
    {
        "id": 2,
        "task": "Write a function to check if a number is even.",
        "prompt": "def is_even(n):",
        "canonical_solution": "    return n % 2 == 0",
        "test_cases": [
            "assert is_even(4) == True",
            "assert is_even(7) == False",
            "assert is_even(0) == True",
        ],
        "difficulty": "easy",
    },
    {
        "id": 3,
        "task": "Write a function to reverse a string.",
        "prompt": "def reverse_string(s):",
        "canonical_solution": "    return s[::-1]",
        "test_cases": [
            "assert reverse_string('hello') == 'olleh'",
            "assert reverse_string('') == ''",
            "assert reverse_string('a') == 'a'",
        ],
        "difficulty": "easy",
    },
    {
        "id": 4,
        "task": "Write a function to compute the factorial of a number.",
        "prompt": "def factorial(n):",
        "canonical_solution": "    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        "test_cases": [
            "assert factorial(5) == 120",
            "assert factorial(0) == 1",
            "assert factorial(1) == 1",
        ],
        "difficulty": "medium",
    },
    {
        "id": 5,
        "task": "Write a function to find the sum of a list of numbers.",
        "prompt": "def sum_list(numbers):",
        "canonical_solution": "    return sum(numbers)",
        "test_cases": [
            "assert sum_list([1, 2, 3, 4]) == 10",
            "assert sum_list([]) == 0",
            "assert sum_list([-1, 1]) == 0",
        ],
        "difficulty": "easy",
    },
    {
        "id": 6,
        "task": "Write a function to check if a string is a palindrome.",
        "prompt": "def is_palindrome(s):",
        "canonical_solution": "    return s == s[::-1]",
        "test_cases": [
            "assert is_palindrome('racecar') == True",
            "assert is_palindrome('hello') == False",
            "assert is_palindrome('') == True",
        ],
        "difficulty": "easy",
    },
    {
        "id": 7,
        "task": "Write a function to find the nth Fibonacci number.",
        "prompt": "def fibonacci(n):",
        "canonical_solution": "    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "test_cases": [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(6) == 8",
        ],
        "difficulty": "medium",
    },
    {
        "id": 8,
        "task": "Write a function to count vowels in a string.",
        "prompt": "def count_vowels(s):",
        "canonical_solution": "    return sum(1 for c in s.lower() if c in 'aeiou')",
        "test_cases": [
            "assert count_vowels('hello') == 2",
            "assert count_vowels('xyz') == 0",
            "assert count_vowels('AEIOU') == 5",
        ],
        "difficulty": "easy",
    },
    {
        "id": 9,
        "task": "Write a function to remove duplicates from a list.",
        "prompt": "def remove_duplicates(lst):",
        "canonical_solution": "    return list(set(lst))",
        "test_cases": [
            "assert sorted(remove_duplicates([1, 2, 2, 3])) == [1, 2, 3]",
            "assert sorted(remove_duplicates([1, 1, 1])) == [1]",
            "assert remove_duplicates([]) == []",
        ],
        "difficulty": "easy",
    },
    {
        "id": 10,
        "task": "Write a function to find the greatest common divisor of two numbers.",
        "prompt": "def gcd(a, b):",
        "canonical_solution": "    while b:\n        a, b = b, a % b\n    return a",
        "test_cases": [
            "assert gcd(48, 18) == 6",
            "assert gcd(7, 5) == 1",
            "assert gcd(100, 50) == 50",
        ],
        "difficulty": "medium",
    },
    {
        "id": 11,
        "task": "Write a function to check if a number is prime.",
        "prompt": "def is_prime(n):",
        "canonical_solution": "    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "test_cases": [
            "assert is_prime(7) == True",
            "assert is_prime(4) == False",
            "assert is_prime(1) == False",
        ],
        "difficulty": "medium",
    },
    {
        "id": 12,
        "task": "Write a function to flatten a nested list.",
        "prompt": "def flatten(lst):",
        "canonical_solution": "    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
        "test_cases": [
            "assert flatten([1, [2, 3], [4, [5]]]) == [1, 2, 3, 4, 5]",
            "assert flatten([]) == []",
            "assert flatten([1, 2, 3]) == [1, 2, 3]",
        ],
        "difficulty": "hard",
    },
    {
        "id": 13,
        "task": "Write a function to merge two sorted lists.",
        "prompt": "def merge_sorted(lst1, lst2):",
        "canonical_solution": "    result = []\n    i = j = 0\n    while i < len(lst1) and j < len(lst2):\n        if lst1[i] <= lst2[j]:\n            result.append(lst1[i])\n            i += 1\n        else:\n            result.append(lst2[j])\n            j += 1\n    result.extend(lst1[i:])\n    result.extend(lst2[j:])\n    return result",
        "test_cases": [
            "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]",
            "assert merge_sorted([], [1, 2]) == [1, 2]",
            "assert merge_sorted([1], []) == [1]",
        ],
        "difficulty": "medium",
    },
    {
        "id": 14,
        "task": "Write a function to find the mode (most common element) in a list.",
        "prompt": "def mode(lst):",
        "canonical_solution": "    from collections import Counter\n    return Counter(lst).most_common(1)[0][0]",
        "test_cases": [
            "assert mode([1, 2, 2, 3]) == 2",
            "assert mode([1, 1, 1, 2, 2]) == 1",
            "assert mode([5]) == 5",
        ],
        "difficulty": "medium",
    },
    {
        "id": 15,
        "task": "Write a function to rotate a list by k positions.",
        "prompt": "def rotate_list(lst, k):",
        "canonical_solution": "    if not lst:\n        return lst\n    k = k % len(lst)\n    return lst[-k:] + lst[:-k]",
        "test_cases": [
            "assert rotate_list([1, 2, 3, 4], 2) == [3, 4, 1, 2]",
            "assert rotate_list([1, 2, 3], 0) == [1, 2, 3]",
            "assert rotate_list([], 5) == []",
        ],
        "difficulty": "medium",
    },
]


def get_problems(num_problems=None, difficulty=None):
    """
    Get problems from the dataset.

    Args:
        num_problems: Number of problems to return (None = all)
        difficulty: Filter by difficulty ("easy", "medium", "hard", or None for all)

    Returns:
        List of problem dicts
    """
    problems = PROBLEMS

    if difficulty:
        problems = [p for p in problems if p.get("difficulty") == difficulty]

    if num_problems:
        problems = problems[:num_problems]

    return problems


def evaluate_code(code: str, test_cases: list[str]) -> tuple:
    """
    Evaluate generated code against test cases.

    Args:
        code: Generated Python code
        test_cases: List of assertion strings

    Returns:
        (passed: bool, num_passed: int, total: int, error: str or None)
    """
    try:
        # Execute the code in a safe namespace
        namespace = {}
        exec(code, namespace)

        # Run each test case
        passed = 0
        for test in test_cases:
            try:
                exec(test, namespace)
                passed += 1
            except AssertionError:
                continue
            except Exception as e:
                return (False, passed, len(test_cases), f"Test error: {e}")

        return (passed == len(test_cases), passed, len(test_cases), None)

    except SyntaxError as e:
        return (False, 0, len(test_cases), f"Syntax error: {e}")
    except Exception as e:
        return (False, 0, len(test_cases), f"Runtime error: {e}")
