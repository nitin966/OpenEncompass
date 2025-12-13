"""
Custom exceptions for the CPS compiler.

These exceptions provide helpful error messages with suggestions for fixes.
"""


class CompilationError(Exception):
    """
    Raised when the CPS compiler encounters an unsupported Python feature.

    Always includes:
    - What went wrong
    - Why it went wrong
    - How to fix it
    """


class ClosureCaptureError(CompilationError):
    """
    Raised when trying to use a closure variable in a compiled agent.

    Example:
        def create_agent(x):
            @compile
            def agent():
                return x  # <-- ClosureCaptureError here
            return agent
    """

    def __init__(self, var_name):
        message = (
            f"Variable '{var_name}' not defined.\n\n"
            "The CPS compiler doesn't capture closure variables.\n"
            "Fix: Pass '{var_name}' as an argument instead:\n\n"
            "    @compile\n"
            "    def agent(x):\n"
            "        return x\n"
            "    return lambda: agent(x_value)\n\n"
            "See docs/CPS_LIMITATIONS.md for details."
        )
        super().__init__(message)


class TupleUnpackingError(CompilationError):
    """
    Raised when using _ in tuple unpacking.

    Example:
        _, x = some_tuple  # <-- TupleUnpackingError here
    """

    def __init__(self):
        message = (
            "Tuple unpacking with '_' is not supported.\n\n"
            "Fix: Use named variables instead:\n\n"
            "    # Don't do this:\n"
            "    _, x = values\n\n"
            "    # Do this:\n"
            "    parts = values\n"
            "    x = parts[1]\n\n"
            "See docs/CPS_LIMITATIONS.md for details."
        )
        super().__init__(message)
