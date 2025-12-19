"""TokenPair - Python translation of TokenPair.java

A simple class to represent a pair of token IDs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPair:
    """Represents a pair of token IDs."""
    
    first: int
    second: int
    
    def __str__(self) -> str:
        return f"({self.first}, {self.second})"
