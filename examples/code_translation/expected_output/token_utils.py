"""TokenUtils - Python translation of TokenUtils.java

Utility functions for token-related operations.
"""

import unicodedata


def render_token(token: bytes) -> str:
    """Renders a token for human-readable output, escaping control characters.
    
    Args:
        token: The token bytes to render.
        
    Returns:
        A string representation of the token.
    """
    s = token.decode('utf-8', errors='replace')
    return replace_control_characters(s)


def replace_control_characters(s: str) -> str:
    """Replaces control characters in a string with their Unicode escape sequences.
    
    Args:
        s: The input string.
        
    Returns:
        The string with control characters replaced.
    """
    result = []
    for ch in s:
        if unicodedata.category(ch) == 'Cc':  # Control character
            result.append(f'\\u{ord(ch):04x}')
        else:
            result.append(ch)
    return ''.join(result)
