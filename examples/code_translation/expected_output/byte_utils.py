"""ByteUtils - Python translation of ByteUtils.java

Utility functions for byte array operations.
"""


def concatenate(a: bytes, b: bytes) -> bytes:
    """Concatenates two byte objects.
    
    Args:
        a: The first byte object.
        b: The second byte object.
        
    Returns:
        The concatenated byte object.
    """
    return a + b
