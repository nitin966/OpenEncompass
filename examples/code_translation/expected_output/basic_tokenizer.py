"""BasicTokenizer - Python translation of BasicTokenizer.java

A minimal byte-level Byte Pair Encoding (BPE) tokenizer.
This implementation follows the algorithmic approach of the GPT tokenizer
but does not handle regular expression splitting patterns or special tokens.
"""

from typing import Dict, List

from token_pair import TokenPair
from tokenizer import Tokenizer
from byte_utils import concatenate


class BasicTokenizer(Tokenizer):
    """A minimal byte-level BPE tokenizer."""
    
    def __init__(self):
        """Constructs a new BasicTokenizer."""
        super().__init__()
    
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Trains the tokenizer on the given text to create a vocabulary of the specified size.
        
        Args:
            text: The training text.
            vocab_size: The desired vocabulary size (must be at least 256).
            verbose: Whether to print progress information.
            
        Raises:
            ValueError: If vocab_size is less than 256.
        """
        if vocab_size < 256:
            raise ValueError("Vocab size must be at least 256")
        
        num_merges = vocab_size - 256
        text_bytes = text.encode('utf-8')
        ids = [b for b in text_bytes]  # Convert to list of unsigned ints
        
        merges: Dict[TokenPair, int] = {}
        vocab: Dict[int, bytes] = self._initialize_vocab()
        
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            
            best_pair = self._find_most_frequent_pair(stats)
            new_token_id = 256 + i
            
            ids = self.merge(ids, best_pair, new_token_id)
            merges[best_pair] = new_token_id
            vocab[new_token_id] = concatenate(vocab[best_pair.first], vocab[best_pair.second])
            
            if verbose:
                token_str = vocab[new_token_id].decode('utf-8', errors='replace')
                print(f"merge {i + 1}/{num_merges}: {best_pair} -> {new_token_id} "
                      f"({token_str}) had {stats[best_pair]} occurrences")
        
        self.merges = merges
        self.vocab = vocab
    
    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs back into text.
        
        Args:
            ids: The list of token IDs to decode.
            
        Returns:
            The decoded text.
        """
        text_bytes = b''.join(self.vocab[idx] for idx in ids)
        return text_bytes.decode('utf-8', errors='replace')
    
    def encode(self, text: str) -> List[int]:
        """Encodes the given text into a list of token IDs.
        
        Args:
            text: The text to encode.
            
        Returns:
            A list of token IDs.
        """
        text_bytes = text.encode('utf-8')
        ids = [b for b in text_bytes]
        
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            best_pair = self._find_best_pair(stats)
            if best_pair is None:
                break
            idx = self.merges[best_pair]
            ids = self.merge(ids, best_pair, idx)
        
        return ids
    
    def _initialize_vocab(self) -> Dict[int, bytes]:
        """Initializes the vocabulary with byte values.
        
        Returns:
            A map of token IDs (0-255) to their byte representations.
        """
        return {i: bytes([i]) for i in range(256)}
    
    def _find_most_frequent_pair(self, stats: Dict[TokenPair, int]) -> TokenPair:
        """Finds the most frequent pair in the statistics.
        
        Args:
            stats: The statistics of token pair occurrences.
            
        Returns:
            The most frequent TokenPair.
            
        Raises:
            ValueError: If no pairs are found.
        """
        if not stats:
            raise ValueError("No pairs found")
        return max(stats.keys(), key=lambda p: stats[p])
    
    def _find_best_pair(self, stats: Dict[TokenPair, int]) -> TokenPair:
        """Finds the best pair for merging based on the merge index.
        
        Args:
            stats: The statistics of token pair occurrences.
            
        Returns:
            The best TokenPair for merging, or None if no merge is possible.
        """
        # Find the pair with the lowest merge index (earliest in vocabulary)
        pairs_in_vocab = [(p, self.merges.get(p, float('inf'))) for p in stats.keys()]
        pairs_in_vocab = [(p, idx) for p, idx in pairs_in_vocab if p in self.merges]
        
        if not pairs_in_vocab:
            return None
        
        return min(pairs_in_vocab, key=lambda x: x[1])[0]
