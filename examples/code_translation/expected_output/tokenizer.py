"""Tokenizer - Python translation of Tokenizer.java

A tokenizer class that implements Byte Pair Encoding (BPE) algorithm.
Provides functionality to encode text into tokens and decode tokens back to text,
as well as train new vocabularies and save/load models.
"""

from typing import Dict, List
from pathlib import Path

from token_pair import TokenPair
from byte_utils import concatenate


class Tokenizer:
    """Base tokenizer class implementing Byte Pair Encoding (BPE)."""
    
    def __init__(self):
        """Constructs a new Tokenizer with default settings."""
        self.merges: Dict[TokenPair, int] = {}
        self.pattern: str = ""
        self.special_tokens: Dict[str, int] = {}
        self.vocab: Dict[int, bytes] = self._build_vocab()
    
    def _build_vocab(self) -> Dict[int, bytes]:
        """Builds the vocabulary based on merges and special tokens.
        
        Returns:
            A map of token IDs to their byte representations.
        """
        vocab = {i: bytes([i]) for i in range(256)}
        
        for pair, idx in self.merges.items():
            vocab[idx] = concatenate(vocab[pair.first], vocab[pair.second])
        
        for token, idx in self.special_tokens.items():
            vocab[idx] = token.encode('utf-8')
        
        return vocab
    
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Trains the tokenizer on the given text to create a vocabulary of the specified size.
        
        Args:
            text: The training text.
            vocab_size: The desired vocabulary size.
            verbose: Whether to print progress information.
        """
        raise NotImplementedError("Train method not implemented")
    
    def encode(self, text: str) -> List[int]:
        """Encodes the given text into a list of token IDs.
        
        Args:
            text: The text to encode.
            
        Returns:
            A list of token IDs.
        """
        raise NotImplementedError("Encode method not implemented")
    
    def decode(self, ids: List[int]) -> str:
        """Decodes the given list of token IDs back into text.
        
        Args:
            ids: The list of token IDs to decode.
            
        Returns:
            The decoded text.
        """
        raise NotImplementedError("Decode method not implemented")
    
    def save(self, file_prefix: str) -> None:
        """Saves the current tokenizer model to files.
        
        Args:
            file_prefix: The prefix for the output files.
        """
        self._save_model(f"{file_prefix}.model")
        self._save_vocab(f"{file_prefix}.vocab")
    
    def _save_model(self, file_name: str) -> None:
        """Saves the model information to a file."""
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for token, idx in self.special_tokens.items():
                f.write(f"{token} {idx}\n")
            for pair, idx in self.merges.items():
                f.write(f"{pair.first} {pair.second}\n")
    
    def _save_vocab(self, file_name: str) -> None:
        """Saves the vocabulary information to a file for human inspection."""
        from token_utils import render_token
        
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        
        with open(file_name, 'w', encoding='utf-8') as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    pair = inverted_merges[idx]
                    s0 = render_token(self.vocab[pair.first])
                    s1 = render_token(self.vocab[pair.second])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
    
    def load(self, model_file: str) -> None:
        """Loads a tokenizer model from a file.
        
        Args:
            model_file: The file to load the model from.
        """
        with open(model_file, 'r', encoding='utf-8') as f:
            version = f.readline().strip()
            if version != "minbpe v1":
                raise ValueError("Incorrect version")
            
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            
            self._load_special_tokens(f, num_special)
            self._load_merges(f)
        
        self.vocab = self._build_vocab()
    
    def _load_special_tokens(self, f, num_special: int) -> None:
        """Loads special tokens from the file."""
        for _ in range(num_special):
            parts = f.readline().strip().split(' ')
            self.special_tokens[parts[0]] = int(parts[1])
    
    def _load_merges(self, f) -> None:
        """Loads merges from the file."""
        idx = 256
        for line in f:
            parts = line.strip().split(' ')
            idx1 = int(parts[0])
            idx2 = int(parts[1])
            self.merges[TokenPair(idx1, idx2)] = idx
            idx += 1
    
    def get_stats(self, ids: List[int], stats: Dict[TokenPair, int] = None) -> Dict[TokenPair, int]:
        """Counts occurrence of consecutive token pairs.
        
        Args:
            ids: List of token IDs.
            stats: Optional dict to update with counts.
            
        Returns:
            Dictionary of token pairs to their counts.
        """
        if stats is None:
            stats = {}
        for i in range(len(ids) - 1):
            pair = TokenPair(ids[i], ids[i + 1])
            stats[pair] = stats.get(pair, 0) + 1
        return stats
    
    def merge(self, ids: List[int], pair: TokenPair, new_id: int) -> List[int]:
        """Merges occurrences of a token pair in the id list.
        
        Args:
            ids: List of token IDs.
            pair: The pair to merge.
            new_id: The new ID for the merged pair.
            
        Returns:
            New list with merged pairs.
        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair.first and ids[i + 1] == pair.second:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
