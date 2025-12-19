"""Test cases for verifying translated Python code quality.

These tests verify that the translated tokenizer works correctly.
The same tests are run against both base agent and EnCompass agent outputs.
"""

import sys
import os
import unittest
from pathlib import Path


class TestTokenPair(unittest.TestCase):
    """Test the TokenPair class."""
    
    def test_creation_and_access(self):
        """Test TokenPair creation and attribute access."""
        from token_pair import TokenPair
        
        pair = TokenPair(first=1, second=2)
        self.assertEqual(pair.first, 1)
        self.assertEqual(pair.second, 2)
    
    def test_equality(self):
        """Test TokenPair equality comparisons."""
        from token_pair import TokenPair
        
        pair1 = TokenPair(1, 2)
        pair2 = TokenPair(1, 2)
        pair3 = TokenPair(2, 1)
        
        self.assertEqual(pair1, pair2)
        self.assertNotEqual(pair1, pair3)
    
    def test_hashable(self):
        """Test TokenPair can be used as dict key."""
        from token_pair import TokenPair
        
        pair = TokenPair(1, 2)
        d = {pair: "test"}
        self.assertEqual(d[pair], "test")
    
    def test_str(self):
        """Test TokenPair string representation."""
        from token_pair import TokenPair
        
        pair = TokenPair(1, 2)
        self.assertEqual(str(pair), "(1, 2)")


class TestByteUtils(unittest.TestCase):
    """Test the byte utility functions."""
    
    def test_concatenate(self):
        """Test byte concatenation."""
        from byte_utils import concatenate
        
        result = concatenate(b"hello", b"world")
        self.assertEqual(result, b"helloworld")
    
    def test_concatenate_empty(self):
        """Test concatenation with empty bytes."""
        from byte_utils import concatenate
        
        self.assertEqual(concatenate(b"", b"test"), b"test")
        self.assertEqual(concatenate(b"test", b""), b"test")
        self.assertEqual(concatenate(b"", b""), b"")


class TestTokenUtils(unittest.TestCase):
    """Test the token utility functions."""
    
    def test_render_token(self):
        """Test rendering a normal token."""
        from token_utils import render_token
        
        result = render_token(b"hello")
        self.assertEqual(result, "hello")
    
    def test_replace_control_characters(self):
        """Test control character replacement."""
        from token_utils import replace_control_characters
        
        result = replace_control_characters("hello\nworld")
        self.assertIn("\\u000a", result)  # \n is 0x0A


class TestBasicTokenizer(unittest.TestCase):
    """Test the BasicTokenizer class - the main quality tests."""
    
    def test_encode_decode_roundtrip_simple(self):
        """Test that encode/decode roundtrip preserves text."""
        from basic_tokenizer import BasicTokenizer
        
        tokenizer = BasicTokenizer()
        text = "hello world"
        tokenizer.train(text, vocab_size=260, verbose=False)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        self.assertEqual(decoded, text)
    
    def test_encode_decode_roundtrip_complex(self):
        """Test roundtrip with more complex text."""
        from basic_tokenizer import BasicTokenizer
        
        tokenizer = BasicTokenizer()
        text = "The quick brown fox jumps over the lazy dog. 123 !@#"
        tokenizer.train(text, vocab_size=280, verbose=False)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        self.assertEqual(decoded, text)
    
    def test_vocab_size_validation(self):
        """Test that vocab_size < 256 raises error."""
        from basic_tokenizer import BasicTokenizer
        
        tokenizer = BasicTokenizer()
        with self.assertRaises(ValueError):
            tokenizer.train("test", vocab_size=100)
    
    def test_encode_produces_integers(self):
        """Test that encode returns a list of integers."""
        from basic_tokenizer import BasicTokenizer
        
        tokenizer = BasicTokenizer()
        text = "hello"
        tokenizer.train(text, vocab_size=260)
        
        encoded = tokenizer.encode(text)
        
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(x, int) for x in encoded))
    
    def test_train_creates_merges(self):
        """Test that training creates merge rules."""
        from basic_tokenizer import BasicTokenizer
        
        tokenizer = BasicTokenizer()
        text = "aaabbb"  # Should find 'aa' and 'bb' pairs
        tokenizer.train(text, vocab_size=260, verbose=False)
        
        self.assertGreater(len(tokenizer.merges), 0)
        self.assertGreater(len(tokenizer.vocab), 256)
    
    def test_multiple_encodes(self):
        """Test encoding different strings with the same tokenizer."""
        from basic_tokenizer import BasicTokenizer
        
        tokenizer = BasicTokenizer()
        training_text = "hello world hello world"
        tokenizer.train(training_text, vocab_size=270, verbose=False)
        
        # Encode various strings
        texts = ["hello", "world", "hello world"]
        for text in texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            self.assertEqual(decoded, text)


def run_tests(output_dir: str) -> tuple:
    """Run tests against the output directory.
    
    Args:
        output_dir: Path to the directory containing translated Python files.
        
    Returns:
        Tuple of (success_count, failure_count, error_messages)
    """
    # Add output dir to path so imports work
    if output_dir not in sys.path:
        sys.path.insert(0, output_dir)
    
    # Reload modules to pick up new versions
    for mod_name in ['token_pair', 'byte_utils', 'token_utils', 'tokenizer', 'basic_tokenizer']:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    
    # Run the tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTokenPair))
    suite.addTests(loader.loadTestsFromTestCase(TestByteUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicTokenizer))
    
    # Run with result capture
    result = unittest.TestResult()
    suite.run(result)
    
    success_count = result.testsRun - len(result.failures) - len(result.errors)
    failure_count = len(result.failures) + len(result.errors)
    
    error_messages = []
    for test, traceback in result.failures:
        error_messages.append(f"FAIL: {test}\n{traceback}")
    for test, traceback in result.errors:
        error_messages.append(f"ERROR: {test}\n{traceback}")
    
    return success_count, failure_count, error_messages


if __name__ == "__main__":
    # When run directly, test the expected_output directory
    expected_output_dir = str(Path(__file__).parent.parent / "expected_output")
    print(f"Testing: {expected_output_dir}")
    
    success, failures, errors = run_tests(expected_output_dir)
    print(f"\nResults: {success} passed, {failures} failed")
    
    if errors:
        print("\nErrors:")
        for err in errors:
            print(err)
    
    sys.exit(0 if failures == 0 else 1)
