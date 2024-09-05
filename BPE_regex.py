import regex as re
from BPE_basic import BytePairEncoding

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BytePairEncodingRegex(BytePairEncoding):
    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        
    def train(self, text, vocab_size, verbose = False) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)
        
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {} 
        vocab = {idx: bytes([idx]) for idx in range(256)} 
        for i in range(num_merges):
            
            stats = {}
            
            for chunk_ids in ids:
                super().get_stats(ids = chunk_ids, counts = stats)
                
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [super().merge(chunk_ids, pair, idx) for chunk_ids in ids]
            
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges 
        self.vocab = vocab   
        
        
if __name__ == "__main__":
    text = "Hello World how are you doing?"
    bpe = BytePairEncodingRegex()
    
    # Training process
    bpe.train(text, 300)
    
    # Encoding process
    encoded_ids = bpe.encode(text)
    print(f"Encoded IDs: {encoded_ids}")
    
    # Decoding process
    decoded_text = bpe.decode(encoded_ids)
    print(f"Decoded text: {decoded_text}")